# 注入新模型到 verl-omni（零侵入 3+1 步）

> 用当前三层分治架构写就。训练侧完全零侵入（3 步），
> rollout 侧需要 gate patch GP-004（第 4 步，一次性）。
> 本指南是权威版本——旧文档声称"不需要 entry_points / monkey-patch"是错误的。

---

## 机制总览（前置必读）

```
启动脚本:
  export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
        ↓
verl-omni 在每个 Ray worker: import_external_libs("verl_omni_ext")
        ↓
verl_omni_ext/__init__.py: _load_all()
        ├── 遍历 entry_points("verl_omni.models")     ← pyproject.toml 声明
        ├── import models/<model>/__init__.py
        │     ├── 触发模块级 L2 Patch（⚠ 必须在 from_pretrained 之前）
        │     └── 触发 @OmniModelBase.register() → 注册到 _registry
        └── @OmniRolloutPipelineBase.register() → 注册到 rollout registry
        ↓
verl-omni: OmniModelBase.get_class_by_name(architecture, "thinker")
        ↓
调用 configure_tokenizer / configure_processor / configure_model
```

三层分治（按对象所有权分层）：

| 层 | 占比 | 放什么 | 在哪 |
|----|------|--------|------|
| L1 插件 | ≥95% | 适配器、数据集、config | `models/<m>/*_adapter.py`、`dataset.py` |
| L2 monkey patch | ~4% | 打 transformers/remote code | `models/<m>/patches.py` |
| L3 gate patch | ≤1% | 打 verl/vllm-omni 源码（要有台账） | `gates/` + `verl_omni_ext/gates/ledger.md` |

---

## 第 1 步：写训练侧适配器

在 `verl_omni_ext/models/<your_model>/` 下新建 `thinker_adapter.py`：

```python
# verl_omni_ext/models/<your_model>/thinker_adapter.py
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase


@OmniModelBase.register("<ArchitectureName>", stage="thinker")
class XxxThinkerAdapter(OmniModelBase):
    """注册键 <ArchitectureName> 必须与 config.json 的 architectures[0] 一致"""

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """返回不参与训练剥离的模块名。不剥离就返回 []"""
        return []

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """加载多模态处理器。

        ⚠ 如果 processor 类名不在 verl.utils.hf_processor 白名单里，
        hf_processor 会 raise 被吞成 None → 静默丢多模态信息。
        此时必须自建 processor（不能走白名单）。
        先跑探针：python -m verl_omni_ext.probes.processor_whitelist --model_path ...
        """
        from verl.utils.hf_processor import hf_processor
        return hf_processor(model_config.local_path,
                            trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """加载 tokenizer"""
        from verl.utils.hf_tokenizer import hf_tokenizer
        return hf_tokenizer(model_config.local_path,
                            trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_model(cls, module, model_config):
        """⚠ 打的是 from_pretrained 返回的已加载实例。

        时序陷阱：from_pretrained 在 configure_model 之前执行
        （fsdp/omni_impl.py:185）。所以"必须在模型加载前生效"的补丁
        不能放这里，必须放包 import 期的 __init__.py。
        这里只放实例级补丁：forward 签名适配、device fix、子模块修剪。
        """
        module = super().configure_model(module, model_config)
        # 例: apply_xxx_device_fix(module)
        return module
```

**关键：不是 3 个方法，是 4 个**。`configure_model` 不是可选——它是实例级补丁的主入口。

---

## 第 2 步：pyproject.toml 声明 entry_point（自动发现）

```toml
# verl-omni-plugin/pyproject.toml
[project.entry-points."verl_omni.models"]
qwen3_5_moe  = "verl_omni_ext.models.qwen3_5_moe"      # 已有
minicpmo_5_0 = "verl_omni_ext.models.minicpmo_5_0"      # 已有
my_new_model = "verl_omni_ext.models.my_new_model"      # ← 加这一行
```

**为什么是 entry_points 而不是手动 import？**
`_load_all()` 遍历这组入口点逐个 import，触发 `@register` 装饰器。
你**不用**改任何上游 `__init__.py`——这就是消灭 42 行 B 类改动（上游合并冲突唯一来源）的机制。

同时创建 `models/<your_model>/__init__.py`：

```python
# verl_omni_ext/models/<your_model>/__init__.py
# 模块级触发该模型特有的 L2 Patch（⚠ 必须在任何 remote code import 之前执行）
from . import thinker_adapter, patches  # noqa: F401
```

如果模型需要数据集防御（槽位④），加 `dataset.py` 并把类路径填到 config：

```yaml
data:
  custom_cls:
    path: pkg://verl_omni_ext.models.my_new_model.dataset
    name: MyDataset
```

---

## 第 3 步：config.yaml 三字段对齐

```yaml
actor_rollout_ref:
  model:
    path: /path/to/my-model
    external_lib: verl_omni_ext            # ← ext 包名（逗号分隔可多个）
    model_type: omni_model
    model_stage: thinker
    trust_remote_code: true
    architecture: <ArchitectureName>        # ← 与 @register 键一致
  rollout:
    engine_backend: vllm_omni
    engine_kwargs:
      vllm_omni:
        pipeline_name: my_new_model         # ← 与槽位②注册键一致
        pipeline_mode: thinker_only
```

三个键必须互相咬合：

| config 字段 | 对应注册点 | 不一致后果 |
|------------|-----------|-----------|
| `architecture` | `@OmniModelBase.register("<Arch>", ...)` | KeyError / 找不到适配器 |
| `external_lib` | pyproject 包名 + `VERL_USE_EXTERNAL_MODULES` | 适配器根本没被 import |
| `pipeline_name` | `@OmniRolloutPipelineBase.register("<name>")` | rollout 侧找不到拓扑 |

---

## 第 4 步（rollout 侧）：pipeline 定义 + gate patch GP-004

训练侧零侵入了；rollout 侧 vllm-omni 的 `_OMNI_MODELS` 是硬编码字典，**没有** plugin 机制。
解决：先给 vllm-omni 打一次性 gate patch（5 行），然后 pipeline 定义放 ext 包里：

```bash
# 一次性（每台机器一次）：
bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni

# 启动脚本加：
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.my_new_model.vllm_omni
```

```python
# verl_omni_ext/models/<your_model>/vllm_omni/__init__.py
from vllm_omni.model_executor.models.registry import _OMNI_MODELS
_OMNI_MODELS["<ArchitectureName>"] = ("my_model", "my_model", "<ClassName>")
from . import pipeline  # noqa: F401
```

gate 行为：环境变量未设 → 不执行任何额外代码 → vllm-omni 行为与上游逐字相同。

---

## 探针前置（写适配器之前先测量）

适配决策是测量出来的，不是猜的：

```bash
python -m verl_omni_ext.probes.forward_signature --model_path /path/to/model
python -m verl_omni_ext.probes.processor_whitelist --model_path /path/to/model
```

| 探针 | 测什么 | 决定什么 |
|------|--------|---------|
| forward_signature | forward 签名（labels/position_ids 是位置参数还是 kwargs） | configure_model 的 forward 适配写法 |
| processor_whitelist | processor 类名是否在 verl 白名单 | configure_processor 用 hf_processor 还是自建 |

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `KeyError: architecture not found` | 注册键与 config.json 的 architectures[0] 不一致 | 检查 `@register` 参数与 config `architecture` |
| 适配器方法从未被调用 | `_load_all()` 没 import 到你的模块 | 检查 pyproject entry_point + `__init__.py` import |
| processor 静默变 None | 类名不在白名单，raise 被吞 | 探针确认后自建 processor |
| 全部样本被过滤到 0 行 | `maybe_filter_out_long_prompts` 吞异常 | 数据集子类显式报错（槽位④） |
| forward 到第一个 micro-batch 才炸 | 签名不兼容，启动时不报错 | 先跑 forward_signature 探针 |
| 补丁打晚了 | from_pretrained 在 configure_model 之前 | 模块级补丁放 `__init__.py`，不是 configure_model |

---

## 相关文档

- [架构设计](plugin_architecture_design.md) — 9 扩展点 + 5 阶段时序 + 零侵入边界
- [三层分治策略](three_layer_strategy.md) — 什么放哪一层，为什么
- [数据处理 Add-on](data_pipeline.md) — 槽位④ + 静默失败陷阱
- [Rollout 侧适配](rollout_adaptation.md) — vllm-omni / vllm 适配（GP-004）