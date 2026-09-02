# verl-omni 插件架构设计

> 基于 verl-omni 真实源码 + 实际开发总结。代码位置带行号可核对。

---

## 一、为什么适配代码 98% 已经是插件

实际开发中两次模型适配的改动分布：

| 适配 | 新文件 | 核心文件改动 | 碰到的上游文件 |
|------|--------|-------------|---------------|
| Qwen3.5-MoE | 908 行 | 11 行 | 2 个 `__init__.py` |
| MiniCPM-o 5.0 | 4241 行 | 31 行 | 同 2 个 `__init__.py` |

**42 行核心改动全部是 `from .xxx import yyy` + `__all__ += [...]`**，用入口点自动发现可归零（见 §四）。

之所以能做到这个比例，因为 verl-omni 上游预留了 **5 个扩展点**。适配新模型 = 往 5 个槽位里填东西。

---

## 二、5 个扩展点

### 槽位 ① OmniModelBase 注册表 —— 训练侧适配器

```python
# verl_omni/pipelines/model_base.py:449
@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen35MoeThinkerAdapter(OmniModelBase): ...
```

注册键 = `(architecture, stage)`，`architecture` 取自 checkpoint `config.json` 的 `architectures[0]`。

**两个调用点（时序理解关键）：**

```
A) 配置构造期 — model.py:184
   OmniModelConfig.__post_init__()
     ├─ import_external_libs(self.external_lib)     ← 触发注册
     ├─ adapter_cls = OmniModelBase.get_class_by_name(...)
     ├─ self.tokenizer = adapter_cls.configure_tokenizer(...)   ← 1a
     └─ self.processor = adapter_cls.configure_processor(...)   ← 1b

B) 模型构建期 — fsdp/omni_impl.py:185
   OmniFSDPEngine._build_module()
     ├─ module = AutoModelForMultimodalLM.from_pretrained(...)  ← 注意：在 configure_model 之前！
     ├─ adapter_cls = OmniModelBase.get_class_by_name(...)
     └─ module = adapter_cls.configure_model(module, ...)        ← 1c
            └─ 基类默认实现：按 get_strip_modules() 删子模块   ← 1d
```

⚠ **时序陷阱**：`from_pretrained` 在 `configure_model` **之前**执行。所以"必须在模型加载前生效"的补丁（如让 transformers 能 import 得动 remote code），放在 `configure_model` 里**来不及**，必须放在**包 import 期**：

```python
# verl_omni_ext/models/minicpmo_5_0/__init__.py 模块级
apply_minicpmo_auto_register_guard()     # 必须在任何 remote code 被 import 之前
apply_minicpmo_automodel_fallback()
```

而 forward 适配器打的是 `from_pretrained` 返回的实例，放在 `configure_model` 里恰好正确。"补丁放哪一层"完全由它作用的对象的生命周期决定。

**四个钩子的实际用法对照：**

| 钩子 | Qwen3.5 | MiniCPM-o |
|------|---------|-----------|
| get_strip_modules | `[]`（全量训练 ViT） | `[]`（不剥 tts.*，保持 state_dict 逐字相同） |
| configure_model | ViT 位置编码设备补丁 | forward 适配器 + MTP 汇报 |
| configure_tokenizer | 标准 hf_tokenizer | 标准（trust_remote_code=True） |
| configure_processor | 标准 hf_processor | **必须自建**（hf_processor 的 match 默认 raise） |

### 槽位 ② OmniRolloutPipelineBase 注册表 —— 推理侧拓扑

```python
# model_base.py:625
@OmniRolloutPipelineBase.register("qwen3_5_moe")
@OmniRolloutPipelineBase.register("minicpmo_5_0")
```

注册键 = `model_type` 字符串，由 `+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name` 传入。

适配器本身几乎不含逻辑——只是把 vllm-omni 那边 frozen 的 pipeline 定义转发过来。真正的拓扑知识在 vllm-omni 仓里。

**两条经验：**
- 注册键 ≠ pipeline id（`qwen3_5_moe` vs `qwen3_5_moe_thinker_only`），`get_pipeline_id` 必须覆写
- rollout adapter 的 import 必须 try/except 降级——否则运行环境 vllm-omni 没打补丁时，硬失败会把已跑通的模型一起带下水

### 槽位 ③ VERL_USE_EXTERNAL_MODULES —— 注册触发器

```bash
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
```

每个 Ray worker 进程都会 `import_external_libs()` → 执行 `__init__.py` → 触发 `@register`。

**关键**：接受逗号分隔的多个模块名。`verl_omni_ext` 可以和 `verl_omni` 并存。

### 槽位 ④ data.custom_cls —— 数据集注入

```yaml
data:
  custom_cls:
    path: pkg://verl_omni_ext.models.minicpmo_5_0.dataset
    name: MiniCPMOThinkerRLHFDataset
```

MiniCPM-o 用它把一个**静默失败**变成显式报错：上游的 `maybe_filter_out_long_prompts` 用 `except Exception: return max_prompt_length+1` 兜底，处理器一抛异常就把 7473 行样本全过滤成 0 行，只在日志刷 warning。子类覆写后强制走 tokenizer 分支，过滤比例超阈值时直接报错。

**凡是上游有 `except Exception` 吞掉的路径、而你的模型正好会走进去，就用 custom_cls 子类化它。**

### 槽位 ⑤ examples 脚本 —— 配置载体

不算代码扩展点，但是最重要的配置载体。MiniCPM-o 脚本明确标注"本脚本是 run_qwen35_moe_thinker_gspo_npu.sh 的移植版，只改了 6 处，每处标了 [MiniCPM]"。

**标注 diff 点 + 内置前置自检** = 把"跑到第 40 分钟才炸"变成"启动 3 秒内退出并告诉你缺什么"。

---

## 三、改动的三种性质分类

### A 类：填槽位的新文件（纯插件，零冲突）

| 文件 | 槽位 | 换模型时 |
|------|------|---------|
| `models/<m>/adapter.py` | ① | 必写，一模型一份 |
| `models/<m>/rollout.py` | ② | 必写，一模型一份 |
| `models/<m>/dataset.py` | ④ | 按需 |
| `examples/<m>/run_*.sh` | ⑤ | 必写，从最近模型移植 |
| `examples/<m>/probes/v0_*.sh` | — | **强烈建议先写探针再写适配器** |
| `tests/<m>/*.py` | — | 适配器边界 CPU 测试 |

### B 类：核心文件编辑（唯一冲突面，**必须消灭**）

```
verl_omni/models/transformers/__init__.py   Qwen3.5: +8    MiniCPM-o: +28/-1
verl_omni/pipelines/__init__.py             Qwen3.5: +3    MiniCPM-o: +3
```

全部是 `from .xxx import yyy` + `__all__ += [...]`。→ §四的入口点自动发现归零它。

### C 类：Monkey patch（打在第三方代码上）

| 函数 | 打谁 | 为什么只能 monkey patch |
|------|------|----------------------|
| `apply_qwen3_5_vision_device_fix` | Qwen3_5MoeVisionModel | FSDP2 CPUOffload 下参数报 cpu、激活在 npu |
| `apply_minicpmo_auto_register_guard` | transformers AutoClass | checkpoint 的 processing 用 str 当 config class 传 |
| `apply_minicpmo_automodel_fallback` | AutoModelForMultimodalLM | checkpoint 的 auto_map 缺键 |
| `build_minicpmo_forward_adapter` | 模型实例 forward | remote code 是单位置字典约定，verl 是全关键字调用 |
| `build_minicpmo_processor` | 绕过 verl.utils.hf_processor | 上游 match 默认 raise，外层 except 吞成 None |

**为什么这是正确的选择**：这些代码的所有权不在你手上（pip 装的 / checkpoint 带的 / 运行时的），没有 PR 权的地方，也没有加 hook 的地方。

---

## 四、入口点自动发现：消灭 42 行核心编辑

**现状**：加一个模型要编辑 `pipelines/__init__.py` 和 `models/transformers/__init__.py`。
**目标**：加一个模型只新增一个目录 + `pyproject.toml` 加一行，零编辑。

```toml
# verl_omni_ext/pyproject.toml
[project.entry-points."verl_omni.models"]
qwen3_5_moe  = "verl_omni_ext.models.qwen3_5_moe"
minicpmo_5_0 = "verl_omni_ext.models.minicpmo_5_0"
```

```python
# verl_omni_ext/__init__.py
def _load_all():
    for ep in entry_points(group="verl_omni.models"):
        try:
            ep.load()          # import 触发 @register + 模块级 patch
        except Exception as e:
            logger.warning("model plugin %r unavailable: %s", ep.name, e)
```

收益：
- 对 verl-omni 上游文件的编辑**归零**（B 类彻底消失）
- 团队两人加两模型，改 pyproject.toml 不同行 + 各自新目录，冲突面从"import 块"缩到"一行声明"
- 禁用某模型：删一行声明，或依赖没装时自动降级

---

## 五、换模型时的改动矩阵

| 变更 | 槽位① | 槽位② | 槽位④ | L2 补丁 | 启动脚本 |
|------|-------|-------|-------|---------|---------|
| 换 LLM 主干 | forward 签名、strip、tie | vllm-omni 新 pipeline | 一般不动 | MoE → weight_loader | 显存/精度全重测 |
| 换音频编码头 | 可能要 device fix | stage 拓扑可能变多段 | audio_key 分支 | processor audio 分支 | max_audio_tokens |
| 换视频编码头 | m-RoPE / position_ids | 同上 | 视频列形状 | process_vision_info | max_video_tokens |

**跨模型成立的经验：**
1. LLM 主干相同 ⇒ 显存/精度/micro-batch 结论可搬。MiniCPM-o 的 thinker 主干就是 Qwen3.5-MoE，脚本只改 6 处
2. 视觉/音频栈不同 ⇒ 一行都不能抄
3. `configure_processor` 最容易翻车——verl.utils.hf_processor 白名单 match，类名不在六个已知里就 raise → 吞成 None
4. "能加载"和"能前向"是两件事——forward 签名不兼容在启动时不报错，到第一个 micro-batch 才炸
5. 静默失败比崩溃贵得多——主动找上游的 `except Exception` 和 `warnings.warn`

---

## 六、零侵入边界

| 层 | 零侵入? | 说明 |
|----|---------|------|
| verl-omni 训练侧 | ✅ | 三层分治完全覆盖 |
| verl-omni rollout adapter | ✅ | 槽位②只是转发 vllm-omni 定义 |
| **vllm-omni pipeline/模型** | **❌** | **必须改源码树，无 plugin 机制** |
| vllm 平台适配 | ✅ | `vllm.platform_plugins` entry_points |
| vllm 模型加载 | ✅ | `trust_remote_code` 动态加载 |
| vllm weight_loader | ⚠ L2 | monkey patch 打 vllm 对象 |
| 数据处理 | ✅ | 槽位①+④完全覆盖 |

详见：
- [Rollout 侧适配分析](rollout_adaptation.md) — vllm-omni / vllm 需要改什么
- [数据处理 Add-on](data_pipeline.md) — 6 个数据扩展点 + 静默失败陷阱

---

## 七、完整扩展点覆盖

除了 verl-omni 的 5 个槽位，verl/verl-omni 还有更多注册表可被 ext 包使用：

| 扩展点 | 注册表 | ext 包位置 | entry_points? | 配置字段 |
|--------|--------|-----------|---------------|---------|
| 模型 adapter | `@OmniModelBase.register` | `models/` | ✅ `verl_omni.models` | `architecture` |
| 推理 rollout | `@OmniRolloutPipelineBase.register` | `models/` | ✅ `verl_omni.models` | `pipeline_name` |
| 训练范式 | `@register_trainer` | `trainer/` | ✅ `verl_omni.trainers` | `trainer_name` |
| reward 管理器 | `@register` (reward_loop) | `reward/managers.py` | ✅ `verl_omni.reward` | `reward_manager` |
| 优势估计器 | `@register_adv_est` | `algos/adv_est.py` | import 触发 | `adv_estimator` |
| policy loss | `@register_policy_loss` | `algos/policy_loss.py` | import 触发 | `policy_loss` |
| 自定义 reward 函数 | `load_extern_object` | `reward/functions.py` | 路径指向 | `custom_reward_function` |
| 数据集 | `load_extern_object` | `models/<m>/dataset.py` | 路径指向 | `data.custom_cls` |
| 自定义 worker | `worker_cls` config | `workers/` | 路径指向 | `worker_cls` |

**入口点自动发现覆盖 3 组**：models、trainers、reward。
**algos 和 workers 通过 import 触发**（它们的注册表不在 `verl_omni` 命名空间下）。

加新模型/trainer/reward 的操作：
```bash
# 1. pyproject.toml 加一行 entry_points 声明
# 2. 新建一个目录/文件
# 3. 从最近模型移植 config + 启动脚本
# 不碰任何上游 __init__.py
```
