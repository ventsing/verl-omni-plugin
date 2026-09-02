# Rollout 侧适配分析

> verl-omni 的零侵入（三层分治）**只覆盖训练侧**。rollout 侧有真实的侵入面。

---

## 全景

```
verl-omni rollout adapter（槽位②）
    │  只是"转发"vllm-omni 的 pipeline 定义，不重新描述拓扑
    ▼
vllm-omni（推理引擎）           ← ⚠ 必须改源码树，没有 plugin 机制
    │  新增 pipeline.py + 模型实现 + deploy yaml + 注册表
    ▼
vllm（底层引擎）                ← 部分零侵入
       平台：✅ vllm.platform_plugins entry_points
       模型：✅ trust_remote_code 从 checkpoint 动态加载
       registry：❌ 硬编码字典
       weight_loader：⚠ MoE 场景需要补丁
```

---

## 1. vllm-omni：必须改源码树

### 为什么无法零侵入

```python
# vllm-omni/vllm_omni/model_executor/models/registry.py:8
_OMNI_MODELS = {                          # ← 硬编码字典
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni", "qwen3_omni", "Qwen3OmniMoeForConditionalGeneration"
    ),
    ...
}
_VLLM_OMNI_MODELS = {**_VLLM_MODELS, **_OMNI_MODELS}  # 合并后使用
```

vllm-omni **没有** `entry_points` 机制，也没有 `external_lib` / `importlib.import_module` 用于外部库加载。

### 每加一个模型需要改的文件

| 改动 | 文件位置 | 侵入性 |
|------|---------|--------|
| pipeline 拓扑定义 | `vllm_omni/model_executor/models/<your_model>/pipeline.py` | 新增文件 |
| 模型实现 | `vllm_omni/model_executor/models/<your_model>/modeling_*.py` | 新增文件 |
| architecture → module 映射 | `_OMNI_MODELS` 字典（`registry.py:8`） | **改上游文件** |
| deploy yaml | `vllm_omni/deploy/<your_model>.yaml` | 新增文件 |
| stage input processors | `vllm_omni/model_executor/stage_input_processors/<your_model>.py` | 新增文件 |

对应报告里的 ③仓（`package_install/MiniCPM`）：
- Qwen3.5: 4 文件 72 行
- MiniCPM-o: 5 文件 243 行

### pipeline.py 示例（frozen 拓扑定义）

```python
# vllm_omni/model_executor/models/qwen3_omni/pipeline.py
QWEN3_OMNI_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe",
    default_deploy_config_name="qwen3_omni_moe.yaml",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            ...
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            ...
        ),
    ),
)
```

### 建议：给 vllm-omni 提上游 PR 要 plugin 扩展点

**已实现 gate patch（GP-004）**：在 ext 包里有一个 5 行的 git patch，给 vllm-omni 加 `VLLM_OMNI_EXTERNAL_MODULES` 扩展点。

```python
# verl_omni_ext/gates/vllm_omni_external_modules.patch（5 行核心逻辑）
for _mod in (m.strip() for m in _os.environ.get("VLLM_OMNI_EXTERNAL_MODULES", "").split(",") if m.strip()):
    _importlib.import_module(_mod)
```

打补丁后，pipeline 定义放在 ext 包里（`models/<model>/vllm_omni/pipeline.py`），不需要改 vllm-omni 源码树：

```bash
# 1. 打补丁（一次性）
bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni

# 2. 启动时设环境变量
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni
```

gate 行为：环境变量未设 → 不执行额外代码 → 与上游逐字相同。这 5 行补丁本身就是一个很好的上游 PR。

---

## 2. vllm：部分零侵入

### 有 plugin 机制

```python
# vllm/vllm/plugins/__init__.py
PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"     # ← entry_points 机制
DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"

# 通过 VLLM_PLUGINS 环境变量控制加载
allowed_plugins = envs.VLLM_PLUGINS
```

| 需求 | 机制 | 零侵入? |
|------|------|---------|
| NPU 平台适配 | `vllm.platform_plugins` entry_points | ✅ vllm-ascend 已用 |
| 模型加载（checkpoint 自带 remote code） | `trust_remote_code` + `try_get_class_from_dynamic_module` | ✅ |
| 模型 registry 注册 | `_VLLM_MODELS` 硬编码字典 | ❌ 要么改字典，要么靠 remote code |
| MoE weight_loader | `process_weights_after_loading` 补丁 | ⚠ L2 monkey patch |

### 关键：trust_remote_code 路径

如果 checkpoint 自带 `modeling_*.py`（remote code），vllm 可以通过 `trust_remote_code=True` 动态加载，不需要改 `_VLLM_MODELS` 字典：

```python
# vllm/vllm/model_executor/models/registry.py
model_module = try_get_class_from_dynamic_module(
    ...,
    trust_remote_code=model_config.trust_remote_code,
)
```

**这就是为什么报告里说 checkpoint remote code "零改动"**——所有不兼容都在 verl-omni 侧用 monkey patch 兜住，而不是改 checkpoint。

---

## 3. weight_loader 的特殊情况（GP-002）

报告提到 `_attach_moe_weight_loaders`：

> process_weights_after_loading 重建 fused expert 参数时丢掉 weight_loader；
> verl 自带的 patch 假设扁平布局，对 Qwen3-Omni 四层嵌套静默跳过

| 属性 | 值 |
|------|-----|
| 打的对象 | vllm 模型对象（`process_weights_after_loading`） |
| 补丁逻辑 | L2（打第三方代码 vllm，合规） |
| 补丁代码位置 | verl-omni 的 `vllm_rollout/utils.py`（**改了 verl-omni 源码**） |
| 归类 | **L3 gate patch → GP-002** |

正确做法：补丁代码应该写在 `verl_omni_ext` 里（L2），不应该写在 verl-omni 源码里（L3）。

---

## 4. verl-omni 的 rollout adapter（槽位②）只是转发

```python
# verl_omni_ext/models/qwen3_5_moe/rollout.py
try:
    from vllm_omni.model_executor.models.qwen3_5_moe.pipeline import (
        QWEN3_5_MOE_THINKER_ONLY_STAGES,
    )
except ImportError:
    ...  # 降级，不硬失败
```

verl-omni 的 rollout adapter 几乎不含逻辑——只是把 vllm-omni 那边 frozen 的 pipeline 定义**转发**过来。真正的拓扑知识在 vllm-omni 仓里。这个分工是对的。

---

## 5. "零侵入"的真实边界

| 层 | 零侵入? | 说明 |
|----|---------|------|
| verl-omni 训练侧 | ✅ | 三层分治完全覆盖 |
| verl-omni rollout adapter | ✅ | 槽位②只是转发 |
| **vllm-omni pipeline/模型** | **✅ gate patch** | **GP-004: 5 行补丁加 VLLM_OMNI_EXTERNAL_MODULES** |
| vllm 平台适配 | ✅ | `vllm.platform_plugins` entry_points |
| vllm 模型加载 | ✅ | `trust_remote_code` 动态加载 |
| vllm weight_loader | ⚠ L2 | monkey patch 打 vllm 对象 |
