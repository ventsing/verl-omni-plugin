# vllm-omni 侧跨仓适配记录

> verl-omni 的零侵入只覆盖训练侧。rollout 侧需要改 vllm-omni 源码树。
> 本文档记录每个模型的 vllm-omni 侧改动清单，确保不遗漏。

详见 [rollout 适配分析](rollout_adaptation.md) 的完整分析。

---

## 为什么 vllm-omni 无法零侵入

vllm-omni 的 `_OMNI_MODELS` 是硬编码字典（`registry.py:8`），没有 entry_points / external_lib 机制。

**但是：GP-004 gate patch 已解决这个问题。**

打补丁后（5 行代码），pipeline 定义放在 ext 包里，不需要改 vllm-omni 源码树：

```bash
# 一次性打补丁
bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni

# 启动时设环境变量
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni
```

### GP-004 补丁内容

在 `_OMNI_MODELS` 字典定义后、`_VLLM_OMNI_MODELS` 合并前，加 5 行：

```python
for _mod in (m.strip() for m in _os.environ.get("VLLM_OMNI_EXTERNAL_MODULES", "").split(",") if m.strip()):
    _importlib.import_module(_mod)  # 外部模块往 _OMNI_MODELS 字典注册
```

- gate off（环境变量未设）→ 不执行额外代码 → 与上游逐字相同
- gate on → import 外部模块 → pipeline 定义从 ext 包加载

### ext 包里的 pipeline 定义

```
verl_omni_ext/models/qwen3_5_moe/vllm_omni/
├── __init__.py          # 往 _OMNI_MODELS 字典注册 architecture → module 映射
└── pipeline.py          # PipelineConfig 拓扑定义（frozen）
```

详见 [`verl_omni_ext/gates/`](../verl_omni_ext/gates/) 目录。

---

## 每个模型的 vllm-omni 侧改动清单

### Qwen3.5-MoE

| # | 改动 | 文件 | 行数 |
|---|------|------|------|
| 1 | pipeline 拓扑定义 | `vllm_omni/model_executor/models/qwen3_5_moe/pipeline.py` | ~40 |
| 2 | 模型实现 | `vllm_omni/model_executor/models/qwen3_5_moe/modeling_*.py` | ~20 |
| 3 | architecture → module 映射 | `_OMNI_MODELS` 字典（`registry.py`） | +2 |
| 4 | deploy yaml | `vllm_omni/deploy/qwen3_5_moe.yaml` | ~10 |

**合计**：4 文件 72 行

### MiniCPM-o 5.0

| # | 改动 | 文件 | 行数 |
|---|------|------|------|
| 1 | pipeline 拓扑定义 | `vllm_omni/model_executor/models/minicpmo_5_0/pipeline.py` | ~80 |
| 2 | 模型实现 | `vllm_omni/model_executor/models/minicpmo_5_0/modeling_*.py` | ~120 |
| 3 | architecture → module 映射 | `_OMNI_MODELS` 字典（`registry.py`） | +2 |
| 4 | deploy yaml | `vllm_omni/deploy/minicpmo_5_0.yaml` | ~20 |
| 5 | stage input processors | `vllm_omni/model_executor/stage_input_processors/minicpmo_5_0.py` | ~21 |

**合计**：5 文件 243 行

### 全双工（如果需要流式推理）

| # | 改动 | 文件 | 说明 |
|---|------|------|------|
| 1 | DuplexAdapter 适配 | `vllm_omni/experimental/fullduplex/personaplex/adapter.py` | 适配到你的模型 |
| 2 | DuplexSession 适配 | `vllm_omni/experimental/fullduplex/personaplex/session.py` | 会话管理 |
| 3 | stage0 适配 | `vllm_omni/experimental/fullduplex/personaplex/stage0.py` | 输入处理 |

**注意**：如果全双工只做"训练和推理并发"（不做流式推理），不需要改 vllm-omni——verl 的 `FullyAsyncLLMServerClient` 已够用。

---

## vllm 侧改动

| 需求 | 机制 | 侵入性 |
|------|------|--------|
| NPU 平台 | `vllm.platform_plugins` entry_points | ✅ 零侵入（vllm-ascend） |
| 模型加载 | `trust_remote_code` | ✅ 零侵入 |
| MoE weight_loader | L2 monkey patch | ⚠ 打 vllm 对象 |
| model registry | `_VLLM_MODELS` 硬编码 | ❌ 要么改字典，要么靠 remote code |

---

## 建议：给 vllm-omni 提上游 PR

```python
# 建议给 vllm-omni 提的扩展点（上游 PR）
# vllm_omni/__init__.py
def _load_external_pipelines():
    """从 VLLM_OMNI_EXTERNAL_MODULES 加载外部 pipeline 定义"""
    for module in os.environ.get("VLLM_OMNI_EXTERNAL_MODULES", "").split(","):
        if module:
            importlib.import_module(module)
```

这样未来新模型的 pipeline 定义可以放在 `verl_omni_ext` 里，不需要改 vllm-omni 源码树。

---

## 换模型时的 vllm-omni 侧检查清单

- [ ] pipeline.py 定义了正确的 stage 拓扑
- [ ] `_OMNI_MODELS` 注册表加了 architecture → module 映射
- [ ] deploy yaml 配置正确
- [ ] stage input processors 正确（如有多 stage）
- [ ] 如果跑 NPU：vllm-ascend 已安装
- [ ] 如果 MoE：weight_loader 补丁已打
