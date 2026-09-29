# vllm-omni 侧跨仓适配记录

> verl-omni 的零侵入覆盖训练侧；rollout 侧经 vllm-omni 原生插件组
> `vllm_omni.general_plugins` 注册（零 patch）。
> 本文档记录每个模型的 vllm-omni 侧改动清单，确保不遗漏。

详见 [rollout 适配分析](rollout_adaptation.md) 的完整分析。

---

## vllm-omni 侧注册：原生插件机制（零 patch）

**更新（源码复查，推翻早前判断）**：vllm-omni 的 `_OMNI_MODELS` 确实是
硬编码字典（`model_executor/models/registry.py:8`），但 vllm-omni 有原生
入口点插件组 **`vllm_omni.general_plugins`**（`vllm_omni/plugins/__init__.py`）。
`load_omni_general_plugins()` 在 `OmniEngineArgs.__post_init__`
（`engine/arg_utils.py:252`）发现入口点并**执行**，所有进程加载，
时机早于注册表消费。

**GP-004 gate patch 已退役**——它不仅多余，其 `_OMNI_MODELS` 元组注册
还有三个 bug（硬编码前缀拼不进树外路径 / 时机早于 `OmniModelRegistry`
实例构造 / 静默覆盖上游同名 arch）。见 `gates/ledger.md` 退役记录。

新机制（pip install 即生效）：

```toml
[project.entry-points."vllm_omni.general_plugins"]
verl_omni_ext = "verl_omni_ext.vllm_omni_plugins:register"
```

### ext 包里的注册链

```
vllm_omni.general_plugins 入口点
  → verl_omni_ext/vllm_omni_plugins.py: register()
  → models/<model>/vllm_omni/__init__.py: register()
      ├── register_pipeline(PipelineConfig)   → OMNI_PIPELINES
      └── (可选) OmniModelRegistry.register_model(arch, "module:Class")
```

```
verl_omni_ext/models/qwen3_5_moe/vllm_omni/
├── __init__.py          # register()：register_pipeline + 模型类注册骨架
└── pipeline.py          # PipelineConfig 拓扑定义（frozen）
```

---

## 每个模型的 vllm-omni 侧改动清单

### Qwen3.5-MoE

| # | 改动 | 文件 | 行数 |
|---|------|------|------|
| 1 | pipeline 拓扑定义 | `verl_omni_ext/models/qwen3_5_moe/vllm_omni/pipeline.py` | ~40 |
| 2 | 模型实现（落地后） | `verl_omni_ext/models/qwen3_5_moe/vllm_omni/modeling_*.py` | ~20 |
| 3 | 注册 | `vllm_omni/__init__.py: register()`（原生插件组编排） | ~10 |
| 4 | deploy yaml | `verl_omni_ext/.../qwen3_5_moe.yaml`（或 vllm-omni deploy 目录软链） | ~10 |

**合计**：ext 包内 4 文件 ~80 行（vllm-omni 源码树零改动）

### MiniCPM-o 5.0

| # | 改动 | 文件 | 行数 |
|---|------|------|------|
| 1 | pipeline 拓扑定义 | `verl_omni_ext/models/minicpmo_5_0/vllm_omni/pipeline.py` | ~80 |
| 2 | 模型实现（复用上游 MiniCPMO 类） | — | 0 |
| 3 | 注册 | `vllm_omni/__init__.py: register()`（只注册拓扑，不碰模型类） | ~10 |
| 4 | deploy yaml | `verl_omni_ext/.../minicpmo_5_0.yaml` | ~20 |
| 5 | stage input processors | `verl_omni_ext/.../stage_input_processors.py`（PipelineConfig 虚线引用） | ~21 |

**合计**：ext 包内 4 文件 ~130 行（vllm-omni 源码树零改动）

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

## 换模型时的 vllm-omni 侧检查清单

- [ ] pipeline.py 定义了正确的 stage 拓扑
- [ ] `vllm_omni/__init__.py: register()` 调用 `register_pipeline`
- [ ] （仅新架构）`_maybe_register_model_class` 的注册代码已解开
- [ ] deploy yaml 配置正确
- [ ] stage input processors 正确（如有多 stage）
- [ ] 如果跑 NPU：vllm-ascend 已安装
- [ ] 如果 MoE：weight_loader 补丁已打
- [ ] 推理环境 `pip install -e` 了 verl-omni-ext（entry point 可见）
