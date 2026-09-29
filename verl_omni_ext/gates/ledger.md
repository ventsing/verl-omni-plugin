# L3 Gate Patch 台账

> 硬性上限：≤ 5 条。超了说明扩展点不够，该给上游提 issue 要扩展点。

## 当前台账

| ID | 仓库 | 文件 | 现象 | gate 变量 | patch 文件 | 上游 PR | 状态 |
|----|------|------|------|-----------|-----------|---------|------|
| GP-001 | verl | `trainer/main_ppo.py:140` | omni trainer 未注册 | 无（无条件） | — | 未提 | ⚠ 最紧急 |
| GP-002 | verl-omni | `workers/rollout/utils.py:263` | MoE weight_loader 丢失 | `VERL_OMNI_MOE_LOADER_FIX` | — | 应提 | 待办 |
| GP-003 | verl-omni | `workers/rollout/vllm_omni_async_server.py` | additional_config 提升 | 无 | — | 应提 | 待办 |
| ~~GP-004~~ | ~~vllm-omni~~ | ~~`model_executor/models/registry.py`~~ | ~~无外部模块加载机制~~（判断有误，见下） | ~~`VLLM_OMNI_EXTERNAL_MODULES`~~ | `vllm_omni_external_modules.patch`（仅存档） | 无需 | 🪦 **已退役** |

## GP-004 退役记录（源码复查后撤销）

**原判断错了**：当时认为 vllm-omni 的 `_OMNI_MODELS` 硬编码字典"无 plugin
机制"。复查 `vllm_omni/plugins/__init__.py` 发现 vllm-omni 有原生入口点
插件组 **`vllm_omni.general_plugins`**：

- `load_omni_general_plugins()`（`engine/arg_utils.py:252`，
  `OmniEngineArgs.__post_init__` 内）发现该组的入口点并**执行**
  （不只 import，是 `func()`）
- 所有进程都会加载：主进程、引擎核心、stage 子进程
  （`worker/mixins.py:13`、`engine/stage_init_utils.py:755`、
  diffusion worker / stage proc）
- 时机在 `ModelConfig.registry` 首次消费与 `resolve_pipeline_config`
  之前——插件注册的一切都能被引擎看到

**新机制**（L1，零 patch，`pip install` 即生效）：

```toml
# pyproject.toml
[project.entry-points."vllm_omni.general_plugins"]
verl_omni_ext = "verl_omni_ext.vllm_omni_plugins:register"
```

`register()`（`verl_omni_ext/vllm_omni_plugins.py`）使用 vllm-omni 公开 API：
- `register_pipeline(PipelineConfig)`（`config/pipeline_registry.py:192`，
  docstring 明说 "Register an out of tree pipeline"）
- `OmniModelRegistry.register_model(arch, "module:Class")`——惰性字符串
  支持外部包路径；需要时同步注册 vllm 上游 `ModelRegistry`
  （vllm-omni 自己在 `engine/arg_utils.py:138` 就这么镜像）

**GP-004 补丁本身还有三个 bug**（即使保留也不可用）：
1. 元组被硬编码前缀 `vllm_omni.model_executor.models.{folder}.{relname}`
   拼接——外部包路径拼不进去，惰性导入必 `ModuleNotFoundError`
2. 注入时机在 registry 模块 import 期（:477），早于 `OmniModelRegistry`
   实例构造（:484）——无法改用实例 API `register_model`
3. `_OMNI_MODELS` 字典赋值静默覆盖上游同名 arch（如 `MiniCPMO`）

**前置条件**：verl_omni_ext 需 pip install 到推理环境
（entry_points 发现只看已安装 distribution）——与训练侧
`verl_omni.models` 入口点组的既有假设一致，无新增约束。

**回滚方法**（如果之前打过补丁）：
```bash
cd /path/to/vllm-omni && git checkout -- vllm_omni/model_executor/models/registry.py
unset VLLM_OMNI_EXTERNAL_MODULES
```

patch 文件保留在 `gates/` 仅作历史存档；`apply_patches.sh` 已改为
提示退役的 no-op。

## 其余条目

GP-001 / GP-002 / GP-003 详解见
[docs/gate_patch_ledger.md](../../docs/gate_patch_ledger.md)（未变）。
