# L3 Gate Patch 台账

> 硬性上限：≤ 5 条。超了说明扩展点不够，该给上游提 issue 要扩展点。

## 当前台账

| ID | 仓库 | 文件 | 现象 | gate 变量 | patch 文件 | 上游 PR | 状态 |
|----|------|------|------|-----------|-----------|---------|------|
| GP-001 | verl | `trainer/main_ppo.py:140` | omni trainer 未注册 | 无（无条件） | — | 未提 | ⚠ 最紧急 |
| GP-002 | verl-omni | `workers/rollout/utils.py:263` | MoE weight_loader 丢失 | `VERL_OMNI_MOE_LOADER_FIX` | — | 应提 | 待办 |
| GP-003 | verl-omni | `workers/rollout/vllm_omni_async_server.py` | additional_config 提升 | 无 | — | 应提 | 待办 |
| GP-004 | **vllm-omni** | `model_executor/models/registry.py:476` | **无外部模块加载机制** | `VLLM_OMNI_EXTERNAL_MODULES` | `vllm_omni_external_modules.patch` | 应提 | ✅ 已实现 |

## GP-004 详解：vllm-omni 外部模块加载

**问题**：vllm-omni 的 `_OMNI_MODELS` 是硬编码字典，加模型必须改源码树。

**补丁**：在 `_OMNI_MODELS` 字典定义后、`_VLLM_OMNI_MODELS` 合并前，加 5 行代码：
```python
for _mod in (m.strip() for m in _os.environ.get("VLLM_OMNI_EXTERNAL_MODULES", "").split(",") if m.strip()):
    _importlib.import_module(_mod)
```

**gate 行为**：
- `VLLM_OMNI_EXTERNAL_MODULES` 未设 → 不执行任何额外代码 → 行为与上游逐字相同
- 设了 → import 外部模块 → 外部模块往 `_OMNI_MODELS` 字典注册

**应用方式**：
```bash
bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni
```

**收益**：加新模型不需要改 vllm-omni 源码树了——pipeline 定义放在 ext 包里。

**上游 PR 计划**：这 5 行补丁本身就是一个很好的上游 PR——给 vllm-omni 加一个公开的扩展点。

详见 [docs/gate_patch_ledger.md](../../docs/gate_patch_ledger.md)
