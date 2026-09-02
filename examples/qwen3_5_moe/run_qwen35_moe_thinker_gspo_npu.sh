#!/bin/bash
# ============================================================================
# Qwen3.5-MoE Thinker GSPO 训练启动脚本（模板）
#
# 槽位⑤: 配置载体 + 前置自检
# ============================================================================

set -euo pipefail

# ---- 槽位③: 注册触发器 ----
# verl_omni_ext 和 verl_omni 本体并存，逗号分隔
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext

# ---- GP-004: vllm-omni 外部模块加载 ----
# 前提: 已执行 bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni
# 让 vllm-omni 从 ext 包加载 pipeline 定义，不改 vllm-omni 源码树
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni

# ---- L2 patch 严格模式：任何补丁未打上直接 raise，不静默继续 ----
export VERL_OMNI_EXT_PATCH_STRICT=1

# ---- 模型路径 ----
MODEL_PATH="/path/to/Qwen3.5-MoE"
DATA_PATH="/path/to/training_data.parquet"

# ============================================================================
# 前置自检（把"跑到第 40 分钟才炸"变成"启动 3 秒内退出"）
# ============================================================================
echo "=== 前置自检 ==="

# 1. 适配器注册校验
python -c "
from verl_omni.pipelines.model_base import OmniModelBase
import verl_omni_ext  # 触发入口点自动发现
key = ('Qwen3_5MoeForConditionalGeneration', 'thinker')
assert key in OmniModelBase._registry, f'Adapter {key} not registered!'
print('✓ Adapter registered')
"

# 2. Pipeline 注册校验
python -c "
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
import verl_omni_ext
assert 'qwen3_5_moe' in OmniRolloutPipelineBase._registry, 'Rollout adapter not registered!'
print('✓ Rollout adapter registered')
"

# 3. L2 补丁自检（含签名指纹上报 + watchdog）
#    VERL_OMNI_EXT_PATCH_STRICT=1 时任何补丁未打上会直接 raise
python -c "
from verl_omni_ext._patchkit import self_check, patch_state_line, verify_patches_alive
results = self_check()
for name, ok in results.items():
    assert ok, f'Patch {name} not applied!'
# 输出单行可解析的 patch 状态（Ray worker 日志 grep '^PATCH_STATE ' 收集，
# driver 侧用 assert_patch_consensus() 校验所有 worker 一致）
print(patch_state_line())
# 运行时 watchdog：复查 flag 还在（被 reload/覆盖会响）
assert all(verify_patches_alive().values()), 'Some patch lost at runtime!'
print('✓ All L2 patches applied + alive')
"

# 4. 数据集 schema 校验
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_PATH')
assert len(df) > 0, 'Dataset is empty!'
print(f'✓ Dataset: {len(df)} rows')
"

echo "=== 自检通过 ==="

# ============================================================================
# 训练命令
# ============================================================================
python -m verl_omni.trainer.main_omni \
    --config-path ./config \
    --config-name qwen35_moe_thinker_gspo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.external_lib=verl_omni_ext \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.model_stage=thinker \
    data.train_files=$DATA_PATH \
    "$@"
