#!/bin/bash
# ============================================================================
# MiniCPM-o 5.0 Thinker GSPO 训练启动脚本
#
# 本脚本是 run_qwen35_moe_thinker_gspo_npu.sh 的移植版，只改了 6 处，每处标了 [MiniCPM]
# ============================================================================

set -euo pipefail

export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext

# GP-004: vllm-omni 外部模块加载（前提: 已打 gates/apply_patches.sh）
export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.minicpmo_5_0.vllm_omni

# L2 patch 严格模式：任何补丁未打上直接 raise，不静默继续
export VERL_OMNI_EXT_PATCH_STRICT=1

# [MiniCPM] 改动 1: 模型路径
MODEL_PATH="/path/to/MiniCPM-o-2.6"
DATA_PATH="/path/to/training_data.parquet"

# [MiniCPM] 改动 2: 环境开关
export MINICPMO_FREEZE_MTP=0           # 冻结未使用的 MTP 头
export MINICPMO_DISABLE_PROCESSOR_FALLBACK=0  # 排障用二分开关

# ============================================================================
# 前置自检
# ============================================================================
echo "=== 前置自检 ==="

python -c "
from verl_omni.pipelines.model_base import OmniModelBase
import verl_omni_ext
# [MiniCPM] 改动 3: architecture 不同
key = ('MiniCPMO', 'thinker')
assert key in OmniModelBase._registry, f'Adapter {key} not registered!'
print('✓ Adapter registered')
"

python -c "
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
import verl_omni_ext
# [MiniCPM] 改动 4: pipeline_name 不同
assert 'minicpmo_5_0' in OmniRolloutPipelineBase._registry, 'Rollout adapter not registered!'
print('✓ Rollout adapter registered')
"

python -c "
from verl_omni_ext._patchkit import self_check, patch_state_line, verify_patches_alive
results = self_check()
for name, ok in results.items():
    assert ok, f'Patch {name} not applied!'
print(patch_state_line())
assert all(verify_patches_alive().values()), 'Some patch lost at runtime!'
print('✓ All L2 patches applied + alive')
"

# [MiniCPM] 改动 5: 数据集 schema 校验（custom_cls）
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
# [MiniCPM] 改动 6: custom_cls + max_audio_tokens
python -m verl_omni.trainer.main_omni \
    --config-path ./config \
    --config-name minicpmo_thinker_gspo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.external_lib=verl_omni_ext \
    actor_rollout_ref.model.model_type=omni_model \
    actor_rollout_ref.model.model_stage=thinker \
    data.train_files=$DATA_PATH \
    data.custom_cls.path=pkg://verl_omni_ext.models.minicpmo_5_0.dataset \
    data.custom_cls.name=MiniCPMOThinkerRLHFDataset \
    actor_rollout_ref.model.max_audio_tokens=4096 \
    "$@"
