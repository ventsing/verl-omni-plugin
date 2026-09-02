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

# 3. L2 补丁自检
python -c "
from verl_omni_ext._patchkit import self_check
results = self_check()
for name, ok in results.items():
    assert ok, f'Patch {name} not applied!'
print('✓ All L2 patches applied')
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
