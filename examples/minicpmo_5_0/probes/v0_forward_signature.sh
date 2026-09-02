#!/bin/bash
# ============================================================================
# 探针 v0: MiniCPM-o forward 签名 + processor 行为
#
# 同 Qwen3.5 的探针，但额外测 MiniCPM-o 特有的:
# - AutoClass register 是否会 AttributeError
# - hf_processor 白名单是否能 match
# ============================================================================

set -euo pipefail
MODEL_PATH="${1:?Usage: $0 <model_path>}"

python -c "
import inspect, traceback
from transformers import AutoConfig

config = AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True)
print(f'architecture: {config.architectures}')
print(f'model_type: {config.model_type}')

# 测 1: AutoClass register 是否会炸
try:
    from transformers import AutoImageProcessor
    # 模拟 checkpoint 的 processing_minicpmo.py:478 行为
    AutoImageProcessor.register('minicpmo', 'MiniCPMOProcessor')
    print('✓ AutoClass register OK')
except AttributeError as e:
    print(f'⚠ AutoClass register raises: {e}')
    print('  → 需要 apply_minicpmo_auto_register_guard (L2)')

# 测 2: hf_processor 白名单
try:
    from verl.utils.hf_processor import hf_processor
    proc = hf_processor('$MODEL_PATH', trust_remote_code=True)
    print(f'✓ hf_processor: {type(proc).__name__}')
except (ValueError, Exception) as e:
    print(f'⚠ hf_processor raises: {e}')
    print('  → 需要 build_minicpmo_processor (绕过白名单)')

# 测 3: forward 签名
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH', trust_remote_code=True, device_map='cpu'
)
sig = inspect.signature(model.forward)
print(f'forward signature: {sig}')
params = list(sig.parameters.keys())
if len(params) == 1 and params[0] != 'self':
    print(f'⚠ forward 是单位置参数约定 ({params}) → 需要 forward 适配器 (L2)')
"
