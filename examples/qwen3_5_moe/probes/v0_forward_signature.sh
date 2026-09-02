#!/bin/bash
# ============================================================================
# 探针 v0: 测量 forward 签名
#
# 适配决策是被测量出来的，不是被猜出来的。
# 先写探针把关键事实测出来，再写适配器——这个顺序能省掉大部分返工。
#
# 用法: bash probes/v0_forward_signature.sh /path/to/checkpoint
# ============================================================================

set -euo pipefail
MODEL_PATH="${1:?Usage: $0 <model_path>}"

python -c "
import inspect
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True)
print(f'architecture: {config.architectures}')
print(f'model_type: {config.model_type}')

# 加载模型，检查 forward 签名
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH', trust_remote_code=True, device_map='cpu'
)

sig = inspect.signature(model.forward)
print(f'forward signature: {sig}')
print(f'forward params: {list(sig.parameters.keys())}')

# 判断是单位置字典约定还是全关键字调用
params = list(sig.parameters.keys())
if len(params) == 1 and params[0] != 'self':
    print('⚠ forward 是单位置参数约定 → 需要 forward 适配器（L2 monkey patch）')
elif 'input_ids' in params:
    print('✓ forward 是全关键字调用 → 与 verl 兼容')
else:
    print(f'⚠ forward 签名不明确，需要手动验证: {params}')
"
