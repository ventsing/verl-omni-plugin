#!/bin/bash
# ============================================================================
# Gate patch 应用脚本 —— GP-004 已退役，本脚本现在是提示性 no-op
#
# GP-004（vllm_omni_external_modules.patch）已被 vllm-omni 的原生插件机制
# 取代：入口点组 vllm_omni.general_plugins。
# 详见 verl_omni_ext/gates/ledger.md 的"GP-004 退役记录"。
#
# 新用法（零 patch）：
#   pip install -e /path/to/verl-omni-ext
#   # pyproject.toml 里的 [project.entry-points."vllm_omni.general_plugins"]
#   # 会让 vllm-omni 在所有进程自动发现并执行 register()。
#
# 如果你之前打过 GP-004 补丁，回滚：
#   cd /path/to/vllm-omni && git checkout -- vllm_omni/model_executor/models/registry.py
# ============================================================================
set -euo pipefail

VLLM_OMNI_DIR="${1:-}"

echo "=== GP-004 已退役（superseded by vllm_omni.general_plugins）==="
echo ""
echo "不再需要给 vllm-omni 打补丁。新机制："
echo "  pip install -e /path/to/verl-omni-ext"
echo "  # 入口点 [project.entry-points.\"vllm_omni.general_plugins\"]"
echo "  # → verl_omni_ext.vllm_omni_plugins:register"
echo "  # → load_omni_general_plugins() 在所有进程自动执行"
echo ""

if [ -n "$VLLM_OMNI_DIR" ]; then
    if grep -q "VLLM_OMNI_EXTERNAL_MODULES" "$VLLM_OMNI_DIR/vllm_omni/model_executor/models/registry.py" 2>/dev/null; then
        echo "⚠ 检测到 $VLLM_OMNI_DIR 仍带着 GP-004 补丁，建议回滚："
        echo "  cd $VLLM_OMNI_DIR && git checkout -- vllm_omni/model_executor/models/registry.py"
    else
        echo "✓ $VLLM_OMNI_DIR 干净（无 GP-004 补丁），无需任何操作。"
    fi
fi
