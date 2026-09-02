"""
MiniCPM-o vllm-omni pipeline 拓扑定义

加载方式同 qwen3_5_moe：
  export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.minicpmo_5_0.vllm_omni
"""
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

_OMNI_MODELS["MiniCPMO"] = (
    "minicpmo_5_0",
    "minicpmo_5_0",
    "MiniCPMOForConditionalGeneration",
)

from . import pipeline  # noqa: F401
