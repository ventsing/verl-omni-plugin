"""
vllm-omni 侧 pipeline 定义（通过 VLLM_OMNI_EXTERNAL_MODULES 加载）

加载链路（需要先给 vllm-omni 打 gate patch GP-004）：
  1. 启动脚本设 export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni
  2. vllm-omni 的 registry.py（打了 GP-004 补丁后）会 importlib.import_module 本模块
  3. 本模块的 __init__.py 往 _OMNI_MODELS 字典注册 architecture → module 映射
  4. 同时 import pipeline.py，PipelineConfig 被 vllm-omni 的 stage_config 发现

前提：vllm-omni 已打 gate patch
  bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni
"""
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

# 注册 architecture → (mod_folder, mod_relname, class_name)
# mod_folder 是 vllm_omni/model_executor/models/ 下的目录名
# 但我们用的是 ext 包里的，所以这里注册的是"转发"路径
_OMNI_MODELS["Qwen3_5MoeForConditionalGeneration"] = (
    "qwen3_5_moe",          # mod_folder（vllm-omni 侧的目录名）
    "qwen3_5_moe",          # mod_relname
    "Qwen3_5MoeForConditionalGeneration",  # class_name
)

# import pipeline 定义——PipelineConfig 在 import 时注册到 vllm-omni 的 stage_config
from . import pipeline  # noqa: F401
