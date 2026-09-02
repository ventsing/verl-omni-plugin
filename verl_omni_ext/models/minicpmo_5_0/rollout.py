"""
槽位②: MiniCPM-o 5.0 推理侧拓扑适配器

try/except 降级 import——如果运行环境的 vllm-omni 还没打补丁，
硬失败会把已跑通的 Qwen3.5 / Qwen3-Omni 一起带下水。
"""
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

try:
    from vllm_omni.model_executor.models.minicpmo_5_0.pipeline import (
        MINICPMO_5_0_THINKER_ONLY_STAGES,
    )
    _VLLM_OMNI_AVAILABLE = True
except ImportError:
    _VLLM_OMNI_AVAILABLE = False
    MINICPMO_5_0_THINKER_ONLY_STAGES = None


@OmniRolloutPipelineBase.register("minicpmo_5_0")
class MiniCPMORolloutAdapter(OmniRolloutPipelineBase):

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        if not _VLLM_OMNI_AVAILABLE:
            raise ImportError("vllm_omni pipeline for minicpmo_5_0 not available")
        return MINICPMO_5_0_THINKER_ONLY_STAGES

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        return "minicpmo_5_0_thinker_only"
