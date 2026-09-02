"""
槽位②: Qwen3.5-MoE 推理侧拓扑适配器

@OmniRolloutPipelineBase.register("qwen3_5_moe")

适配器本身几乎不含逻辑——只是把 vllm-omni 那边 frozen 的 pipeline 定义转发过来。
真正的拓扑知识在 vllm-omni 仓里。不要在 verl-omni 里重新描述一遍推理拓扑。
"""
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

try:
    from vllm_omni.model_executor.models.qwen3_5_moe.pipeline import (
        QWEN3_5_MOE_THINKER_ONLY_STAGES,
    )
    _VLLM_OMNI_AVAILABLE = True
except ImportError:
    _VLLM_OMNI_AVAILABLE = False
    QWEN3_5_MOE_THINKER_ONLY_STAGES = None


@OmniRolloutPipelineBase.register("qwen3_5_moe")
class Qwen35MoeRolloutAdapter(OmniRolloutPipelineBase):
    """Qwen3.5-MoE 推理拓扑"""

    @classmethod
    def build_stage_configs(cls, pipeline_mode: str = "thinker_only") -> list:
        if not _VLLM_OMNI_AVAILABLE:
            raise ImportError("vllm_omni pipeline for qwen3_5_moe not available")
        return QWEN3_5_MOE_THINKER_ONLY_STAGES

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """注册键 ≠ pipeline id：qwen3_5_moe vs qwen3_5_moe_thinker_only

        基类默认返回注册键，会生成错误的 YAML。必须覆写。
        """
        return "qwen3_5_moe_thinker_only"

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """确保 vLLM-Omni 的 pipeline registry 里有这个 model_type"""
        if not _VLLM_OMNI_AVAILABLE:
            return
        # vllm_omni 在 import 时自动注册，这里只是确保 import 发生了
