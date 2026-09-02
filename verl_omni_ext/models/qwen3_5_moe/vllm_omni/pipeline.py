"""
Qwen3.5-MoE vllm-omni pipeline 拓扑定义（frozen）

这是真正的推理拓扑知识——stage 数量、execution_type、input_sources 等。
verl-omni 侧的 rollout.py（槽位②）只是转发这里的定义。

Stage 0: Thinker — 多模态理解 + 文本生成
（如果需要 talker/code2wav stage，在这里追加）
"""
from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_5_MOE_PIPELINE = PipelineConfig(
    model_type="qwen3_5_moe",
    default_deploy_config_name="qwen3_5_moe.yaml",
    model_arch="Qwen3_5MoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
        ),
    ),
)

# thinker_only 模式的 stage 子集（只有 thinker，不含 talker/code2wav）
QWEN3_5_MOE_THINKER_ONLY_STAGES = (
    QWEN3_5_MOE_PIPELINE.stages[0],
)
