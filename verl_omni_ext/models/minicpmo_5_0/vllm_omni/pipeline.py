"""
MiniCPM-o vllm-omni pipeline 拓扑定义（frozen）

Stage 0: Thinker — 多模态理解 + 文本生成
"""
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MINICPMO_5_0_PIPELINE = PipelineConfig(
    model_type="minicpmo_5_0",
    default_deploy_config_name="minicpmo_5_0.yaml",
    model_arch="MiniCPMO",
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

MINICPMO_5_0_THINKER_ONLY_STAGES = (
    MINICPMO_5_0_PIPELINE.stages[0],
)
