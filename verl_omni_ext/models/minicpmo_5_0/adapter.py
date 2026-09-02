"""
槽位①: MiniCPM-o 5.0 Thinker 训练适配器

@OmniModelBase.register("MiniCPMO", stage="thinker")

注意：不调用 apply_qwen3_5_vision_device_fix——那打的是 Qwen3_5MoeVisionModel，
MiniCPM-o 用的是自己的 UHD NaViT，跨模型复制适配器时这类"看起来能抄其实不能抄"的项最危险。
"""
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

from .patches import build_minicpmo_forward_adapter, build_minicpmo_processor


@OmniModelBase.register("MiniCPMO", stage="thinker")
class MiniCPMO50ThinkerAdapter(OmniModelBase):
    """MiniCPM-o 5.0 Thinker"""

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """不剥离 tts.*——保持 state_dict 与 checkpoint 逐字相同"""
        return []

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """必须自建——verl.utils.hf_processor 的 match 默认分支直接 raise

        上游 match 用白名单，MiniCPM-o 的 processor 类名不在六个已知里 → raise ValueError
        → 外层 except 吞成 None → 数据集被过滤到 0 行 → 只在日志刷 warning。
        所以必须绕过 hf_processor，直接 AutoProcessor.from_pretrained。
        """
        return build_minicpmo_processor(model_path, model_config)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        from verl.utils.hf_tokenizer import hf_tokenizer
        return hf_tokenizer(model_path, trust_remote_code=True)

    @classmethod
    def configure_model(cls, module, model_config):
        """装 forward 签名适配器 + MTP 汇报

        forward 适配器打的是 from_pretrained 返回的实例，放在 configure_model 里恰好正确。
        """
        module = super().configure_model(module, model_config)
        build_minicpmo_forward_adapter(module)
        # MTP 头汇报（未使用但需报告显存）
        _report_mtp(module, model_config)
        return module


def _report_mtp(module, model_config):
    """汇报 MTP 头的显存占用（MINICPMO_FREEZE_MTP=1 时冻结）"""
    import os
    if hasattr(module, "mtp"):
        freeze = os.environ.get("MINICPMO_FREEZE_MTP", "0") == "1"
        if freeze:
            for p in module.mtp.parameters():
                p.requires_grad = False
