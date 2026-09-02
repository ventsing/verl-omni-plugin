"""
Qwen3.5-Omni Thinker 训练适配器

架构：Qwen3.5 LLM + Whisper 音频编码器
注册键：("Qwen35OmniForConditionalGeneration", "thinker")

零侵入机制：
    1. 用户在 config.yaml 设置 external_lib: qwen35_whisper_plugin
    2. verl-omni 调用 import_external_libs() → importlib.import_module()
    3. 本文件被 import，@OmniModelBase.register() 装饰器执行
    4. 适配器注册到 OmniModelBase._registry 字典
    5. verl-omni 通过 get_class_by_name() 查找并调用适配器方法

参考实现（verl-omni 自带）：
    verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py
    @OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
"""

import json
import logging
import os
from typing import Any

import numpy as np

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


@OmniModelBase.register("Qwen35OmniForConditionalGeneration", stage="thinker")
class Qwen35ThinkerAdapter(OmniModelBase):
    """Qwen3.5 + Whisper 的 Thinker 训练适配器

    必须实现的 3 个抽象方法：
        get_strip_modules()   - 返回需要剥离的模块
        configure_processor() - 加载多模态处理器
        configure_tokenizer() - 加载 tokenizer

    可选覆盖的方法：
        configure_model()     - 有默认实现（剥离模块），可覆盖做更多定制

    注册键说明：
        "Qwen35OmniForConditionalGeneration" 必须与模型 config.json 中
        architectures[0] 完全一致，否则 verl-omni 查找不到适配器。
    """

    # ==================================================================
    # 必须实现的抽象方法（来自 OmniModelBase）
    # 参考：verl_omni/pipelines/model_base.py 第 562-621 行
    # ==================================================================

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """返回需要剥离的模块名

        Qwen3.5-Omni 包含 thinker/talker/codec 等组件。
        thinker-only 训练时剥离 talker 和 codec 以节省显存。

        verl-omni 在 configure_model() 的默认实现中会 delattr 这些模块。
        参考：verl_omni/pipelines/model_base.py 第 643-646 行
        """
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """加载并配置多模态处理器

        由 verl-omni 在 OmniModelConfig.__post_init__() 第 186 行调用。
        必须返回一个提供以下能力的 processor 对象：
            - apply_chat_template
            - __call__ (处理文本+音频+图像输入)
            - tokenizer
            - chat_template

        参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py
              的 configure_processor() 方法
        """
        import types

        from transformers import AutoConfig, AutoProcessor

        # 1. 加载 processor 和 config
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
        )

        # 2. 切换到 thinker_config（Qwen3.5-Omni 将多模态设置嵌套在子配置中）
        if hasattr(config, "thinker_config"):
            processor.config = config.thinker_config
            # 视觉相关的 token id 可能在 talker_config 中
            if hasattr(config, "talker_config"):
                processor.config.vision_start_token_id = (
                    config.talker_config.vision_start_token_id
                )

        # 3. 绑定 RoPE 索引方法（处理音频/视觉的位置编码）
        # 参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py 第 98-102 行
        def _get_rope_index(self, *args, **kwargs):
            """为多模态输入生成 RoPE 位置编码

            音频 token 需要特殊的位置编码处理。
            实际实现需要参考 Qwen3.5 的 get_rope_index 逻辑。
            """
            import torch

            input_ids = args[0] if args else kwargs.get("input_ids")
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, device=input_ids.device)
            return position_ids.unsqueeze(0).expand(batch_size, -1), None

        processor.get_rope_index = types.MethodType(_get_rope_index, processor)

        # 4. 提供 audio_seqlens 给 verl 的 agent loop
        # 参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py 第 106-112 行
        def _get_rope_index_kwargs(multi_modal_inputs: dict) -> dict:
            """从多模态输入中提取音频长度信息"""
            feature_attention_mask = multi_modal_inputs.get("feature_attention_mask")
            if feature_attention_mask is not None:
                return {"audio_seqlens": feature_attention_mask.sum(-1)}
            return {}

        processor.get_rope_index_kwargs = _get_rope_index_kwargs

        # 5. 绑定去重 pad token 方法
        # vLLM-Omni 会重新展开 pad token，所以需要先去重
        # 参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py 第 117-142 行
        def _dedup_pad_tokens(self, prompt_ids: list[int]) -> list[int]:
            """去除连续的重复 pad token（image_token/audio_token/video_token）"""
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return prompt_ids

            pad_ids = set()
            for attr in ("image_token", "video_token", "audio_token"):
                token = getattr(self, attr, None)
                if token is None:
                    continue
                try:
                    tid = tokenizer.convert_tokens_to_ids(token)
                except Exception:
                    continue
                if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
                    continue
                pad_ids.add(int(tid))

            if not pad_ids:
                return prompt_ids

            arr = np.asarray(prompt_ids, dtype=np.int64)
            if arr.size == 0:
                return prompt_ids

            is_pad = np.isin(arr, list(pad_ids))
            keep = np.ones(arr.size, dtype=bool)
            same_as_prev = is_pad[1:] & is_pad[:-1] & (arr[1:] == arr[:-1])
            keep[1:] &= ~same_as_prev
            return arr[keep].tolist()

        processor.dedup_pad_tokens = types.MethodType(_dedup_pad_tokens, processor)

        logger.info(f"Qwen3.5-Omni processor configured from {model_path}")
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """加载并配置 tokenizer

        由 verl-omni 在 OmniModelConfig.__post_init__() 第 185 行调用。
        参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py
              的 configure_tokenizer() 方法
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code,
        )

        # 加载 chat template（可能在单独的 JSON 文件中）
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r", encoding="utf-8") as f:
                chat_template_data = json.load(f)
            tokenizer.chat_template = chat_template_data.get("chat_template")
            logger.info("Loaded chat_template.json")

        return tokenizer

    # ==================================================================
    # 可选覆盖的方法（有默认实现）
    # 参考：verl_omni/pipelines/model_base.py 第 623-647 行
    # ==================================================================

    @classmethod
    def configure_model(cls, module, model_config):
        """配置模型：在 FSDP 包装前调用

        默认实现只做剥离（调用 get_strip_modules 返回的模块）。
        这里额外做 forward 重定向到 thinker。

        参考：verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py
              的 configure_model() 方法（第 48-64 行）
        """
        # 1. 调用父类默认实现（剥离模块）
        module = super().configure_model(module, model_config)

        # 2. 重定向 forward 到 thinker
        if hasattr(module, "thinker"):
            logger.info("Redirecting forward to thinker")
            module.forward = module.thinker.forward
            module.get_input_embeddings = module.thinker.get_input_embeddings
            module.set_input_embeddings = module.thinker.set_input_embeddings
            # FSDP 不分割的模块
            module._no_split_modules = [
                "Qwen35DecoderLayer",
                "WhisperEncoderLayer",
            ]

        return module

    # ==================================================================
    # 可选的辅助方法
    # ==================================================================

    @classmethod
    def get_lora_target_modules(cls, model_config) -> list[str]:
        """LoRA 目标模块（用于 PEFT 训练）"""
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
