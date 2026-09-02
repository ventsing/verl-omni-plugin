"""
L2: MiniCPM-o 5.0 monkey patch（4 个补丁）

所有补丁打的都是第三方代码（transformers / checkpoint remote code / 模型实例），
这是 L2 的正确用法。

补丁放哪一层由它作用的对象的生命周期决定：
- apply_minicpmo_auto_register_guard: 打 transformers AutoClass → 必须在包 import 期（模块级）
- apply_minicpmo_automodel_fallback: 打 transformers AutoModelForMultimodalLM → 包 import 期
- build_minicpmo_forward_adapter: 打模型实例 forward → configure_model 里
- build_minicpmo_processor: 不算补丁，是槽位① configure_processor 的正常实现
"""
import logging
import os
from typing import Any

from verl_omni_ext._patchkit import idempotent_patch, assert_target

logger = logging.getLogger(__name__)


# ============================================================================
# 补丁 1：AutoClass register guard
# ============================================================================
@idempotent_patch(
    name="minicpmo_auto_register_guard",
    target_module="transformers",
    target_attr="AutoImageProcessor",
    fingerprint="transformers>=4.46",
    expected_signature="register",
    probe_signature=True,  # 记录 register 签名指纹，跨进程一致性校验用
)
def apply_minicpmo_auto_register_guard(original_register, __patch_signature__=None):
    """守卫 6 个 AutoClass 的 register 方法

    原因：checkpoint 的 processing_minicpmo.py:478 用 str 当 config class 传，
    新版 transformers 直接 AttributeError，连带 model 都 import 不进来。

    为什么替换是安全的：这个 register 本来就是死代码——mapping 按 config class
    查，str key 永远命中不了。所以把它变成 no-op 不会改变任何运行时行为。

    __patch_signature__: 原 register 的签名指纹（probe_signature=True 注入）。
    如果 transformers 升级后 register 签名变了，这里能看到指纹变化。
    """
    _ = __patch_signature__  # 保留注入：签名漂移时可在此分支适配
    def guarded_register(self, config_class, *args, **kwargs):
        if isinstance(config_class, str):
            # str key 永远命中不了 mapping，跳过即可
            return
        return original_register(self, config_class, *args, **kwargs)
    return guarded_register


# ============================================================================
# 补丁 2：AutoModelForMultimodalLM fallback
# ============================================================================
@idempotent_patch(
    name="minicpmo_automodel_fallback",
    target_module="transformers",
    target_attr="AutoModelForMultimodalLM",
    fingerprint="transformers>=4.46",
    expected_signature="from_pretrained",
    probe_signature=True,  # 记录 from_pretrained 签名指纹，跨进程一致性校验用
)
def apply_minicpmo_automodel_fallback(original_from_pretrained, __patch_signature__=None):
    """fallback：checkpoint 的 auto_map 缺 AutoModelForMultimodalLM 键

    原因：checkpoint 的 auto_map 只声明了 AutoModel，
    verl 调 AutoModelForMultimodalLM.from_pretrained 时找不到键就炸。
    fallback 到 AutoModel.from_pretrained。

    __patch_signature__: 原 from_pretrained 的签名指纹。
    """
    _ = __patch_signature__
    def patched_from_pretrained(cls, *args, **kwargs):
        try:
            return original_from_pretrained(cls, *args, **kwargs)
        except (KeyError, ValueError) as e:
            if "AutoModelForMultimodalLM" in str(e) or "auto_map" in str(e):
                logger.warning("AutoModelForMultimodalLM not in auto_map, falling back to AutoModel")
                from transformers import AutoModel
                return AutoModel.from_pretrained(*args, **kwargs)
            raise
    return patched_from_pretrained


# ============================================================================
# 补丁 3：forward 签名适配器（打模型实例，不走 @idempotent_patch）
# ============================================================================
def build_minicpmo_forward_adapter(module):
    """适配 forward 签名

    remote code 是 forward(self, data, **kw) 单位置字典约定，
    verl 是全关键字调用——签名不兼容，到第一个 micro-batch 才炸。

    打的是 from_pretrained 返回的实例，所以在 configure_model 里调用。
    返回 bool 供自检。
    """
    original_forward = module.forward

    def adapted_forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict) and not kwargs:
            # verl 全关键字调用，但 remote code 期望单位置字典
            return original_forward(args[0])
        return original_forward(*args, **kwargs)

    import types
    module.forward = types.MethodType(adapted_forward, module)
    logger.info("MiniCPM-o forward adapter installed")
    return True


# ============================================================================
# 补丁 4：processor（其实不算补丁，是槽位① configure_processor 的正常实现）
# ============================================================================
def build_minicpmo_processor(model_path: str, model_config) -> Any:
    """绕过 verl.utils.hf_processor，直接 AutoProcessor.from_pretrained

    原因：verl.utils.hf_processor 用白名单 match processor 类名，
    MiniCPM-o 的类名不在六个已知里 → 走 raise ValueError → 外层 except 吞成 None。
    """
    disable_fallback = os.environ.get("MINICPMO_DISABLE_PROCESSOR_FALLBACK", "0") == "1"
    if disable_fallback:
        # 这是排障用的二分开关：关掉它如果 filter 还是 0，那就是数据的问题
        from verl.utils.hf_processor import hf_processor
        return hf_processor(model_path, trust_remote_code=model_config.trust_remote_code)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=model_config.trust_remote_code
    )
    logger.info("MiniCPM-o processor built directly via AutoProcessor (bypassed hf_processor)")
    return processor
