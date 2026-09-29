"""
Qwen3.5-MoE vllm-omni 侧注册（原生插件机制）

加载链路（零 patch，pip install verl-omni-ext 后自动生效）：

  vllm_omni.general_plugins 入口点
    → verl_omni_ext.vllm_omni_plugins.register()
    → 本模块 register()

注册内容：
  1. ``register_pipeline(QWEN3_5_MOE_PIPELINE)`` —— stage 拓扑进 OMNI_PIPELINES
  2. （暂缓）模型类注册 —— 上游 ``_OMNI_MODELS`` 没有
     ``Qwen3_5MoeForConditionalGeneration``，本包落地 vllm 模型实现后
     解开 ``_maybe_register_model_class`` 里的注册代码。

历史：本文件曾用 GP-004（VLLM_OMNI_EXTERNAL_MODULES + 往 ``_OMNI_MODELS``
塞元组）注册——元组会被硬编码前缀拼接成 vllm_omni 树内路径，外部包
路径拼不进去，惰性导入必炸。详见 verl_omni_ext/vllm_omni_plugins.py 的
说明。GP-004 已退役（gates/ledger.md）。
"""

from .pipeline import QWEN3_5_MOE_PIPELINE

_ARCH = "Qwen3_5MoeForConditionalGeneration"
_LAZY_PATH = "verl_omni_ext.models.qwen3_5_moe.vllm_omni.model:Qwen3_5MoeModel"

_registered = False


def register():
    """注册 pipeline 拓扑 +（若提供实现）模型类。幂等。"""
    global _registered
    if _registered:
        return

    from vllm_omni.config.pipeline_registry import register_pipeline

    register_pipeline(QWEN3_5_MOE_PIPELINE)
    _maybe_register_model_class()
    _registered = True


def _maybe_register_model_class():
    """新架构（上游注册表没有）需要模型类注册。本包暂无 vllm 实现。

    实现落地后（本目录加 model.py），解开下面的注册。两处都要注册：
      - OmniModelRegistry（vllm-omni 的 ModelConfig.registry 消费）
      - vllm 上游 ModelRegistry（entrypoints 走全局表解析时消费；
        vllm-omni 自己在 engine/arg_utils.py:138 就这么镜像同步）
    惰性 "module:Class" 字符串支持外部包路径——这是原生 API，
    与 GP-004 元组方案的本质区别。
    """
    # from vllm_omni.model_executor.models import OmniModelRegistry
    # from vllm.model_executor.models.registry import ModelRegistry
    # OmniModelRegistry.register_model(_ARCH, _LAZY_PATH)
    # ModelRegistry.register_model(_ARCH, _LAZY_PATH)
    return None
