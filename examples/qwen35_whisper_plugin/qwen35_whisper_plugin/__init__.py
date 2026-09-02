"""
Qwen3.5 + Whisper 插件 for verl-omni

零侵入实现：通过 verl-omni 的 external_lib 机制加载。

机制说明：
    verl-omni 在 OmniModelConfig.__post_init__() 中会调用：
        import_external_libs(self.external_lib)
    等价于：
        importlib.import_module("qwen35_whisper_plugin")
    即执行本文件。

    本文件只需要 import 适配器模块，就会触发
    @OmniModelBase.register() 装饰器执行，将适配器注册到
    OmniModelBase._registry 字典中。

    之后 verl-omni 通过 OmniModelBase.get_class_by_name()
    从注册表查找并调用适配器的方法。

使用方式（在 config.yaml 中设置）：
    actor_rollout_ref:
      model:
        path: /path/to/qwen35-omni
        external_lib: qwen35_whisper_plugin
        model_type: omni_model
        model_stage: thinker
"""

# import 适配器模块，触发 @OmniModelBase.register() 注册
from .models import qwen35_thinker  # noqa: F401

__version__ = "0.1.0"

__all__ = ["__version__"]
