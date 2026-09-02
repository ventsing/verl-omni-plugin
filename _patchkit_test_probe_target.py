"""
_patchkit 单测用的假目标模块

不依赖 transformers/vllm——让 _patchkit 的多进程传播、签名指纹、
watchdog、strict 模式可以在纯 CPU 环境单测。
"""


class _Inner:
    def method(self, a, b, c=1):
        return a + b + c


class Target:
    """模拟被 patch 的第三方类（如 transformers.AutoImageProcessor）"""

    _verl_omni_ext_patched_test_watchdog_target = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._verl_omni_ext_patched_test_watchdog_target = cls

    method = _Inner.method