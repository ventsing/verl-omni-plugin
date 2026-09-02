"""
全双工 trainer 模块

通过 @register_trainer("omni_fullduplex") 注册到 verl 的 trainer 注册表。
"""
from .fullduplex_trainer import OmniPPOTrainerFullDuplex  # noqa: F401
