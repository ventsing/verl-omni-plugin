"""
MiniCPM-o 5.0 Thinker 适配器

槽位① OmniModelBase | 槽位② OmniRolloutPipelineBase | L2 monkey patch(4个) | 槽位④ 数据集

⚠ 时序陷阱：模块级补丁必须在任何 remote code 被 import 之前执行。
  from_pretrained 在 configure_model 之前（fsdp/omni_impl.py:185），
  所以让 transformers 能 import 得动 remote code 的补丁放在这里，
  不是放在 configure_model 里——那里来不及。
"""
from . import thinker_adapter, rollout_adapter, patches, dataset  # noqa: F401

# 模块级补丁——必须在任何 remote code 被 import 之前执行
apply_minicpmo_auto_register_guard = patches.apply_minicpmo_auto_register_guard
apply_minicpmo_automodel_fallback = patches.apply_minicpmo_automodel_fallback

apply_minicpmo_auto_register_guard()
apply_minicpmo_automodel_fallback()
