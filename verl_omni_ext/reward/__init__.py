"""
Reward 扩展模块

两个扩展点：
  1. RewardManager 注册表 —— @register("xxx") 注册到 verl 的 REWARD_MANAGER 字典
  2. custom_reward_function —— config 里 path=pkg://verl_omni_ext.reward.functions.xxx

配置方式（config.yaml）：
  # 方式 1：注册 reward manager
  reward:
    reward_model:
      reward_manager: my_audio_reward    # @register 的名字

  # 方式 2：自定义 reward 函数路径
  reward:
    custom_reward_function:
      path: pkg://verl_omni_ext.reward.functions.audio_quality
      name: compute_score
"""
from . import managers, functions  # noqa: F401
