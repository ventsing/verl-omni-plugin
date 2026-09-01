# verl-omni-plugin

一个用于扩展 verl、verl-omni、vllm 和 vllm-omni 的插件框架，支持音频处理、全双工训练和多模态推理。

## 🎯 特性

- **零侵入**: 不修改上游仓库代码，通过 plugin + monkey-patch 注入特性
- **模块化设计**: 按仓库分离插件，每个插件独立管理
- **共享工具**: 跨插件共享音频处理、Patch 管理等工具
- **版本兼容**: 统一的版本检查和兼容性管理
- **可回滚**: 支持动态启用/禁用 patches

## 📦 安装

```bash
# 基础安装
pip install -e .

# 安装特定仓库的支持
pip install -e ".[verl]"
pip install -e ".[verl-omni]"
pip install -e ".[vllm]"
pip install -e ".[vllm-omni]"

# 安装所有支持
pip install -e ".[all]"
```

## 🚀 快速开始

### 1. 启用插件

```python
# 在训练脚本中导入插件（自动注册所有扩展）
import verl_omni_plugin

# 或者只启用特定仓库的插件
from plugins import verl, verl_omni
```

### 2. 使用音频模型

```python
from plugins.verl_omni.models.audio import AudioHead

# 创建音频处理头
audio_head = AudioHead(config)

# 处理音频输入
audio_features = audio_head(audio_tensor)
```

### 3. 使用全双工训练

```python
from plugins.verl.trainer import FullDuplexTrainer

# 创建全双工训练器
trainer = FullDuplexTrainer(config)

# 运行全双工训练
trainer.run_duplex_training()
```

### 4. 使用多模态 Reward

```python
from plugins.verl_omni.reward_loop import AudioRewardManager

# 创建音频 Reward 管理器
reward_manager = AudioRewardManager(config)

# 计算多模态 Reward
reward = reward_manager.compute_reward(outputs, targets)
```

## 📁 项目结构

```
verl-omni-plugin/
├── shared/                          # 跨插件共享工具
│   ├── patch_manager/               # 统一 Patch 管理器
│   ├── audio/                       # 音频处理工具
│   └── utils/                       # 通用工具
│
└── plugins/                         # 各仓库的插件
    ├── verl/                        # verl 核心仓库插件
    │   ├── platform/                # 平台扩展
    │   ├── trainer/                 # 训练器扩展
    │   ├── workers/                 # Worker 扩展
    │   ├── distributed/             # 分布式通信
    │   ├── data/                    # 数据处理
    │   ├── reward/                  # Reward 框架
    │   ├── patches/                 # Monkey-patch 管理
    │   └── utils/                   # verl 专用工具
    │
    ├── verl_omni/                   # verl-omni 仓库插件
    │   ├── models/                  # 多模态模型
    │   ├── pipelines/               # 训练流水线
    │   ├── reward_loop/             # Reward 循环
    │   ├── trainer/                 # 训练器
    │   ├── agent_loop/              # Agent Loop
    │   ├── workers/                 # Worker
    │   ├── patches/                 # Monkey-patch 管理
    │   └── utils/                   # verl_omni 专用工具
    │
    ├── vllm/                        # vllm 核心仓库插件
    │   ├── platform/                # 平台扩展
    │   ├── model_executor/          # 模型执行器
    │   ├── attention/               # Attention 扩展
    │   ├── distributed/             # 分布式扩展
    │   ├── patches/                 # Monkey-patch 管理
    │   └── utils/                   # vllm 专用工具
    │
    └── vllm_omni/                   # vllm-omni 仓库插件
        ├── pipelines/               # 推理流水线
        ├── models/                  # 多模态模型
        ├── patches/                 # Monkey-patch 管理
        └── utils/                   # vllm_omni 专用工具
```

## 🔧 配置

### 环境变量

```bash
# 启用特定插件
export VERL_USE_EXTERNAL_MODULES=verl_omni_plugin
export VLLM_PLUGINS=verl_omni_plugin

# 启用特定功能
export VERL_OMNI_PLUGIN_ENABLE_AUDIO=1
export VERL_OMNI_PLUGIN_ENABLE_FULL_DUPLEX=1
```

### 配置文件

```yaml
# config/plugin_config.yaml
plugin:
  enabled: true
  
  audio:
    enabled: true
    sample_rate: 16000
    feature_dim: 80
  
  full_duplex:
    enabled: true
    weight_sync_interval: 10
  
  reward:
    audio_weight: 0.3
    visual_weight: 0.4
    text_weight: 0.3
```

## 📝 开发指南

### 添加新的 Patch

```python
# plugins/verl/patches/trainer_patches.py
from shared.patch_manager import BasePatchManager

class VerlTrainerPatches(BasePatchManager):
    @classmethod
    def register_all(cls):
        cls.register_patch(
            name="my_custom_trainer",
            target_module="verl.trainer.ppo.v1.trainer_base",
            target_attr="BaseTrainer",
            replacement_fn="plugins.verl.trainer:MyCustomTrainer",
            version_check=lambda: cls._check_version("verl", ">=0.6.0"),
            description="自定义训练器"
        )
```

### 添加新的音频模型

```python
# plugins/verl_omni/models/audio/my_audio_model.py
from shared.audio import AudioProcessor
from verl_omni.pipelines.model_base import OmniModelBase

@OmniModelBase.register("MyAudioModel", stage="thinker")
class MyAudioModelAdapter(OmniModelBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.processor = AudioProcessor(config)
    
    def process_audio(self, audio):
        return self.processor.extract_features(audio)
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定模块的测试
pytest tests/test_audio/
pytest tests/test_trainer/

# 运行带覆盖率的测试
pytest --cov=plugins --cov=shared tests/
```

## 📚 文档

详细文档请参考各插件目录下的 README.md：

- [verl 插件文档](plugins/verl/README.md)
- [verl-omni 插件文档](plugins/verl_omni/README.md)
- [vllm 插件文档](plugins/vllm/README.md)
- [vllm-omni 插件文档](plugins/vllm_omni/README.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache-2.0
