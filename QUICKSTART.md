# verl-omni-plugin 快速开始指南

## 📦 安装

```bash
# 进入项目目录
cd verl-omni-plugin

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装插件（开发模式）
pip install -e .

# 安装所有依赖
pip install -e ".[all]"
```

## 🚀 快速使用

### 1. 基础音频处理

```python
from shared.audio import AudioProcessor
import torch

# 创建音频处理器
config = {
    "sample_rate": 16000,
    "n_mels": 80,
}
processor = AudioProcessor(config)

# 处理音频
audio = torch.randn(2, 16000)  # [batch, time]
features = processor.preprocess(audio)
print(f"Features shape: {features.shape}")  # [batch, n_mels, time]
```

### 2. 音频模型

```python
from plugins.verl_omni.models.audio import AudioHead
import torch

# 创建音频头
config = {
    "n_mels": 80,
    "hidden_size": 512,
    "audio_length": 100,
}
audio_head = AudioHead(config)

# 编码音频
audio_input = torch.randn(2, 80, 100)
features = audio_head(audio_input, mode="encode")
print(f"Encoded shape: {features.shape}")  # [batch, hidden_size]

# 解码音频
decoded = audio_head(features, mode="decode")
print(f"Decoded shape: {decoded.shape}")  # [batch, n_mels, audio_length]
```

### 3. 多模态 Reward

```python
from plugins.verl.reward import MultimodalRewardManager
import torch

# 创建 reward 管理器
config = {
    "audio_weight": 0.3,
    "visual_weight": 0.4,
    "text_weight": 0.3,
}
reward_manager = MultimodalRewardManager(config)

# 计算多模态 reward
outputs = {
    "text": "generated text",
    "audio": torch.randn(2, 80, 100),
}
targets = {
    "text": "target text",
    "audio": torch.randn(2, 80, 100),
}

reward = reward_manager.compute_multimodal_reward(outputs, targets)
print(f"Reward: {reward:.4f}")
```

### 4. 全双工训练

```python
import asyncio
from plugins.verl.trainer import FullDuplexTrainer

async def train():
    config = {
        "duplex_enabled": True,
        "weight_sync_interval": 10,
    }
    
    # 创建训练器
    trainer = FullDuplexTrainer(config)
    
    # 添加训练数据
    for i in range(100):
        await trainer.training_queue.put({"batch_id": i})
    
    # 运行全双工训练
    await trainer.run_duplex_training()

asyncio.run(train())
```

### 5. 流式音频推理

```python
import asyncio
import torch
from plugins.vllm_omni.pipelines import AudioStreamingPipeline

async def stream():
    config = {
        "sample_rate": 16000,
        "n_mels": 80,
        "chunk_size": 10,
    }
    
    # 创建流式管道
    pipeline = AudioStreamingPipeline(config)
    
    # 创建音频流
    async def audio_stream():
        for i in range(50):
            yield torch.randn(1, 80, 10)
            await asyncio.sleep(0.01)
    
    # 处理流
    async for output in pipeline.stream_infer(audio_stream()):
        print(f"Processed chunk: {output.keys()}")

asyncio.run(stream())
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_audio.py -v

# 运行带覆盖率的测试
pytest tests/ --cov=shared --cov=plugins -v
```

## 📝 运行示例

```bash
# 运行音频训练示例
python examples/audio_training_example.py
```

## 🔧 配置

创建配置文件 `config.yaml`:

```yaml
plugin:
  enabled: true
  
  audio:
    sample_rate: 16000
    n_mels: 80
    hidden_size: 512
  
  full_duplex:
    enabled: true
    weight_sync_interval: 10
  
  reward:
    audio_weight: 0.3
    visual_weight: 0.4
    text_weight: 0.3
```

在代码中加载配置:

```python
from shared.utils import load_config

config = load_config("config.yaml")
```

## 📚 更多文档

- [verl 插件文档](plugins/verl/README.md)
- [verl-omni 插件文档](plugins/verl_omni/README.md)
- [vllm 插件文档](plugins/vllm/README.md)
- [vllm-omni 插件文档](plugins/vllm_omni/README.md)

## ❓ 常见问题

### Q: 如何只启用特定插件？

A: 使用环境变量控制：

```bash
export VERL_OMNI_PLUGIN_ENABLE_VERL=1
export VERL_OMNI_PLUGIN_ENABLE_VERL_OMNI=1
export VERL_OMNI_PLUGIN_ENABLE_VLLM=0
export VERL_OMNI_PLUGIN_ENABLE_VLLM_OMNI=0
```

### Q: 如何添加自定义的 patch？

A: 在对应的 `patches.py` 文件中添加：

```python
from shared.patch_manager import BasePatchManager

class MyPatchManager(BasePatchManager):
    @classmethod
    def register_all_patches(cls):
        cls.register_patch(
            name="my_patch",
            target_module="target.module",
            target_attr="TargetClass",
            replacement_fn="my_plugin.module:MyClass",
            description="My custom patch",
        )
```

### Q: 如何测试我的插件？

A: 参考 `tests/` 目录下的测试文件，使用 pytest 编写测试。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache-2.0
