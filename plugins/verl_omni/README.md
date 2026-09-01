# verl-omni Plugin

Extensions for verl-omni multimodal RL training framework.

## Features

### Audio Models
- **AudioHead**: Audio processing head for encoding/decoding
- **AudioEncoder**: Neural network for audio feature extraction
- **AudioDecoder**: Network for audio generation

### Omni Models
- **CustomOmniModelAdapter**: Multimodal model with attention-based fusion
- Support for text, image, and audio modalities

### Training Pipelines
- **AudioFlowGRPO**: Flow-GRPO pipeline for audio diffusion models
- **FullDuplexOmni**: Full-duplex training for omni models

### Reward Management
- **AudioRewardManager**: Audio quality assessment and reward computation
- Multimodal reward fusion

### Agent Loops
- Audio agent loop for interactive audio generation
- Multimodal agent loop

## Usage

```python
from plugins.verl_omni import AudioHead, AudioRewardManager

# Create audio head
audio_head = AudioHead(config)
audio_features = audio_head(audio_tensor)

# Use audio reward manager
reward_manager = AudioRewardManager(config)
reward = reward_manager.compute_reward(outputs, targets)
```

## Configuration

```yaml
plugin:
  verl_omni:
    enabled: true
    audio:
      sample_rate: 16000
      n_mels: 80
    reward:
      audio_weight: 0.3
      visual_weight: 0.4
      text_weight: 0.3
```

## Modification Points

This plugin modifies the following verl-omni components:
- `verl_omni/models/transformers/qwen3_omni_thinker.py`: Audio head
- `verl_omni/pipelines/model_base.py`: Omni model base
- `verl_omni/pipelines/audio_flow_grpo/`: Audio Flow-GRPO pipeline
- `verl_omni/reward_loop/reward_manager/multi.py`: Audio reward
- `verl_omni/agent_loop/diffusion_agent_loop.py`: Audio agent loop
