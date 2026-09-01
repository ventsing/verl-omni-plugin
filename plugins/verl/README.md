# verl Plugin

Extensions for the verl reinforcement learning framework.

## Features

### Platform Extension
- Custom hardware platform support
- Platform-specific optimizations

### Trainer Extensions
- **FullDuplexTrainer**: Full-duplex training with simultaneous training and inference
- **AsyncTrainerEnhanced**: Enhanced asynchronous training with dynamic scheduling

### Worker Extensions
- **EnhancedEngineWorkerGroup**: Multimodal input processing (text, audio, image)
- Optimized rollout workers

### Distributed Communication
- Multimodal communicators with per-modality compression
- Async communication primitives

### Data Processing
- Multimodal data processor
- Audio feature extraction
- Data validation and cleaning

### Reward Framework
- **MultimodalRewardManager**: Compute rewards for multiple modalities
- Audio quality assessment
- Custom reward fusion strategies

## Usage

```python
from plugins.verl import FullDuplexTrainer, MultimodalRewardManager

# Use full-duplex trainer
trainer = FullDuplexTrainer(config)
trainer.run_duplex_training()

# Use multimodal reward manager
reward_manager = MultimodalRewardManager(config)
reward = reward_manager.compute_multimodal_reward(outputs, targets)
```

## Configuration

```yaml
plugin:
  verl:
    enabled: true
    full_duplex:
      enabled: true
      weight_sync_interval: 10
    reward:
      audio_weight: 0.3
      visual_weight: 0.4
      text_weight: 0.3
```

## Modification Points

This plugin modifies the following verl components:
- `verl/plugin/platform/platform_base.py`: Custom platform
- `verl/trainer/ppo/v1/trainer_base.py`: Full-duplex training
- `verl/experimental/fully_async_policy/fully_async_trainer.py`: Async training
- `verl/workers/engine_workers.py`: Multimodal workers
- `verl/distributed/parallel_state.py`: Communication optimizations
- `verl/protocol.py`: Data processing
- `verl/experimental/reward_loop/reward_manager/base.py`: Reward framework
