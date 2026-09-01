# vllm-omni Plugin

Extensions for vllm-omni multimodal inference.

## Features

### Audio Pipeline
- **AudioInferencePipeline**: Audio inference with preprocessing/postprocessing
- **AudioStreamingPipeline**: Streaming audio inference for full-duplex

### Full-Duplex
- Concurrent inference and generation
- Real-time audio processing

## Usage

```python
from plugins.vllm_omni import AudioInferencePipeline

# Create audio inference pipeline
pipeline = AudioInferencePipeline(config)
output = await pipeline.infer(audio_input)
```

## Configuration

```yaml
plugin:
  vllm_omni:
    enabled: true
    audio:
      streaming: true
      chunk_size: 1024
```

## Modification Points

This plugin modifies the following vllm-omni components:
- `vllm-omni/pipeline/base.py`: Audio inference pipeline
- `vllm-omni/models/registry.py`: Audio model registration
