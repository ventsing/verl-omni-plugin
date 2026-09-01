"""
Example: Audio model training with full-duplex support.

This example demonstrates how to use the verl-omni-plugin for:
1. Audio model training
2. Full-duplex training (simultaneous training and inference)
3. Multimodal reward computation
"""

import asyncio
import logging

import torch

# Import plugin components
from plugins.verl import FullDuplexTrainer, MultimodalRewardManager
from plugins.verl_omni import AudioHead, AudioRewardManager
from shared.audio import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_audio_training():
    """Example: Train an audio model with full-duplex support."""
    
    # Configuration
    config = {
        "sample_rate": 16000,
        "n_mels": 80,
        "hidden_size": 512,
        "audio_length": 100,
        "duplex_enabled": True,
        "weight_sync_interval": 10,
        "audio_weight": 0.3,
        "visual_weight": 0.4,
        "text_weight": 0.3,
    }
    
    logger.info("=== Audio Training Example ===")
    
    # 1. Create audio head
    logger.info("Creating audio head...")
    audio_head = AudioHead(config)
    
    # 2. Create dummy audio data
    logger.info("Creating dummy audio data...")
    batch_size = 4
    audio_input = torch.randn(batch_size, 80, 100)  # [batch, n_mels, time]
    
    # 3. Encode audio
    logger.info("Encoding audio...")
    audio_features = audio_head(audio_input, mode="encode")
    logger.info(f"Audio features shape: {audio_features.shape}")
    
    # 4. Decode audio
    logger.info("Decoding audio...")
    decoded_audio = audio_head(audio_features, mode="decode")
    logger.info(f"Decoded audio shape: {decoded_audio.shape}")
    
    # 5. Create reward manager
    logger.info("Creating reward manager...")
    reward_manager = MultimodalRewardManager(config)
    
    # 6. Compute multimodal reward
    logger.info("Computing multimodal reward...")
    outputs = {
        "text": "generated text",
        "audio": decoded_audio,
    }
    targets = {
        "text": "target text",
        "audio": audio_input,
    }
    
    reward = reward_manager.compute_multimodal_reward(outputs, targets)
    logger.info(f"Multimodal reward: {reward:.4f}")
    
    # 7. Create full-duplex trainer
    logger.info("Creating full-duplex trainer...")
    trainer = FullDuplexTrainer(config)
    
    # 8. Add some data to queues
    logger.info("Adding data to training queue...")
    for i in range(10):
        await trainer.training_queue.put({"batch_id": i, "data": torch.randn(4, 512)})
    
    # 9. Run full-duplex training (for a few steps)
    logger.info("Starting full-duplex training...")
    
    # Create a task to stop training after some time
    async def stop_after_delay():
        await asyncio.sleep(2.0)
        trainer.stop()
    
    # Run training and stop task concurrently
    await asyncio.gather(
        trainer.run_duplex_training(),
        stop_after_delay(),
    )
    
    logger.info("Training completed!")


async def example_audio_streaming():
    """Example: Streaming audio inference."""
    
    from plugins.vllm_omni import AudioStreamingPipeline
    
    config = {
        "sample_rate": 16000,
        "n_mels": 80,
        "chunk_size": 10,
    }
    
    logger.info("=== Audio Streaming Example ===")
    
    # Create streaming pipeline
    pipeline = AudioStreamingPipeline(config)
    
    # Create dummy audio stream
    async def audio_stream():
        for i in range(50):
            yield torch.randn(1, 80, 10)
            await asyncio.sleep(0.01)
    
    # Process stream
    logger.info("Processing audio stream...")
    chunk_count = 0
    
    async for output in pipeline.stream_infer(audio_stream()):
        chunk_count += 1
        if chunk_count % 5 == 0:
            logger.info(f"Processed chunk {chunk_count}")
    
    logger.info(f"Streaming completed! Processed {chunk_count} chunks")


async def main():
    """Run all examples."""
    logger.info("Starting verl-omni-plugin examples\n")
    
    # Run audio training example
    await example_audio_training()
    
    logger.info("\n" + "="*50 + "\n")
    
    # Run audio streaming example
    await example_audio_streaming()
    
    logger.info("\n=== All examples completed ===")


if __name__ == "__main__":
    asyncio.run(main())
