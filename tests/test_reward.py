"""
Tests for multimodal reward manager.
"""

import pytest
import torch

from plugins.verl.reward import MultimodalRewardManager


class TestMultimodalRewardManager:
    """Tests for MultimodalRewardManager."""
    
    def test_init(self):
        """Test initialization."""
        config = {
            "audio_weight": 0.3,
            "visual_weight": 0.4,
            "text_weight": 0.3,
        }
        manager = MultimodalRewardManager(config)
        assert manager.audio_weight == 0.3
        assert manager.visual_weight == 0.4
        assert manager.text_weight == 0.3
    
    def test_compute_text_reward(self):
        """Test text reward computation."""
        config = {}
        manager = MultimodalRewardManager(config)
        
        # Test exact match
        reward = manager._compute_text_reward("hello", "hello")
        assert reward == 1.0
        
        # Test mismatch
        reward = manager._compute_text_reward("hello", "world")
        assert reward == 0.0
    
    def test_compute_audio_reward(self):
        """Test audio reward computation."""
        config = {}
        manager = MultimodalRewardManager(config)
        
        # Create dummy audio
        output = torch.randn(2, 80, 100)
        target = torch.randn(2, 80, 100)
        
        # Compute reward
        reward = manager._compute_audio_reward(output, target)
        
        # Check range
        assert reward >= 0.0
        assert reward <= 1.0
    
    def test_compute_visual_reward(self):
        """Test visual reward computation."""
        config = {}
        manager = MultimodalRewardManager(config)
        
        # Create identical images (should have high reward)
        image = torch.randn(2, 3, 224, 224)
        reward = manager._compute_visual_reward(image, image)
        
        # Reward should be high for identical images
        assert reward > 0.9
    
    def test_fuse_rewards(self):
        """Test reward fusion."""
        config = {
            "audio_weight": 0.3,
            "visual_weight": 0.4,
            "text_weight": 0.3,
        }
        manager = MultimodalRewardManager(config)
        
        # Test with all modalities
        rewards = {
            "text": 1.0,
            "audio": 0.8,
            "visual": 0.9,
        }
        fused = manager._fuse_rewards(rewards)
        
        # Check that fused reward is weighted average
        expected = 0.3 * 1.0 + 0.4 * 0.9 + 0.3 * 0.8
        assert abs(fused - expected) < 0.01
    
    def test_compute_multimodal_reward(self):
        """Test full multimodal reward computation."""
        config = {
            "audio_weight": 0.3,
            "visual_weight": 0.4,
            "text_weight": 0.3,
        }
        manager = MultimodalRewardManager(config)
        
        # Create outputs and targets
        outputs = {
            "text": "hello",
            "audio": torch.randn(2, 80, 100),
        }
        targets = {
            "text": "hello",
            "audio": torch.randn(2, 80, 100),
        }
        
        # Compute reward
        reward = manager.compute_multimodal_reward(outputs, targets)
        
        # Check that reward is in valid range
        assert reward >= 0.0
        assert reward <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
