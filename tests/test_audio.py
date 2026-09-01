"""
Tests for audio processing utilities.
"""

import pytest
import torch

from shared.audio import AudioProcessor, AudioFeatureExtractor, AudioQualityModel


class TestAudioProcessor:
    """Tests for AudioProcessor."""
    
    def test_init(self):
        """Test initialization."""
        config = {
            "sample_rate": 16000,
            "n_mels": 80,
        }
        processor = AudioProcessor(config)
        assert processor.sample_rate == 16000
        assert processor.n_mels == 80
    
    def test_preprocess(self):
        """Test audio preprocessing."""
        config = {
            "sample_rate": 16000,
            "n_mels": 80,
            "n_fft": 1024,
            "hop_length": 256,
        }
        processor = AudioProcessor(config)
        
        # Create dummy audio
        audio = torch.randn(2, 16000)  # [batch, time]
        
        # Preprocess
        features = processor.preprocess(audio)
        
        # Check output shape
        assert features.dim() == 3  # [batch, n_mels, time]
        assert features.size(0) == 2
        assert features.size(1) == 80
    
    def test_normalize(self):
        """Test audio normalization."""
        config = {}
        processor = AudioProcessor(config)
        
        # Create audio with different scales
        audio1 = torch.randn(2, 16000) * 10.0
        audio2 = torch.randn(2, 16000) * 0.1
        
        # Normalize
        normalized1 = processor._normalize(audio1)
        normalized2 = processor._normalize(audio2)
        
        # Check that max value is 1.0
        assert normalized1.abs().max() <= 1.0
        assert normalized2.abs().max() <= 1.0


class TestAudioFeatureExtractor:
    """Tests for AudioFeatureExtractor."""
    
    def test_init(self):
        """Test initialization."""
        config = {
            "input_dim": 80,
            "hidden_dim": 256,
            "output_dim": 512,
        }
        extractor = AudioFeatureExtractor(config)
        assert extractor.input_dim == 80
        assert extractor.output_dim == 512
    
    def test_forward(self):
        """Test forward pass."""
        config = {
            "input_dim": 80,
            "hidden_dim": 256,
            "output_dim": 512,
            "num_layers": 3,
        }
        extractor = AudioFeatureExtractor(config)
        
        # Create dummy input
        features = torch.randn(2, 80, 100)  # [batch, input_dim, time]
        
        # Forward pass
        output = extractor(features)
        
        # Check output shape
        assert output.shape == (2, 512)  # [batch, output_dim]


class TestAudioQualityModel:
    """Tests for AudioQualityModel."""
    
    def test_init(self):
        """Test initialization."""
        model = AudioQualityModel()
        assert model is not None
    
    def test_evaluate(self):
        """Test quality evaluation."""
        model = AudioQualityModel()
        
        # Create dummy audio
        output = torch.randn(2, 80, 100)
        target = torch.randn(2, 80, 100)
        
        # Evaluate
        metrics = model.evaluate(output, target)
        
        # Check metrics
        assert "mcd" in metrics
        assert "f0_correlation" in metrics
        assert "spectral_loss" in metrics
        assert "overall" in metrics
        
        # Check ranges
        assert metrics["overall"] >= 0.0
        assert metrics["overall"] <= 1.0
    
    def test_compute_mcd(self):
        """Test MCD computation."""
        model = AudioQualityModel()
        
        # Create identical audio (should have low MCD)
        audio = torch.randn(2, 80, 100)
        mcd = model.compute_mcd(audio, audio)
        
        # MCD should be low for identical audio
        assert mcd < 1.0
    
    def test_compute_f0_correlation(self):
        """Test F0 correlation computation."""
        model = AudioQualityModel()
        
        # Create identical audio (should have high correlation)
        audio = torch.randn(2, 80, 100)
        correlation = model.compute_f0_correlation(audio, audio)
        
        # Correlation should be high for identical audio
        assert correlation > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
