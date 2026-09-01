"""
音频支持完整样例

本样例展示如何使用 verl-omni-plugin 进行音频模型训练和推理：
1. 音频数据预处理
2. 音频特征提取
3. 音频模型训练
4. 音频质量评估
5. 多模态（文本+音频）训练

运行方式：
    python examples/audio_support_example.py
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 导入插件组件
from shared.audio import AudioProcessor, AudioFeatureExtractor, AudioQualityModel
from plugins.verl_omni.models.audio import AudioHead, AudioEncoder, AudioDecoder
from plugins.verl_omni.models.omni import CustomOmniModelAdapter
from plugins.verl.reward import MultimodalRewardManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 音频数据集
# ============================================================================

class AudioTextDataset(Dataset):
    """
    音频-文本配对数据集
    
    用于训练多模态模型，每个样本包含：
    - 音频波形
    - 对应的文本描述
    """
    
    def __init__(self, num_samples=100, sample_rate=16000, audio_length=16000):
        """
        初始化数据集
        
        Args:
            num_samples: 样本数量
            sample_rate: 采样率
            audio_length: 音频长度（采样点数）
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        
        # 生成模拟数据（实际应用中从文件加载）
        self.audio_data = torch.randn(num_samples, audio_length)
        self.text_data = [f"Sample text {i}" for i in range(num_samples)]
        
        logger.info(f"Created dataset with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "audio": self.audio_data[idx],
            "text": self.text_data[idx],
            "sample_rate": self.sample_rate,
        }


# ============================================================================
# 2. 音频预处理模块
# ============================================================================

class AudioPreprocessor:
    """
    音频预处理器
    
    负责将原始音频波形转换为模型可用的特征
    """
    
    def __init__(self, config):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.audio_processor = AudioProcessor(config)
        
        logger.info("AudioPreprocessor initialized")
    
    def preprocess_batch(self, batch):
        """
        预处理一个 batch 的音频数据
        
        Args:
            batch: 包含 'audio' 键的字典
        
        Returns:
            处理后的特征字典
        """
        audio_waveforms = batch["audio"]
        
        # 确保是 [batch, time] 格式
        if audio_waveforms.dim() == 1:
            audio_waveforms = audio_waveforms.unsqueeze(0)
        
        # 提取 Mel 频谱特征
        mel_features = self.audio_processor.preprocess(audio_waveforms)
        
        logger.debug(f"Preprocessed audio: {audio_waveforms.shape} -> {mel_features.shape}")
        
        return {
            "mel_features": mel_features,
            "audio_waveforms": audio_waveforms,
        }


# ============================================================================
# 3. 音频模型
# ============================================================================

class AudioModel(nn.Module):
    """
    完整的音频处理模型
    
    包含编码器、解码器和可选的文本处理
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 音频头（编码器和解码器）
        self.audio_head = AudioHead(config)
        
        # 可选：文本编码器（简化版）
        self.text_encoder = self._build_text_encoder(config)
        
        # 融合层
        hidden_size = config.get("hidden_size", 512)
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, config.get("vocab_size", 1000))
        
        logger.info("AudioModel initialized")
    
    def _build_text_encoder(self, config):
        """构建简单的文本编码器"""
        hidden_size = config.get("hidden_size", 512)
        return nn.Sequential(
            nn.Embedding(1000, 256),  # 简化版
            nn.LSTM(256, hidden_size, batch_first=True),
        )
    
    def forward(self, audio_features, text_ids=None):
        """
        前向传播
        
        Args:
            audio_features: 音频特征 [batch, n_mels, time]
            text_ids: 文本 token IDs [batch, seq_len]（可选）
        
        Returns:
            模型输出
        """
        # 编码音频
        audio_encoded = self.audio_head(audio_features, mode="encode")
        
        # 编码文本（如果有）
        if text_ids is not None:
            text_encoded, _ = self.text_encoder(text_ids)
            text_encoded = text_encoded[:, -1, :]  # 取最后一个时间步
            
            # 融合音频和文本特征
            fused = torch.cat([audio_encoded, text_encoded], dim=-1)
            fused = self.fusion_layer(fused)
        else:
            fused = audio_encoded
        
        # 输出
        output = self.output_layer(fused)
        
        return {
            "output": output,
            "audio_features": audio_encoded,
            "fused_features": fused,
        }


# ============================================================================
# 4. 训练循环
# ============================================================================

class AudioTrainer:
    """
    音频模型训练器
    
    负责模型的训练、验证和评估
    """
    
    def __init__(self, model, config):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            config: 配置字典
        """
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 音频质量评估
        self.quality_model = AudioQualityModel(config)
        
        # 多模态 Reward 管理器
        self.reward_manager = MultimodalRewardManager(config)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"AudioTrainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader, epoch):
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 编号
        
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 预处理音频
            audio_waveforms = batch["audio"].to(self.device)
            
            # 提取 Mel 特征
            mel_features = self.model.audio_head.processor.preprocess(audio_waveforms)
            mel_features = mel_features.to(self.device)
            
            # 前向传播
            outputs = self.model(mel_features)
            
            # 计算损失（简化版：使用随机目标）
            batch_size = mel_features.size(0)
            targets = torch.randint(0, 1000, (batch_size,), device=self.device)
            loss = self.criterion(outputs["output"], targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate_audio_quality(self, audio_output, audio_target):
        """
        评估音频质量
        
        Args:
            audio_output: 生成的音频
            audio_target: 目标音频
        
        Returns:
            质量指标字典
        """
        metrics = self.quality_model.evaluate(audio_output, audio_target)
        
        logger.info(f"Audio quality metrics: {metrics}")
        
        return metrics
    
    def compute_multimodal_reward(self, outputs, targets):
        """
        计算多模态 Reward
        
        Args:
            outputs: 模型输出
            targets: 目标输出
        
        Returns:
            Reward 值
        """
        reward = self.reward_manager.compute_multimodal_reward(outputs, targets)
        
        logger.info(f"Multimodal reward: {reward:.4f}")
        
        return reward


# ============================================================================
# 5. 推理模块
# ============================================================================

class AudioInference:
    """
    音频推理模块
    
    负责使用训练好的模型进行推理
    """
    
    def __init__(self, model, config):
        """
        初始化推理模块
        
        Args:
            model: 训练好的模型
            config: 配置字典
        """
        self.model = model
        self.config = config
        self.audio_processor = AudioProcessor(config)
        
        # 设备
        self.device = next(model.parameters()).device
        
        logger.info("AudioInference initialized")
    
    @torch.no_grad()
    def infer_from_audio(self, audio_waveform):
        """
        从音频波形进行推理
        
        Args:
            audio_waveform: 音频波形 [time] 或 [batch, time]
        
        Returns:
            推理结果
        """
        self.model.eval()
        
        # 确保是 batch 格式
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)
        
        audio_waveform = audio_waveform.to(self.device)
        
        # 提取特征
        mel_features = self.audio_processor.preprocess(audio_waveform)
        mel_features = mel_features.to(self.device)
        
        # 推理
        outputs = self.model(mel_features)
        
        return {
            "output": outputs["output"],
            "audio_features": outputs["audio_features"],
        }
    
    @torch.no_grad()
    def generate_audio(self, features):
        """
        从特征生成音频
        
        Args:
            features: 音频特征 [batch, hidden_size]
        
        Returns:
            生成的音频波形
        """
        self.model.eval()
        
        # 解码音频
        audio_features = features.to(self.device)
        generated_audio = self.model.audio_head(audio_features, mode="decode")
        
        # 后处理
        audio_waveform = self.audio_processor.postprocess(generated_audio)
        
        return audio_waveform


# ============================================================================
# 6. 主函数
# ============================================================================

def main():
    """主函数：运行完整的音频支持样例"""
    
    logger.info("=" * 80)
    logger.info("音频支持完整样例")
    logger.info("=" * 80)
    
    # 配置
    config = {
        # 音频配置
        "sample_rate": 16000,
        "n_mels": 80,
        "n_fft": 1024,
        "hop_length": 256,
        "audio_length": 100,
        
        # 模型配置
        "hidden_size": 512,
        "num_layers": 3,
        "vocab_size": 1000,
        
        # 训练配置
        "learning_rate": 1e-4,
        "batch_size": 4,
        "num_epochs": 3,
        
        # Reward 配置
        "audio_weight": 0.5,
        "visual_weight": 0.0,
        "text_weight": 0.5,
    }
    
    # 1. 创建数据集
    logger.info("\n[步骤 1] 创建数据集...")
    dataset = AudioTextDataset(num_samples=100, sample_rate=config["sample_rate"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    logger.info(f"Dataset created: {len(dataset)} samples")
    
    # 2. 创建模型
    logger.info("\n[步骤 2] 创建音频模型...")
    model = AudioModel(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. 创建训练器
    logger.info("\n[步骤 3] 创建训练器...")
    trainer = AudioTrainer(model, config)
    
    # 4. 训练模型
    logger.info("\n[步骤 4] 开始训练...")
    for epoch in range(config["num_epochs"]):
        logger.info(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        avg_loss = trainer.train_epoch(dataloader, epoch + 1)
        logger.info(f"Epoch {epoch + 1} completed, Average Loss: {avg_loss:.4f}")
    
    # 5. 评估音频质量
    logger.info("\n[步骤 5] 评估音频质量...")
    sample_audio = torch.randn(2, 80, 100)
    target_audio = torch.randn(2, 80, 100)
    quality_metrics = trainer.evaluate_audio_quality(sample_audio, target_audio)
    
    # 6. 推理测试
    logger.info("\n[步骤 6] 推理测试...")
    inference = AudioInference(model, config)
    
    # 从音频推理
    test_audio = torch.randn(16000)  # 1秒音频
    inference_result = inference.infer_from_audio(test_audio)
    logger.info(f"Inference result shape: {inference_result['output'].shape}")
    
    # 生成音频
    features = torch.randn(2, config["hidden_size"])
    generated_audio = inference.generate_audio(features)
    logger.info(f"Generated audio shape: {generated_audio.shape}")
    
    # 7. 多模态 Reward 测试
    logger.info("\n[步骤 7] 多模态 Reward 测试...")
    outputs = {
        "text": "generated text",
        "audio": generated_audio,
    }
    targets = {
        "text": "target text",
        "audio": torch.randn_like(generated_audio),
    }
    reward = trainer.compute_multimodal_reward(outputs, targets)
    
    logger.info("\n" + "=" * 80)
    logger.info("音频支持样例完成！")
    logger.info("=" * 80)
    
    # 输出总结
    logger.info("\n样例总结：")
    logger.info(f"✓ 音频预处理：Mel 频谱提取")
    logger.info(f"✓ 音频编码：波形 -> 特征")
    logger.info(f"✓ 音频解码：特征 -> 波形")
    logger.info(f"✓ 模型训练：{config['num_epochs']} epochs")
    logger.info(f"✓ 音频质量评估：MCD、F0、频谱损失")
    logger.info(f"✓ 多模态 Reward：音频 + 文本")
    logger.info(f"✓ 推理测试：成功")


if __name__ == "__main__":
    main()
