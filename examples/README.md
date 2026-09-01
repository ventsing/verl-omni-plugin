# 样例说明文档

本文档详细介绍 verl-omni-plugin 提供的两个核心样例。

## 📋 样例列表

| 样例 | 文件 | 功能 | 复杂度 |
|------|------|------|--------|
| 音频支持 | `audio_support_example.py` | 音频处理、编码、解码、质量评估 | ⭐⭐⭐ |
| 全双工训练 | `full_duplex_example.py` | 并发训练推理、权重同步、双向数据流 | ⭐⭐⭐⭐⭐ |

---

## 🎵 样例 1：音频支持

### 文件位置
```
examples/audio_support_example.py
```

### 功能概述
展示如何使用插件进行完整的音频模型训练和推理流程。

### 核心功能
1. **音频预处理**
   - Mel 频谱特征提取
   - 音频归一化
   - 批量处理

2. **音频模型**
   - AudioHead（编码器 + 解码器）
   - AudioEncoder（卷积神经网络）
   - AudioDecoder（全连接网络）

3. **训练流程**
   - 数据加载和预处理
   - 前向传播和损失计算
   - 反向传播和参数更新
   - 训练监控

4. **质量评估**
   - MCD（梅尔倒谱失真）
   - F0 相关性
   - 频谱损失
   - 综合质量评分

5. **多模态 Reward**
   - 音频质量评分
   - 文本匹配评分
   - 加权融合

### 运行方式
```bash
cd verl-omni-plugin
python examples/audio_support_example.py
```

### 预期输出
```
================================================================================
音频支持完整样例
================================================================================

[步骤 1] 创建数据集...
Dataset created: 100 samples

[步骤 2] 创建音频模型...
Model created with XXXXX parameters

[步骤 3] 创建训练器...
AudioTrainer initialized on device: cuda

[步骤 4] 开始训练...

--- Epoch 1/3 ---
Epoch 1, Batch 10/25, Loss: X.XXXX
...
Epoch 1 completed, Average Loss: X.XXXX

[步骤 5] 评估音频质量...
Audio quality metrics: {'mcd': X.XX, 'f0_correlation': X.XX, ...}

[步骤 6] 推理测试...
Inference result shape: torch.Size([1, 1000])
Generated audio shape: torch.Size([2, 80, 100])

[步骤 7] 多模态 Reward 测试...
Multimodal reward: X.XXXX

================================================================================
音频支持样例完成！
================================================================================

样例总结：
✓ 音频预处理：Mel 频谱提取
✓ 音频编码：波形 -> 特征
✓ 音频解码：特征 -> 波形
✓ 模型训练：3 epochs
✓ 音频质量评估：MCD、F0、频谱损失
✓ 多模态 Reward：音频 + 文本
✓ 推理测试：成功
```

### 关键代码片段

#### 1. 音频预处理
```python
from shared.audio import AudioProcessor

# 创建处理器
processor = AudioProcessor({
    "sample_rate": 16000,
    "n_mels": 80,
})

# 提取 Mel 特征
mel_features = processor.preprocess(audio_waveform)
```

#### 2. 音频模型
```python
from plugins.verl_omni.models.audio import AudioHead

# 创建音频头
audio_head = AudioHead(config)

# 编码
features = audio_head(audio, mode="encode")

# 解码
reconstructed = audio_head(features, mode="decode")
```

#### 3. 质量评估
```python
from shared.audio import AudioQualityModel

quality_model = AudioQualityModel()
metrics = quality_model.evaluate(output_audio, target_audio)

# metrics 包含:
# - mcd: 梅尔倒谱失真（越小越好）
# - f0_correlation: 基频相关性（越大越好）
# - spectral_loss: 频谱损失（越小越好）
# - overall: 综合评分（0-1）
```

#### 4. 多模态 Reward
```python
from plugins.verl.reward import MultimodalRewardManager

reward_manager = MultimodalRewardManager({
    "audio_weight": 0.5,
    "text_weight": 0.5,
})

reward = reward_manager.compute_multimodal_reward(
    outputs={"audio": gen_audio, "text": gen_text},
    targets={"audio": tgt_audio, "text": tgt_text}
)
```

### 自定义扩展

#### 添加自定义音频数据集
```python
class MyAudioDataset(Dataset):
    def __init__(self, audio_dir, text_dir):
        # 从文件加载音频和文本
        self.audio_files = list(Path(audio_dir).glob("*.wav"))
        self.text_files = list(Path(text_dir).glob("*.txt"))
    
    def __getitem__(self, idx):
        audio = load_audio(self.audio_files[idx])
        text = load_text(self.text_files[idx])
        return {"audio": audio, "text": text}
```

#### 使用真实音频数据
```python
import torchaudio

# 加载音频
waveform, sample_rate = torchaudio.load("audio.wav")

# 重采样
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)

# 使用处理器
processor = AudioProcessor({"sample_rate": 16000})
features = processor.preprocess(waveform)
```

---

## 🔄 样例 2：全双工训练

### 文件位置
```
examples/full_duplex_example.py
```

### 功能概述
展示如何实现训练和推理的并发执行，实现全双工训练模式。

### 核心概念

**全双工训练** = 训练（Training）+ 推理（Inference）同时进行

```
┌─────────────────┐         ┌─────────────────┐
│   Training      │         │   Inference     │
│   Worker        │         │   Worker        │
│                 │  权重   │                 │
│  前向传播       │◄───────►│  生成文本       │
│  反向传播       │  同步   │  计算 Reward    │
│  参数更新       │         │                 │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │      ┌─────────────┐      │
         └─────►│   Queue     │◄─────┘
                │  数据流动   │
                └─────────────┘
```

### 核心功能
1. **并发执行**
   - 训练 Worker：处理训练 batch
   - 推理 Worker：生成文本
   - 权重同步 Worker：定期同步权重

2. **数据流动**
   - 训练数据 → 训练队列 → 训练 Worker
   - 推理提示 → 推理队列 → 推理 Worker
   - 推理结果 → 训练队列（用于 RLHF）

3. **权重同步**
   - 定期从训练同步到推理
   - 确保推理使用最新模型

4. **性能监控**
   - 训练吞吐量
   - 推理吞吐量
   - 延迟统计

### 运行方式
```bash
cd verl-omni-plugin
python examples/full_duplex_example.py
```

### 预期输出
```
================================================================================
全双工支持完整样例
================================================================================

[步骤 1] 创建语言模型...
Model created with XXXXX parameters

[步骤 2] 创建全双工训练器...
EnhancedFullDuplexTrainer initialized on device: cuda

[步骤 3] 创建数据生产者...
DataProducer initialized

[步骤 4] 创建结果收集器...
ResultCollector initialized

[步骤 5] 创建性能监控器...
PerformanceMonitor initialized: interval=5.0s

[步骤 6] 启动全双工训练...
Training and inference will run concurrently!

Running for 30.0 seconds...

2024-XX-XX 10:00:05 - INFO - Performance Report:
  Training: 50 steps, 10.00 steps/s
  Inference: 20 steps, 4.00 steps/s
  Avg training time: 100.00ms
  Avg inference time: 250.00ms

...

[步骤 7] 停止所有组件...

[步骤 8] 最终统计...

================================================================================
全双工训练完成！
================================================================================

运行时间: 30.00 秒

训练统计:
  - 总步数: 300
  - 吞吐量: 10.00 steps/s
  - 平均时间: 100.00 ms/step

推理统计:
  - 总步数: 120
  - 吞吐量: 4.00 steps/s
  - 平均时间: 250.00 ms/step

收集的结果: 120

================================================================================
全双工样例总结：
================================================================================
✓ 训练和推理并发执行
✓ 实时权重同步
✓ 双向数据流（推理结果反馈到训练）
✓ 性能监控和日志
✓ 异步任务管理

全双工训练可以显著提高 GPU 利用率和训练效率！
```

### 关键代码片段

#### 1. 创建全双工训练器
```python
from plugins.verl.trainer import FullDuplexTrainer

trainer = FullDuplexTrainer({
    "duplex_enabled": True,
    "weight_sync_interval": 2.0,  # 每 2 秒同步一次
})
```

#### 2. 自定义训练步骤
```python
class MyTrainer(FullDuplexTrainer):
    async def _train_step(self, batch):
        # 前向传播
        output = self.model(batch["input"])
        
        # 计算损失
        loss = self.criterion(output, batch["target"])
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

#### 3. 自定义推理步骤
```python
class MyTrainer(FullDuplexTrainer):
    async def _inference_step(self, prompt):
        # 生成文本
        generated = self.model.generate(
            prompt["input_ids"],
            max_length=20
        )
        
        # 计算 Reward（用于 RLHF）
        reward = self.reward_model(generated)
        
        return {
            "generated": generated,
            "reward": reward,
        }
```

#### 4. 数据生产者
```python
class DataProducer:
    async def produce_training_data(self):
        while True:
            batch = generate_batch()
            await trainer.training_queue.put(batch)
            await asyncio.sleep(0.001)
    
    async def produce_inference_prompts(self):
        while True:
            prompt = generate_prompt()
            await trainer.inference_queue.put(prompt)
            await asyncio.sleep(0.005)
```

#### 5. 启动全双工训练
```python
# 创建所有任务
tasks = [
    asyncio.create_task(producer.produce_training_data()),
    asyncio.create_task(producer.produce_inference_prompts()),
    asyncio.create_task(collector.collect_results()),
    asyncio.create_task(monitor.monitor()),
    asyncio.create_task(trainer.run_duplex_training()),
]

# 并发运行
await asyncio.gather(*tasks)
```

### 性能优化建议

#### 1. 调整队列大小
```python
trainer = FullDuplexTrainer({
    "training_queue_size": 100,   # 训练队列大小
    "inference_queue_size": 50,   # 推理队列大小
})
```

#### 2. 优化权重同步频率
```python
# 根据训练速度调整
trainer = FullDuplexTrainer({
    "weight_sync_interval": 1.0,  # 快速训练：1秒
    # "weight_sync_interval": 5.0,  # 慢速训练：5秒
})
```

#### 3. 使用多个推理 Worker
```python
# 可以创建多个推理 Worker 提高吞吐量
inference_workers = [
    InferenceWorker(model, config)
    for _ in range(num_workers)
]
```

### 实际应用场景

#### 场景 1：RLHF（人类反馈强化学习）
```
训练 Worker: 更新模型参数
    ↓
推理 Worker: 生成回复
    ↓
Reward Model: 评分
    ↓
反馈到训练 Worker: 使用 Reward 优化
```

#### 场景 2：在线学习
```
训练 Worker: 从历史数据学习
    ↓
推理 Worker: 处理实时请求
    ↓
收集用户反馈
    ↓
反馈到训练 Worker: 持续改进
```

#### 场景 3：数据增强
```
训练 Worker: 训练主模型
    ↓
推理 Worker: 生成合成数据
    ↓
质量过滤
    ↓
反馈到训练 Worker: 扩充训练集
```

---

## 🔧 常见问题

### Q1: 音频样例运行时 CUDA OOM？
**A**: 减小 batch_size 或 audio_length
```python
config = {
    "batch_size": 2,      # 从 4 减到 2
    "audio_length": 8000, # 从 16000 减到 8000
}
```

### Q2: 全双工样例中训练和推理速度不匹配？
**A**: 调整数据生产速度
```python
# 如果训练快于推理
await asyncio.sleep(0.005)  # 增加训练数据生产延迟

# 如果推理快于训练
await asyncio.sleep(0.001)  # 减少推理提示生产延迟
```

### Q3: 如何使用真实的音频数据？
**A**: 参考"自定义扩展"部分，使用 torchaudio 加载真实音频文件。

### Q4: 全双工训练如何保存 checkpoint？
**A**: 在训练步骤中添加保存逻辑
```python
async def _train_step(self, batch):
    # ... 训练逻辑 ...
    
    if self.training_steps % 1000 == 0:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.training_steps,
        }, f"checkpoint_{self.training_steps}.pt")
```

### Q5: 如何评估全双工训练的效果？
**A**: 比较以下指标
- GPU 利用率（应该更高）
- 训练吞吐量（steps/s）
- 推理吞吐量（steps/s）
- 总训练时间

---

## 📚 相关文档

- [快速开始指南](../QUICKSTART.md)
- [项目结构说明](../PROJECT_STRUCTURE.txt)
- [项目总结](../PROJECT_SUMMARY.md)
- [verl 插件文档](../plugins/verl/README.md)
- [verl-omni 插件文档](../plugins/verl_omni/README.md)

---

## 🎯 下一步

1. **运行样例**
   ```bash
   python examples/audio_support_example.py
   python examples/full_duplex_example.py
   ```

2. **修改参数**
   - 尝试不同的配置
   - 观察性能变化

3. **扩展功能**
   - 添加自定义数据集
   - 实现自定义模型
   - 集成真实音频数据

4. **性能优化**
   - 调整队列大小
   - 优化权重同步频率
   - 使用多个 Worker

5. **集成到项目**
   - 将样例代码集成到你的项目
   - 根据需求定制功能
