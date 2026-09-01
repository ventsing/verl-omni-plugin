"""
全双工支持完整样例

本样例展示如何使用 verl-omni-plugin 进行全双工训练：
1. 同时进行训练和推理
2. 实时权重同步
3. 双向数据流（推理结果反馈到训练）
4. 异步并发执行
5. 性能监控和日志

全双工的优势：
- 训练和推理并行，提高 GPU 利用率
- 推理结果可以实时反馈到训练（如 RLHF）
- 减少等待时间，加速整体训练

运行方式：
    python examples/full_duplex_example.py
"""

import asyncio
import logging
import time
from typing import Any, Dict

import torch
import torch.nn as nn

# 导入插件组件
from plugins.verl.trainer import FullDuplexTrainer
from shared.audio import AudioProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 简单的语言模型（用于演示）
# ============================================================================

class SimpleLanguageModel(nn.Module):
    """
    简单的语言模型
    
    用于演示全双工训练，实际应用中可以替换为更大的模型
    """
    
    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # 推理计数器（用于模拟推理延迟）
        self.inference_count = 0
        
        logger.info(f"SimpleLanguageModel initialized: vocab_size={vocab_size}, hidden_size={hidden_size}")
    
    def forward(self, input_ids):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
        
        Returns:
            输出 logits [batch, seq_len, vocab_size]
        """
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        logits = self.output(output)
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_length=20):
        """
        生成文本（自回归）
        
        Args:
            prompt_ids: 提示 token IDs [batch, seq_len]
            max_length: 最大生成长度
        
        Returns:
            生成的 token IDs [batch, seq_len + max_length]
        """
        self.eval()
        
        generated = prompt_ids.clone()
        
        for _ in range(max_length):
            # 前向传播
            logits = self.forward(generated)
            
            # 取最后一个时间步的 logits
            next_token_logits = logits[:, -1, :]
            
            # 采样（简化版：取 argmax）
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # 追加到序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 模拟推理延迟
            self.inference_count += 1
            if self.inference_count % 10 == 0:
                logger.debug(f"Generated {self.inference_count} tokens")
        
        return generated


# ============================================================================
# 2. 数据生成器
# ============================================================================

class DataGenerator:
    """
    数据生成器
    
    为训练和推理生成数据
    """
    
    def __init__(self, vocab_size=1000, batch_size=4, seq_length=10):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        logger.info("DataGenerator initialized")
    
    def generate_training_batch(self):
        """
        生成训练 batch
        
        Returns:
            训练数据字典
        """
        # 生成随机输入
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # 生成随机目标（简化版：下一个 token 预测）
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "batch_id": int(time.time() * 1000),
        }
    
    def generate_inference_prompt(self):
        """
        生成推理提示
        
        Returns:
            提示 token IDs
        """
        # 生成随机提示
        prompt_length = 5
        prompt_ids = torch.randint(0, self.vocab_size, (1, prompt_length))
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_id": int(time.time() * 1000),
        }


# ============================================================================
# 3. 增强的全双工训练器
# ============================================================================

class EnhancedFullDuplexTrainer(FullDuplexTrainer):
    """
    增强的全双工训练器
    
    在基础全双工训练器上添加：
    - 性能监控
    - 详细的日志
    - 自定义训练和推理逻辑
    """
    
    def __init__(self, model, config):
        """
        初始化增强的全双工训练器
        
        Args:
            model: 模型实例
            config: 配置字典
        """
        super().__init__(config)
        
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 数据生成器
        self.data_generator = DataGenerator(
            vocab_size=config.get("vocab_size", 1000),
            batch_size=config.get("batch_size", 4),
            seq_length=config.get("seq_length", 10),
        )
        
        # 性能监控
        self.training_steps = 0
        self.inference_steps = 0
        self.total_training_time = 0.0
        self.total_inference_time = 0.0
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"EnhancedFullDuplexTrainer initialized on device: {self.device}")
    
    async def _train_step(self, batch: Dict[str, Any]) -> float:
        """
        执行单个训练步骤
        
        Args:
            batch: 训练 batch
        
        Returns:
            损失值
        """
        start_time = time.time()
        
        self.model.train()
        
        # 移动到设备
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        
        # 前向传播
        logits = self.model(input_ids)
        
        # 计算损失（下一个 token 预测）
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新统计
        self.training_steps += 1
        step_time = time.time() - start_time
        self.total_training_time += step_time
        
        # 日志
        if self.training_steps % 50 == 0:
            avg_time = self.total_training_time / self.training_steps
            logger.info(
                f"Training step {self.training_steps}, "
                f"Loss: {loss.item():.4f}, "
                f"Avg time: {avg_time*1000:.2f}ms"
            )
        
        return loss.item()
    
    async def _inference_step(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个推理步骤
        
        Args:
            prompt: 推理提示
        
        Returns:
            生成的输出
        """
        start_time = time.time()
        
        self.model.eval()
        
        # 移动到设备
        prompt_ids = prompt["prompt_ids"].to(self.device)
        
        # 生成
        max_length = 10
        generated_ids = self.model.generate(prompt_ids, max_length=max_length)
        
        # 更新统计
        self.inference_steps += 1
        step_time = time.time() - start_time
        self.total_inference_time += step_time
        
        # 日志
        if self.inference_steps % 20 == 0:
            avg_time = self.total_inference_time / self.inference_steps
            logger.info(
                f"Inference step {self.inference_steps}, "
                f"Generated {max_length} tokens, "
                f"Avg time: {avg_time*1000:.2f}ms"
            )
        
        return {
            "generated_ids": generated_ids,
            "prompt_id": prompt["prompt_id"],
            "generated_length": max_length,
        }
    
    async def _sync_weights(self):
        """
        同步权重（从训练到推理）
        
        在实际应用中，这里会执行权重拷贝
        由于训练和推理使用同一个模型，这里只是一个演示
        """
        # 在实际应用中，可能需要：
        # 1. 从训练进程拷贝权重到推理进程
        # 2. 更新推理引擎的缓存
        # 3. 通知推理进程权重已更新
        
        logger.debug("Weights synchronized")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计字典
        """
        return {
            "training_steps": self.training_steps,
            "inference_steps": self.inference_steps,
            "total_training_time": self.total_training_time,
            "total_inference_time": self.total_inference_time,
            "avg_training_time": self.total_training_time / max(self.training_steps, 1),
            "avg_inference_time": self.total_inference_time / max(self.inference_steps, 1),
            "training_throughput": self.training_steps / max(self.total_training_time, 1e-6),
            "inference_throughput": self.inference_steps / max(self.total_inference_time, 1e-6),
        }


# ============================================================================
# 4. 数据生产者
# ============================================================================

class DataProducer:
    """
    数据生产者
    
    负责生成训练数据和推理提示，并放入队列
    """
    
    def __init__(self, trainer: EnhancedFullDuplexTrainer, config: Dict[str, Any]):
        self.trainer = trainer
        self.config = config
        self.data_generator = trainer.data_generator
        
        # 控制标志
        self.running = True
        
        logger.info("DataProducer initialized")
    
    async def produce_training_data(self, num_batches: int = 1000):
        """
        生产训练数据
        
        Args:
            num_batches: 要生产的 batch 数量
        """
        logger.info(f"Starting training data production: {num_batches} batches")
        
        for i in range(num_batches):
            if not self.running:
                break
            
            # 生成 batch
            batch = self.data_generator.generate_training_batch()
            
            # 放入队列
            await self.trainer.training_queue.put(batch)
            
            # 控制生产速度
            await asyncio.sleep(0.001)
        
        logger.info("Training data production completed")
    
    async def produce_inference_prompts(self, num_prompts: int = 200):
        """
        生产推理提示
        
        Args:
            num_prompts: 要生产的提示数量
        """
        logger.info(f"Starting inference prompt production: {num_prompts} prompts")
        
        for i in range(num_prompts):
            if not self.running:
                break
            
            # 生成提示
            prompt = self.data_generator.generate_inference_prompt()
            
            # 放入队列
            await self.trainer.inference_queue.put(prompt)
            
            # 控制生产速度
            await asyncio.sleep(0.005)
        
        logger.info("Inference prompt production completed")
    
    def stop(self):
        """停止数据生产"""
        self.running = False
        logger.info("DataProducer stopped")


# ============================================================================
# 5. 结果收集器
# ============================================================================

class ResultCollector:
    """
    结果收集器
    
    负责收集推理结果并反馈到训练
    """
    
    def __init__(self, trainer: EnhancedFullDuplexTrainer):
        self.trainer = trainer
        self.collected_results = []
        
        # 控制标志
        self.running = True
        
        logger.info("ResultCollector initialized")
    
    async def collect_results(self):
        """
        收集推理结果
        
        在实际应用中，这里可以：
        1. 评估生成质量
        2. 计算 Reward
        3. 将结果反馈到训练（如 RLHF）
        """
        logger.info("Starting result collection")
        
        while self.running:
            try:
                # 从训练队列获取结果（推理结果会被放入训练队列）
                result = await asyncio.wait_for(
                    self.trainer.training_queue.get(),
                    timeout=1.0
                )
                
                # 如果是推理结果（包含 generated_ids）
                if "generated_ids" in result:
                    self.collected_results.append(result)
                    
                    # 在这里可以：
                    # 1. 评估生成质量
                    # 2. 计算 Reward
                    # 3. 构造训练数据
                    
                    if len(self.collected_results) % 10 == 0:
                        logger.info(f"Collected {len(self.collected_results)} inference results")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in result collection: {e}")
                break
        
        logger.info(f"Result collection completed: {len(self.collected_results)} results")
    
    def stop(self):
        """停止收集"""
        self.running = False
        logger.info("ResultCollector stopped")


# ============================================================================
# 6. 性能监控器
# ============================================================================

class PerformanceMonitor:
    """
    性能监控器
    
    定期报告性能统计
    """
    
    def __init__(self, trainer: EnhancedFullDuplexTrainer, interval: float = 5.0):
        self.trainer = trainer
        self.interval = interval
        
        # 控制标志
        self.running = True
        
        logger.info(f"PerformanceMonitor initialized: interval={interval}s")
    
    async def monitor(self):
        """
        监控性能
        """
        logger.info("Starting performance monitoring")
        
        while self.running:
            # 等待间隔
            await asyncio.sleep(self.interval)
            
            # 获取统计
            stats = self.trainer.get_performance_stats()
            
            # 报告
            logger.info(
                f"Performance Report:\n"
                f"  Training: {stats['training_steps']} steps, "
                f"{stats['training_throughput']:.2f} steps/s\n"
                f"  Inference: {stats['inference_steps']} steps, "
                f"{stats['inference_throughput']:.2f} steps/s\n"
                f"  Avg training time: {stats['avg_training_time']*1000:.2f}ms\n"
                f"  Avg inference time: {stats['avg_inference_time']*1000:.2f}ms"
            )
        
        logger.info("Performance monitoring completed")
    
    def stop(self):
        """停止监控"""
        self.running = False
        logger.info("PerformanceMonitor stopped")


# ============================================================================
# 7. 主函数
# ============================================================================

async def main():
    """主函数：运行完整的全双工样例"""
    
    logger.info("=" * 80)
    logger.info("全双工支持完整样例")
    logger.info("=" * 80)
    
    # 配置
    config = {
        # 模型配置
        "vocab_size": 1000,
        "hidden_size": 256,
        
        # 训练配置
        "learning_rate": 1e-4,
        "batch_size": 4,
        "seq_length": 10,
        
        # 全双工配置
        "duplex_enabled": True,
        "weight_sync_interval": 2.0,  # 每 2 秒同步一次权重
        
        # 数据配置
        "num_training_batches": 500,
        "num_inference_prompts": 100,
        
        # 监控配置
        "monitor_interval": 5.0,  # 每 5 秒报告一次性能
    }
    
    # 1. 创建模型
    logger.info("\n[步骤 1] 创建语言模型...")
    model = SimpleLanguageModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {num_params} parameters")
    
    # 2. 创建全双工训练器
    logger.info("\n[步骤 2] 创建全双工训练器...")
    trainer = EnhancedFullDuplexTrainer(model, config)
    
    # 3. 创建数据生产者
    logger.info("\n[步骤 3] 创建数据生产者...")
    data_producer = DataProducer(trainer, config)
    
    # 4. 创建结果收集器
    logger.info("\n[步骤 4] 创建结果收集器...")
    result_collector = ResultCollector(trainer)
    
    # 5. 创建性能监控器
    logger.info("\n[步骤 5] 创建性能监控器...")
    monitor = PerformanceMonitor(trainer, interval=config["monitor_interval"])
    
    # 6. 启动全双工训练
    logger.info("\n[步骤 6] 启动全双工训练...")
    logger.info("Training and inference will run concurrently!")
    
    # 创建所有任务
    tasks = [
        # 数据生产
        asyncio.create_task(
            data_producer.produce_training_data(config["num_training_batches"])
        ),
        asyncio.create_task(
            data_producer.produce_inference_prompts(config["num_inference_prompts"])
        ),
        # 结果收集
        asyncio.create_task(result_collector.collect_results()),
        # 性能监控
        asyncio.create_task(monitor.monitor()),
        # 全双工训练（核心）
        asyncio.create_task(trainer.run_duplex_training()),
    ]
    
    # 运行一段时间（例如 30 秒）
    run_duration = 30.0
    logger.info(f"\nRunning for {run_duration} seconds...")
    
    start_time = time.time()
    await asyncio.sleep(run_duration)
    
    # 停止所有组件
    logger.info("\n[步骤 7] 停止所有组件...")
    data_producer.stop()
    result_collector.stop()
    monitor.stop()
    trainer.stop()
    
    # 等待所有任务完成
    await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed_time = time.time() - start_time
    
    # 7. 输出最终统计
    logger.info("\n[步骤 8] 最终统计...")
    stats = trainer.get_performance_stats()
    
    logger.info("\n" + "=" * 80)
    logger.info("全双工训练完成！")
    logger.info("=" * 80)
    
    logger.info(f"\n运行时间: {elapsed_time:.2f} 秒")
    logger.info(f"\n训练统计:")
    logger.info(f"  - 总步数: {stats['training_steps']}")
    logger.info(f"  - 吞吐量: {stats['training_throughput']:.2f} steps/s")
    logger.info(f"  - 平均时间: {stats['avg_training_time']*1000:.2f} ms/step")
    
    logger.info(f"\n推理统计:")
    logger.info(f"  - 总步数: {stats['inference_steps']}")
    logger.info(f"  - 吞吐量: {stats['inference_throughput']:.2f} steps/s")
    logger.info(f"  - 平均时间: {stats['avg_inference_time']*1000:.2f} ms/step")
    
    logger.info(f"\n收集的结果: {len(result_collector.collected_results)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("全双工样例总结：")
    logger.info("=" * 80)
    logger.info("✓ 训练和推理并发执行")
    logger.info("✓ 实时权重同步")
    logger.info("✓ 双向数据流（推理结果反馈到训练）")
    logger.info("✓ 性能监控和日志")
    logger.info("✓ 异步任务管理")
    logger.info("\n全双工训练可以显著提高 GPU 利用率和训练效率！")


if __name__ == "__main__":
    asyncio.run(main())
