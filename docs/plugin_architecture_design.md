# 插件化架构设计 - 按仓库分离的详细目录结构

## 总体架构

```
your-plugin-project/
├── pyproject.toml                          # 项目配置和 entry_points
├── README.md
├── setup.py
│
├── plugins/                                # 按上游仓库分离的插件目录
│   ├── verl/                               # verl 核心仓库插件
│   ├── verl_omni/                          # verl-omni 仓库插件
│   ├── vllm/                               # vllm 核心仓库插件
│   └── vllm_omni/                          # vllm-omni 仓库插件
│
├── shared/                                 # 跨仓库共享的工具和基础设施
│   ├── patch_manager/                      # 统一的 patch 管理器
│   ├── registry_utils/                     # 注册工具
│   └── config/                             # 配置管理
│
├── configs/                                # 配置文件
├── examples/                               # 使用示例
└── tests/                                  # 测试套件
```

---

## 1. verl 插件目录 (plugins/verl/)

针对 verl 核心仓库的扩展,主要涉及训练框架、平台支持、分布式通信等。

### 1.1 目录结构

```
plugins/verl/
├── __init__.py                             # verl 插件入口
├── README.md                               # verl 插件说明
│
├── platform/                               # 硬件平台扩展
│   ├── __init__.py
│   ├── custom_platform.py                  # 自定义平台实现
│   └── platform_hooks.py                   # 平台相关 Hook
│
├── trainer/                                # 训练器扩展
│   ├── __init__.py
│   ├── async_trainer_enhanced.py           # 增强的异步训练器
│   ├── full_duplex_trainer.py              # 全双工训练器
│   └── trainer_hooks.py                    # 训练器 Hook
│
├── workers/                                # Worker 扩展
│   ├── __init__.py
│   ├── engine_workers_enhanced.py          # 增强的引擎 Worker
│   ├── rollout_workers_enhanced.py         # 增强的 Rollout Worker
│   └── worker_hooks.py                     # Worker Hook
│
├── distributed/                            # 分布式通信扩展
│   ├── __init__.py
│   ├── custom_communicators.py             # 自定义通信原语
│   └── communication_hooks.py              # 通信 Hook
│
├── data/                                   # 数据处理扩展
│   ├── __init__.py
│   ├── data_processor_enhanced.py          # 增强的数据处理器
│   └── data_hooks.py                       # 数据处理 Hook
│
├── reward/                                 # Reward 框架扩展
│   ├── __init__.py
│   ├── reward_manager_enhanced.py          # 增强的 Reward Manager
│   └── reward_hooks.py                     # Reward Hook
│
└── patches/                                # Monkey-patch 管理
    ├── __init__.py
    ├── manager.py                          # Patch 管理器
    ├── trainer_patches.py                  # 训练器 patches
    ├── worker_patches.py                   # Worker patches
    ├── distributed_patches.py              # 分布式 patches
    └── data_patches.py                     # 数据处理 patches
```

### 1.2 功能承载点详解

#### 1.2.1 平台扩展 (platform/)

**承载功能**:
- 自定义硬件平台支持
- 平台特定的优化策略
- 设备管理和内存分配

**修改点对应**:
```python
# plugins/verl/platform/custom_platform.py
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="custom")
class CustomPlatform(PlatformBase):
    """自定义平台实现"""
    
    @property
    def device_name(self) -> str:
        return "custom_device"
    
    @property
    def vendor_name(self) -> str:
        return "custom_vendor"
    
    def apply_model_patches(self, model_type: str) -> None:
        """应用模型特定的 patches"""
        # 根据模型类型应用不同的 patches
        if model_type == "audio_model":
            self._apply_audio_patches()
        elif model_type == "omni_model":
            self._apply_omni_patches()
    
    def _apply_audio_patches(self):
        """应用音频模型 patches"""
        # 修改点:verl/plugin/platform/platform_base.py:203
        # 原方法:apply_model_patches()
        # 扩展:添加音频特定的优化
        pass
```

#### 1.2.2 训练器扩展 (trainer/)

**承载功能**:
- 异步训练优化
- 全双工训练支持
- 训练循环自定义

**修改点对应**:
```python
# plugins/verl/trainer/async_trainer_enhanced.py
"""增强的异步训练器,支持更灵活的异步策略"""

# 修改点:verl/experimental/fully_async_policy/fully_async_trainer.py
# 原类:FullyAsyncTrainer
# 扩展点:
# 1. 训练和推理的异步调度策略
# 2. 资源动态分配
# 3. 梯度同步优化

class EnhancedFullyAsyncTrainer:
    """增强的全异步训练器"""
    
    def __init__(self, config):
        self.config = config
        self.training_pipeline = self._build_training_pipeline()
        self.inference_pipeline = self._build_inference_pipeline()
    
    async def train_and_infer_async(self):
        """同时运行训练和推理,支持全双工"""
        # 修改点:verl/experimental/fully_async_policy/fully_async_trainer.py:train_step()
        # 原方法:串行的训练步骤
        # 扩展:并行化训练和推理
        
        await asyncio.gather(
            self._training_loop(),
            self._inference_loop(),
            return_exceptions=True
        )
    
    async def _training_loop(self):
        """训练循环,支持动态 batch size"""
        while not self.stop_event.is_set():
            batch = await self.get_training_batch()
            loss = await self.train_step(batch)
            await self.update_weights()
    
    async def _inference_loop(self):
        """推理循环,支持流式输出"""
        while not self.stop_event.is_set():
            prompt = await self.get_inference_prompt()
            output = await self.inference_step(prompt)
            await self.feedback_to_training(output)


# plugins/verl/trainer/full_duplex_trainer.py
"""全双工训练器,支持边训练边推理"""

# 修改点:verl/trainer/ppo/v1/trainer_base.py
# 原类:BaseTrainer
# 扩展点:
# 1. 双向数据流
# 2. 实时权重同步
# 3. 动态资源分配

class FullDuplexTrainer(BaseTrainer):
    """全双工训练器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.duplex_mode = True
        self.weight_sync_interval = config.get('weight_sync_interval', 10)
    
    async def run_duplex_training(self):
        """运行全双工训练"""
        # 修改点:verl/trainer/ppo/v1/trainer_base.py:fit()
        # 原方法:同步的训练循环
        # 扩展:异步的双向训练
        
        tasks = [
            asyncio.create_task(self.training_worker()),
            asyncio.create_task(self.inference_worker()),
            asyncio.create_task(self.weight_sync_worker()),
        ]
        
        await asyncio.gather(*tasks)
    
    async def weight_sync_worker(self):
        """权重同步 worker,定期同步训练权重到推理引擎"""
        while True:
            await asyncio.sleep(self.weight_sync_interval)
            await self.sync_weights_to_inference()
```

#### 1.2.3 Worker 扩展 (workers/)

**承载功能**:
- 引擎 Worker 增强
- Rollout Worker 优化
- Worker 间通信优化

**修改点对应**:
```python
# plugins/verl/workers/engine_workers_enhanced.py
"""增强的引擎 Worker,支持多模态和音频处理"""

# 修改点:verl/workers/engine_workers.py
# 原类:EngineWorkerGroup
# 扩展点:
# 1. 多模态输入处理
# 2. 音频特征提取
# 3. 动态 batch 管理

class EnhancedEngineWorkerGroup(EngineWorkerGroup):
    """增强的引擎 Worker 组"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_processor = self._init_audio_processor()
    
    def _init_audio_processor(self):
        """初始化音频处理器"""
        # 修改点:verl/workers/engine_workers.py:__init__()
        # 原方法:只处理文本输入
        # 扩展:添加音频处理能力
        from shared.audio import AudioProcessor
        return AudioProcessor(self.config)
    
    async def process_multimodal_batch(self, batch):
        """处理多模态 batch,包括音频"""
        # 修改点:verl/workers/engine_workers.py:compute_log_prob()
        # 原方法:只处理文本
        # 扩展:处理文本+音频+图像
        
        text_features = self.process_text(batch['text'])
        audio_features = self.audio_processor.process(batch['audio'])
        image_features = self.process_image(batch.get('image'))
        
        # 融合多模态特征
        fused_features = self.fuse_modalities(
            text_features, audio_features, image_features
        )
        
        return await self.compute_log_prob(fused_features)
```

#### 1.2.4 分布式通信扩展 (distributed/)

**承载功能**:
- 自定义通信原语
- 通信优化策略
- 跨平台通信支持

**修改点对应**:
```python
# plugins/verl/distributed/custom_communicators.py
"""自定义通信原语,优化多模态训练的通信"""

# 修改点:verl/distributed/parallel_state.py
# 原函数:all_reduce(), broadcast() 等
# 扩展点:
# 1. 分模态通信
# 2. 异步通信
# 3. 通信压缩

class MultimodalCommunicator:
    """多模态通信器,支持分模态通信"""
    
    def __init__(self, group):
        self.group = group
        self.compression_enabled = True
    
    async def all_reduce_multimodal(self, modal_tensors):
        """对多模态张量进行 all-reduce"""
        # 修改点:verl/distributed/parallel_state.py:all_reduce()
        # 原方法:统一的 all-reduce
        # 扩展:分模态通信,支持不同压缩策略
        
        results = {}
        for modal_name, tensor in modal_tensors.items():
            # 根据模态类型选择压缩策略
            if modal_name == 'audio':
                compressed = self._compress_audio(tensor)
            elif modal_name == 'text':
                compressed = self._compress_text(tensor)
            else:
                compressed = tensor
            
            # 异步 all-reduce
            results[modal_name] = await self._async_all_reduce(compressed)
        
        return results
    
    async def _async_all_reduce(self, tensor):
        """异步 all-reduce 实现"""
        # 修改点:verl/distributed/parallel_state.py:all_reduce()
        # 原方法:同步 all-reduce
        # 扩展:异步通信,减少阻塞
        return await asyncio.to_thread(
            torch.distributed.all_reduce, tensor, group=self.group
        )
```

#### 1.2.5 数据处理扩展 (data/)

**承载功能**:
- 多模态数据处理
- 音频数据处理
- 数据增强策略

**修改点对应**:
```python
# plugins/verl/data/data_processor_enhanced.py
"""增强的数据处理器,支持多模态和音频"""

# 修改点:verl/protocol.py (DataProto 相关处理)
# 原类:DataProto
# 扩展点:
# 1. 多模态数据字段
# 2. 音频特征处理
# 3. 数据验证和清洗

class EnhancedDataProcessor:
    """增强的数据处理器"""
    
    def __init__(self, config):
        self.config = config
        self.audio_feature_extractor = self._init_audio_extractor()
    
    def _init_audio_extractor(self):
        """初始化音频特征提取器"""
        # 修改点:verl/protocol.py:DataProto
        # 原方法:只处理文本数据
        # 扩展:添加音频特征提取
        from shared.audio import AudioFeatureExtractor
        return AudioFeatureExtractor(config)
    
    def process_batch(self, batch):
        """处理 batch,包括多模态数据"""
        # 修改点:verl/protocol.py:DataProto.from_single_dict()
        # 原方法:只处理文本字段
        # 扩展:处理多模态字段
        
        processed = {}
        
        # 处理文本
        if 'text' in batch:
            processed['text'] = self._process_text(batch['text'])
        
        # 处理音频
        if 'audio' in batch:
            processed['audio_features'] = self.audio_feature_extractor.extract(
                batch['audio']
            )
        
        # 处理图像
        if 'image' in batch:
            processed['image_features'] = self._process_image(batch['image'])
        
        # 创建增强的 DataProto
        return EnhancedDataProto.from_multimodal_dict(processed)
```

#### 1.2.6 Reward 框架扩展 (reward/)

**承载功能**:
- 多模态 Reward 计算
- 音频质量评估
- 自定义 Reward 策略

**修改点对应**:
```python
# plugins/verl/reward/reward_manager_enhanced.py
"""增强的 Reward Manager,支持多模态 Reward"""

# 修改点:verl/experimental/reward_loop/reward_manager/base.py
# 原类:BaseRewardManager
# 扩展点:
# 1. 多模态 Reward 融合
# 2. 音频质量评分
# 3. 动态 Reward 权重

class EnhancedRewardManager(BaseRewardManager):
    """增强的 Reward Manager"""
    
    def __init__(self, config):
        super().__init__(config)
        self.audio_reward_weight = config.get('audio_reward_weight', 0.3)
        self.visual_reward_weight = config.get('visual_reward_weight', 0.7)
    
    def compute_multimodal_reward(self, outputs, targets):
        """计算多模态 Reward"""
        # 修改点:verl/experimental/reward_loop/reward_manager/base.py:compute_reward()
        # 原方法:只计算文本 Reward
        # 扩展:计算多模态 Reward
        
        rewards = {}
        
        # 文本 Reward
        if 'text' in outputs:
            rewards['text'] = self._compute_text_reward(
                outputs['text'], targets.get('text')
            )
        
        # 音频 Reward
        if 'audio' in outputs:
            rewards['audio'] = self._compute_audio_reward(
                outputs['audio'], targets.get('audio')
            )
        
        # 图像 Reward
        if 'image' in outputs:
            rewards['image'] = self._compute_image_reward(
                outputs['image'], targets.get('image')
            )
        
        # 融合多模态 Reward
        final_reward = self._fuse_rewards(rewards)
        return final_reward
    
    def _compute_audio_reward(self, audio_output, audio_target):
        """计算音频质量 Reward"""
        # 修改点:新增方法
        # 功能:评估音频质量(清晰度、自然度等)
        
        # 使用预训练的音频质量评估模型
        audio_quality_score = self.audio_quality_model.evaluate(
            audio_output, audio_target
        )
        
        return audio_quality_score
```

#### 1.2.7 Monkey-patch 管理 (patches/)

**承载功能**:
- 集中管理所有 patches
- 版本兼容性检查
- 动态启用/禁用 patches

**修改点对应**:
```python
# plugins/verl/patches/manager.py
"""verl 仓库的 Patch 管理器"""

from shared.patch_manager import BasePatchManager


class VerlPatchManager(BasePatchManager):
    """verl 专用的 Patch 管理器"""
    
    @classmethod
    def register_all_patches(cls):
        """注册所有 verl patches"""
        cls._register_trainer_patches()
        cls._register_worker_patches()
        cls._register_distributed_patches()
        cls._register_data_patches()
    
    @classmethod
    def _register_trainer_patches(cls):
        """注册训练器相关 patches"""
        # Patch 1: 增强异步训练器
        cls.register_patch(
            name="enhanced_async_trainer",
            target_module="verl.experimental.fully_async_policy.fully_async_trainer",
            target_attr="FullyAsyncTrainer",
            replacement_fn="plugins.verl.trainer.async_trainer_enhanced:EnhancedFullyAsyncTrainer",
            version_check=lambda: cls._check_verl_version(">=0.6.0"),
            description="增强异步训练器,支持全双工"
        )
        
        # Patch 2: 全双工训练器
        cls.register_patch(
            name="full_duplex_trainer",
            target_module="verl.trainer.ppo.v1.trainer_base",
            target_attr="BaseTrainer",
            replacement_fn="plugins.verl.trainer.full_duplex_trainer:FullDuplexTrainer",
            version_check=lambda: cls._check_verl_version(">=0.6.0"),
            description="全双工训练器"
        )
    
    @classmethod
    def _register_worker_patches(cls):
        """注册 Worker 相关 patches"""
        # Patch: 增强引擎 Worker
        cls.register_patch(
            name="enhanced_engine_worker",
            target_module="verl.workers.engine_workers",
            target_attr="EngineWorkerGroup",
            replacement_fn="plugins.verl.workers.engine_workers_enhanced:EnhancedEngineWorkerGroup",
            version_check=lambda: cls._check_verl_version(">=0.6.0"),
            description="增强引擎 Worker,支持多模态"
        )
    
    @classmethod
    def _register_distributed_patches(cls):
        """注册分布式通信 patches"""
        # Patch: 多模态通信器
        cls.register_patch(
            name="multimodal_communicator",
            target_module="verl.distributed.parallel_state",
            target_attr="all_reduce",
            replacement_fn="plugins.verl.distributed.custom_communicators:multimodal_all_reduce",
            version_check=lambda: cls._check_verl_version(">=0.6.0"),
            description="多模态通信优化"
        )
```

---

## 2. verl-omni 插件目录 (plugins/verl_omni/)

针对 verl-omni 仓库的扩展,主要涉及多模态模型、扩散模型、全模态训练等。

### 2.1 目录结构

```
plugins/verl_omni/
├── __init__.py                             # verl-omni 插件入口
├── README.md                               # verl-omni 插件说明
│
├── models/                                 # 多模态模型扩展
│   ├── __init__.py
│   ├── audio/                              # 音频模型
│   │   ├── __init__.py
│   │   ├── audio_head.py                   # 音频头处理
│   │   ├── audio_encoder.py                # 音频编码器
│   │   └── audio_decoder.py                # 音频解码器
│   ├── omni/                               # 全模态模型
│   │   ├── __init__.py
│   │   ├── custom_omni_model.py            # 自定义全模态模型
│   │   └── omni_adapter.py                 # 全模态适配器
│   └── diffusion/                          # 扩散模型
│       ├── __init__.py
│       ├── custom_diffusion.py             # 自定义扩散模型
│       └── diffusion_adapter.py            # 扩散模型适配器
│
├── pipelines/                              # 训练流水线扩展
│   ├── __init__.py
│   ├── audio_flow_grpo/                    # 音频 Flow-GRPO 流水线
│   │   ├── __init__.py
│   │   ├── diffusers_training_adapter.py   # Diffusers 训练适配器
│   │   ├── vllm_omni_rollout_adapter.py    # vLLM-Omni Rollout 适配器
│   │   └── common.py                       # 公共组件
│   ├── omni_full_duplex/                   # 全双工全模态流水线
│   │   ├── __init__.py
│   │   ├── omni_training_adapter.py        # 全模态训练适配器
│   │   └── duplex_rollout_adapter.py       # 双工 Rollout 适配器
│   └── pipeline_hooks.py                   # 流水线 Hook
│
├── reward_loop/                            # Reward 循环扩展
│   ├── __init__.py
│   ├── reward_manager/                     # Reward Manager
│   │   ├── __init__.py
│   │   ├── audio_reward_manager.py         # 音频 Reward Manager
│   │   ├── multimodal_reward_manager.py    # 多模态 Reward Manager
│   │   └── quality_reward_manager.py       # 质量评估 Reward Manager
│   └── reward_hooks.py                     # Reward Hook
│
├── trainer/                                # 训练器扩展
│   ├── __init__.py
│   ├── diffusion/                          # 扩散模型训练器
│   │   ├── __init__.py
│   │   ├── custom_diffusion_trainer.py     # 自定义扩散训练器
│   │   └── diffusion_algos_enhanced.py     # 增强的扩散算法
│   ├── omni/                               # 全模态训练器
│   │   ├── __init__.py
│   │   ├── omni_trainer_enhanced.py        # 增强的全模态训练器
│   │   └── omni_algos_enhanced.py          # 增强的全模态算法
│   └── trainer_hooks.py                    # 训练器 Hook
│
├── agent_loop/                             # Agent Loop 扩展
│   ├── __init__.py
│   ├── audio_agent_loop.py                 # 音频 Agent Loop
│   ├── multimodal_agent_loop.py            # 多模态 Agent Loop
│   └── agent_hooks.py                      # Agent Hook
│
├── workers/                                # Worker 扩展
│   ├── __init__.py
│   ├── rollout/                            # Rollout Worker
│   │   ├── __init__.py
│   │   ├── vllm_rollout_enhanced.py        # 增强的 vLLM Rollout
│   │   └── rollout_hooks.py                # Rollout Hook
│   └── engine/                             # 引擎 Worker
│       ├── __init__.py
│       ├── fsdp_enhanced.py                # 增强的 FSDP 引擎
│       └── engine_hooks.py                 # 引擎 Hook
│
└── patches/                                # Monkey-patch 管理
    ├── __init__.py
    ├── manager.py                          # Patch 管理器
    ├── model_patches.py                    # 模型 patches
    ├── pipeline_patches.py                 # 流水线 patches
    ├── reward_patches.py                   # Reward patches
    └── trainer_patches.py                  # 训练器 patches
```

### 2.2 功能承载点详解

#### 2.2.1 音频模型扩展 (models/audio/)

**承载功能**:
- 音频头处理
- 音频编码/解码
- 音频特征提取

**修改点对应**:
```python
# plugins/verl_omni/models/audio/audio_head.py
"""音频头处理模块"""

import torch
import torch.nn as nn
from verl_omni.pipelines.model_base import OmniModelBase


class AudioHead(nn.Module):
    """音频处理头,负责音频特征的编码和解码"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
    
    def forward(self, audio_input, mode='encode'):
        """前向传播"""
        # 修改点:verl_omni/models/transformers/qwen3_omni_thinker.py
        # 原类:Qwen3OmniThinker
        # 扩展点:添加音频处理头
        
        if mode == 'encode':
            return self.encoder(audio_input)
        else:
            return self.decoder(audio_input)


class AudioEncoder(nn.Module):
    """音频编码器,将音频波形转换为特征"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 修改点:verl_omni/models/transformers/qwen3_omni_thinker.py
        # 原方法:只处理文本和图像
        # 扩展:添加音频编码
        
        # 音频特征提取网络
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(80, 256, 3, padding=1),  # 80 维 Mel 频谱
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, config.hidden_size, 3, padding=1),
        ])
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, audio_features):
        """编码音频特征"""
        # audio_features: [batch, 80, time_steps]
        
        x = audio_features
        for conv in self.conv_layers:
            x = conv(x)
        
        # 池化到固定长度
        x = self.pooling(x).squeeze(-1)  # [batch, hidden_size]
        
        return x


class AudioDecoder(nn.Module):
    """音频解码器,将特征转换回音频波形"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 修改点:新增模块
        # 功能:从特征生成音频
        
        self.decoder_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 80 * config.audio_length),  # 输出 Mel 频谱
        ])
    
    def forward(self, features):
        """解码音频特征"""
        # features: [batch, hidden_size]
        
        x = features
        for layer in self.decoder_layers[:-1]:
            x = layer(x)
        
        # 输出 Mel 频谱
        mel_spectrogram = self.decoder_layers[-1](x)
        mel_spectrogram = mel_spectrogram.view(-1, 80, self.config.audio_length)
        
        return mel_spectrogram


@OmniModelBase.register("CustomAudioOmniModel", stage="thinker")
class CustomAudioOmniModelAdapter(OmniModelBase):
    """注册自定义音频全模态模型适配器"""
    
    # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
    # 原方法:只处理文本和图像
    # 扩展:添加音频处理
    
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.audio_head = AudioHead(model_config)
    
    def process_audio_input(self, audio_tensor):
        """处理音频输入"""
        # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
        # 原方法:无音频处理
        # 扩展:添加音频输入处理
        
        audio_features = self.audio_head(audio_tensor, mode='encode')
        return audio_features
    
    def generate_audio(self, features, target_length=None):
        """生成音频输出"""
        # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
        # 原方法:无音频生成
        # 扩展:添加音频输出生成
        
        mel_spectrogram = self.audio_head(features, mode='decode')
        
        # 可选:使用 vocoder 将 Mel 频谱转换为波形
        if hasattr(self, 'vocoder'):
            audio_waveform = self.vocoder(mel_spectrogram)
            return audio_waveform
        
        return mel_spectrogram
```

#### 2.2.2 全模态模型扩展 (models/omni/)

**承载功能**:
- 自定义全模态模型
- 模态融合策略
- 全模态适配器

**修改点对应**:
```python
# plugins/verl_omni/models/omni/custom_omni_model.py
"""自定义全模态模型,支持文本、图像、音频"""

import torch
import torch.nn as nn
from verl_omni.pipelines.model_base import OmniModelBase


@OmniModelBase.register("CustomOmniModel", stage="thinker")
class CustomOmniModelAdapter(OmniModelBase):
    """自定义全模态模型适配器"""
    
    # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
    # 原类:OmniModelBase
    # 扩展点:支持更多模态和自定义融合策略
    
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        
        # 模态编码器
        self.text_encoder = self._build_text_encoder(model_config)
        self.image_encoder = self._build_image_encoder(model_config)
        self.audio_encoder = self._build_audio_encoder(model_config)
        
        # 模态融合器
        self.modal_fusion = self._build_modal_fusion(model_config)
    
    def _build_modal_fusion(self, config):
        """构建模态融合器"""
        # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
        # 原方法:简单的特征拼接
        # 扩展:支持注意力融合、门控融合等
        
        fusion_type = config.get('fusion_type', 'attention')
        
        if fusion_type == 'attention':
            return AttentionFusion(config)
        elif fusion_type == 'gating':
            return GatingFusion(config)
        else:
            return ConcatFusion(config)
    
    def forward(self, text=None, image=None, audio=None):
        """前向传播,处理多模态输入"""
        # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase:forward()
        # 原方法:只处理文本和图像
        # 扩展:处理文本、图像、音频
        
        modal_embeddings = {}
        
        # 编码各模态
        if text is not None:
            modal_embeddings['text'] = self.text_encoder(text)
        
        if image is not None:
            modal_embeddings['image'] = self.image_encoder(image)
        
        if audio is not None:
            modal_embeddings['audio'] = self.audio_encoder(audio)
        
        # 融合多模态特征
        fused_features = self.modal_fusion(modal_embeddings)
        
        # 通过主模型
        output = self.main_model(fused_features)
        
        return output


class AttentionFusion(nn.Module):
    """基于注意力的模态融合"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
    
    def forward(self, modal_embeddings):
        """融合多模态特征"""
        # 将各模态特征作为序列
        modal_list = list(modal_embeddings.values())
        
        # 使用第一个模态作为 query,其他作为 key/value
        query = modal_list[0]
        key_values = torch.cat(modal_list[1:], dim=1)
        
        # 跨模态注意力
        fused, _ = self.cross_attention(query, key_values, key_values)
        
        return fused
```

#### 2.2.3 训练流水线扩展 (pipelines/)

**承载功能**:
- 音频 Flow-GRPO 流水线
- 全双工全模态流水线
- 自定义训练策略

**修改点对应**:
```python
# plugins/verl_omni/pipelines/audio_flow_grpo/diffusers_training_adapter.py
"""音频 Flow-GRPO 训练适配器"""

from verl_omni.pipelines.model_base import DiffusionModelBase


@DiffusionModelBase.register("AudioDiffusionPipeline", algorithm="flow_grpo")
class AudioFlowGRPOAdapter(DiffusionModelBase):
    """音频 Flow-GRPO 训练适配器"""
    
    # 修改点:verl_omni/pipelines/model_base.py:DiffusionModelBase
    # 原方法:只处理图像和视频扩散
    # 扩展:添加音频扩散支持
    
    def __init__(self, pipeline, config):
        super().__init__(pipeline, config)
        self.audio_processor = self._init_audio_processor()
    
    def _init_audio_processor(self):
        """初始化音频处理器"""
        # 修改点:verl_omni/pipelines/model_base.py:DiffusionModelBase
        # 原方法:无音频处理
        # 扩展:添加音频预处理和后处理
        
        from plugins.verl_omni.models.audio import AudioProcessor
        return AudioProcessor(self.config)
    
    def prepare_inputs(self, batch):
        """准备输入,包括音频数据"""
        # 修改点:verl_omni/pipelines/model_base.py:DiffusionModelBase:prepare_inputs()
        # 原方法:只处理图像
        # 扩展:处理音频
        
        inputs = super().prepare_inputs(batch)
        
        # 添加音频处理
        if 'audio' in batch:
            inputs['audio_features'] = self.audio_processor.extract_features(
                batch['audio']
            )
        
        return inputs
    
    def compute_loss(self, model_output, target, noise=None):
        """计算音频扩散损失"""
        # 修改点:verl_omni/pipelines/model_base.py:DiffusionModelBase:compute_loss()
        # 原方法:图像扩散损失
        # 扩展:音频扩散损失
        
        # 基础扩散损失
        diffusion_loss = super().compute_loss(model_output, target, noise)
        
        # 音频特定的损失(如频谱损失)
        if hasattr(self, 'audio_processor'):
            audio_loss = self.audio_processor.compute_spectral_loss(
                model_output, target
            )
            diffusion_loss = diffusion_loss + 0.1 * audio_loss
        
        return diffusion_loss


# plugins/verl_omni/pipelines/omni_full_duplex/omni_training_adapter.py
"""全双工全模态训练适配器"""

from verl_omni.pipelines.model_base import OmniModelBase


@OmniModelBase.register("FullDuplexOmniModel", stage="thinker")
class FullDuplexOmniAdapter(OmniModelBase):
    """全双工全模态适配器,支持边训练边推理"""
    
    # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
    # 原方法:同步训练
    # 扩展:异步全双工训练
    
    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.duplex_enabled = True
        self.inference_queue = asyncio.Queue()
        self.training_queue = asyncio.Queue()
    
    async def train_and_infer_duplex(self):
        """全双工训练和推理"""
        # 修改点:verl_omni/pipelines/model_base.py:OmniModelBase
        # 原方法:同步训练
        # 扩展:异步全双工
        
        await asyncio.gather(
            self._training_loop(),
            self._inference_loop(),
            self._weight_sync_loop(),
        )
    
    async def _training_loop(self):
        """训练循环"""
        while True:
            batch = await self.training_queue.get()
            loss = await self.train_step(batch)
            await self.update_weights()
    
    async def _inference_loop(self):
        """推理循环"""
        while True:
            prompt = await self.inference_queue.get()
            output = await self.inference_step(prompt)
            # 将推理结果反馈到训练
            await self.training_queue.put(output)
    
    async def _weight_sync_loop(self):
        """权重同步循环"""
        while True:
            await asyncio.sleep(10)  # 每 10 步同步一次
            await self.sync_weights()
```

#### 2.2.4 Reward 循环扩展 (reward_loop/)

**承载功能**:
- 音频质量评估
- 多模态 Reward 融合
- 自定义 Reward 策略

**修改点对应**:
```python
# plugins/verl_omni/reward_loop/reward_manager/audio_reward_manager.py
"""音频 Reward Manager,评估音频生成质量"""

from verl_omni.reward_loop.reward_manager import VisualRewardManager


class AudioRewardManager(VisualRewardManager):
    """音频 Reward Manager"""
    
    # 修改点:verl_omni/reward_loop/reward_manager/multi.py:MultiVisualRewardManager
    # 原类:MultiVisualRewardManager
    # 扩展:添加音频质量评估
    
    def __init__(self, config):
        super().__init__(config)
        self.audio_weight = config.get('audio_weight', 0.3)
        self.audio_quality_model = self._load_audio_quality_model()
    
    def _load_audio_quality_model(self):
        """加载音频质量评估模型"""
        # 修改点:verl_omni/reward_loop/reward_manager/multi.py
        # 原方法:只加载视觉质量模型
        # 扩展:加载音频质量模型
        
        # 可以使用预训练的音频质量评估模型
        # 如:AudioMel cepstral Distortion (MCD) 模型
        from shared.audio import AudioQualityModel
        return AudioQualityModel()
    
    def compute_reward(self, outputs, targets):
        """计算包含音频质量的综合 Reward"""
        # 修改点:verl_omni/reward_loop/reward_manager/multi.py:compute_reward()
        # 原方法:只计算视觉 Reward
        # 扩展:计算多模态 Reward
        
        rewards = {}
        
        # 视觉 Reward
        if 'image' in outputs or 'video' in outputs:
            rewards['visual'] = super().compute_reward(outputs, targets)
        
        # 音频 Reward
        if 'audio' in outputs:
            rewards['audio'] = self._compute_audio_reward(
                outputs['audio'], targets.get('audio')
            )
        
        # 文本 Reward
        if 'text' in outputs:
            rewards['text'] = self._compute_text_reward(
                outputs['text'], targets.get('text')
            )
        
        # 融合多模态 Reward
        final_reward = self._fuse_rewards(rewards)
        return final_reward
    
    def _compute_audio_reward(self, audio_output, audio_target):
        """计算音频质量 Reward"""
        # 修改点:新增方法
        # 功能:评估音频质量
        
        # 1. 音频清晰度 (MCD - Mel Cepstral Distortion)
        mcd_score = self.audio_quality_model.compute_mcd(
            audio_output, audio_target
        )
        
        # 2. 音频自然度 (F0 相关性)
        f0_corr = self.audio_quality_model.compute_f0_correlation(
            audio_output, audio_target
        )
        
        # 3. 频谱损失
        spectral_loss = self.audio_quality_model.compute_spectral_loss(
            audio_output, audio_target
        )
        
        # 综合音频质量分数
        audio_quality = (
            0.4 * (1.0 / (1.0 + mcd_score)) +  # MCD 越小越好
            0.3 * f0_corr +                      # F0 相关性越大越好
            0.3 * (1.0 - spectral_loss)          # 频谱损失越小越好
        )
        
        return audio_quality
    
    def _fuse_rewards(self, rewards):
        """融合多模态 Reward"""
        # 修改点:新增方法
        # 功能:加权融合多模态 Reward
        
        weights = {
            'visual': self.config.get('visual_weight', 0.4),
            'audio': self.config.get('audio_weight', 0.3),
            'text': self.config.get('text_weight', 0.3),
        }
        
        total_weight = sum(weights.get(k, 0) for k in rewards.keys())
        
        fused = sum(
            rewards[k] * weights.get(k, 0) / total_weight
            for k in rewards.keys()
        )
        
        return fused
```

#### 2.2.5 Monkey-patch 管理 (patches/)

**承载功能**:
- 集中管理 verl-omni patches
- 版本兼容性检查
- 动态启用/禁用

**修改点对应**:
```python
# plugins/verl_omni/patches/manager.py
"""verl-omni 仓库的 Patch 管理器"""

from shared.patch_manager import BasePatchManager


class VerlOmniPatchManager(BasePatchManager):
    """verl-omni 专用的 Patch 管理器"""
    
    @classmethod
    def register_all_patches(cls):
        """注册所有 verl-omni patches"""
        cls._register_model_patches()
        cls._register_pipeline_patches()
        cls._register_reward_patches()
        cls._register_trainer_patches()
    
    @classmethod
    def _register_model_patches(cls):
        """注册模型相关 patches"""
        # Patch 1: 音频头处理
        cls.register_patch(
            name="audio_head_processing",
            target_module="verl_omni.models.transformers.qwen3_omni_thinker",
            target_attr="Qwen3OmniThinker",
            replacement_fn="plugins.verl_omni.models.audio.audio_head:CustomAudioOmniModelAdapter",
            version_check=lambda: cls._check_verl_omni_version(">=0.2.0"),
            description="添加音频头处理"
        )
        
        # Patch 2: 全模态模型融合
        cls.register_patch(
            name="omni_modal_fusion",
            target_module="verl_omni.pipelines.model_base",
            target_attr="OmniModelBase",
            replacement_fn="plugins.verl_omni.models.omni.custom_omni_model:CustomOmniModelAdapter",
            version_check=lambda: cls._check_verl_omni_version(">=0.2.0"),
            description="增强全模态融合策略"
        )
    
    @classmethod
    def _register_pipeline_patches(cls):
        """注册流水线相关 patches"""
        # Patch: 音频 Flow-GRPO
        cls.register_patch(
            name="audio_flow_grpo",
            target_module="verl_omni.pipelines.model_base",
            target_attr="DiffusionModelBase",
            replacement_fn="plugins.verl_omni.pipelines.audio_flow_grpo.diffusers_training_adapter:AudioFlowGRPOAdapter",
            version_check=lambda: cls._check_verl_omni_version(">=0.2.0"),
            description="添加音频 Flow-GRPO 支持"
        )
    
    @classmethod
    def _register_reward_patches(cls):
        """注册 Reward 相关 patches"""
        # Patch: 音频 Reward
        cls.register_patch(
            name="audio_reward",
            target_module="verl_omni.reward_loop.reward_manager.multi",
            target_attr="MultiVisualRewardManager",
            replacement_fn="plugins.verl_omni.reward_loop.reward_manager.audio_reward_manager:AudioRewardManager",
            version_check=lambda: cls._check_verl_omni_version(">=0.2.0"),
            description="添加音频质量评估"
        )
```

---

## 3. vllm 插件目录 (plugins/vllm/)

针对 vllm 核心仓库的扩展,主要涉及推理引擎、平台支持、算子优化等。

### 3.1 目录结构

```
plugins/vllm/
├── __init__.py                             # vllm 插件入口
├── README.md                               # vllm 插件说明
│
├── platform/                               # 平台扩展
│   ├── __init__.py
│   ├── custom_platform.py                  # 自定义平台
│   └── platform_hooks.py                   # 平台 Hook
│
├── model_executor/                         # 模型执行器扩展
│   ├── __init__.py
│   ├── audio_models/                       # 音频模型支持
│   │   ├── __init__.py
│   │   ├── audio_encoder.py                # 音频编码器
│   │   └── audio_decoder.py                # 音频解码器
│   ├── omni_models/                        # 全模态模型支持
│   │   ├── __init__.py
│   │   └── custom_omni.py                  # 自定义全模态模型
│   └── executor_hooks.py                   # 执行器 Hook
│
├── attention/                              # Attention 扩展
│   ├── __init__.py
│   ├── audio_attention.py                  # 音频 Attention
│   ├── multimodal_attention.py             # 多模态 Attention
│   └── attention_hooks.py                  # Attention Hook
│
├── distributed/                            # 分布式扩展
│   ├── __init__.py
│   ├── custom_communicators.py             # 自定义通信器
│   └── distributed_hooks.py                # 分布式 Hook
│
├── worker/                                 # Worker 扩展
│   ├── __init__.py
│   ├── custom_worker.py                    # 自定义 Worker
│   └── worker_hooks.py                     # Worker Hook
│
└── patches/                                # Monkey-patch 管理
    ├── __init__.py
    ├── manager.py                          # Patch 管理器
    ├── model_patches.py                    # 模型 patches
    ├── attention_patches.py                # Attention patches
    └── distributed_patches.py              # 分布式 patches
```

### 3.2 功能承载点详解

#### 3.2.1 音频模型支持 (model_executor/audio_models/)

**承载功能**:
- 音频编码器推理
- 音频解码器推理
- 音频特征提取

**修改点对应**:
```python
# plugins/vllm/model_executor/audio_models/audio_encoder.py
"""vLLM 中的音频编码器推理支持"""

import torch
import torch.nn as nn
from vllm.model_executor.models import ModelRegistry


class VllmAudioEncoder(nn.Module):
    """vLLM 优化的音频编码器"""
    
    # 修改点:vllm/model_executor/models/registry.py
    # 原方法:只注册文本和视觉模型
    # 扩展:注册音频模型
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用 vLLM 优化的算子
        self.layers = self._build_optimized_layers(config)
    
    def _build_optimized_layers(self, config):
        """构建优化的编码层"""
        # 修改点:vllm/model_executor/models/registry.py
        # 原方法:标准 PyTorch 层
        # 扩展:使用 vLLM 优化的算子(如 Flash Attention)
        
        from vllm.model_executor.layers import Linear, LayerNorm
        
        layers = nn.ModuleList([
            Linear(80, 256),
            LayerNorm(256),
            nn.ReLU(),
            Linear(256, 512),
            LayerNorm(512),
            nn.ReLU(),
            Linear(512, config.hidden_size),
        ])
        
        return layers
    
    def forward(self, audio_features):
        """编码音频特征"""
        # 修改点:vllm/model_executor/models/registry.py
        # 原方法:无音频编码
        # 扩展:添加音频编码
        
        x = audio_features
        for layer in self.layers:
            x = layer(x)
        
        return x


# 注册音频编码器模型
ModelRegistry.register_model(
    "AudioEncoder",
    "plugins.vllm.model_executor.audio_models.audio_encoder:VllmAudioEncoder"
)
```

#### 3.2.2 Monkey-patch 管理 (patches/)

**承载功能**:
- 集中管理 vllm patches
- 算子替换
- 性能优化

**修改点对应**:
```python
# plugins/vllm/patches/manager.py
"""vllm 仓库的 Patch 管理器"""

from shared.patch_manager import BasePatchManager


class VllmPatchManager(BasePatchManager):
    """vllm 专用的 Patch 管理器"""
    
    @classmethod
    def register_all_patches(cls):
        """注册所有 vllm patches"""
        cls._register_model_patches()
        cls._register_attention_patches()
        cls._register_distributed_patches()
    
    @classmethod
    def _register_model_patches(cls):
        """注册模型相关 patches"""
        # Patch: 音频模型支持
        cls.register_patch(
            name="audio_model_support",
            target_module="vllm.model_executor.models.registry",
            target_attr="ModelRegistry",
            replacement_fn="plugins.vllm.model_executor.audio_models:register_audio_models",
            version_check=lambda: cls._check_vllm_version(">=0.6.0"),
            description="添加音频模型支持"
        )
    
    @classmethod
    def _register_attention_patches(cls):
        """注册 Attention 相关 patches"""
        # Patch: 多模态 Attention
        cls.register_patch(
            name="multimodal_attention",
            target_module="vllm.attention.layer",
            target_attr="Attention",
            replacement_fn="plugins.vllm.attention.multimodal_attention:MultimodalAttention",
            version_check=lambda: cls._check_vllm_version(">=0.6.0"),
            description="多模态 Attention 优化"
        )
```

---

## 4. vllm-omni 插件目录 (plugins/vllm_omni/)

针对 vllm-omni 仓库的扩展,主要涉及多模态推理、流水线优化等。

### 4.1 目录结构

```
plugins/vllm_omni/
├── __init__.py                             # vllm-omni 插件入口
├── README.md                               # vllm-omni 插件说明
│
├── pipelines/                              # 推理流水线扩展
│   ├── __init__.py
│   ├── audio_pipeline/                     # 音频推理流水线
│   │   ├── __init__.py
│   │   ├── audio_inference.py              # 音频推理
│   │   └── audio_streaming.py              # 音频流式推理
│   ├── omni_pipeline/                      # 全模态推理流水线
│   │   ├── __init__.py
│   │   ├── omni_inference.py               # 全模态推理
│   │   └── full_duplex.py                  # 全双工推理
│   └── pipeline_hooks.py                   # 流水线 Hook
│
├── models/                                 # 多模态模型扩展
│   ├── __init__.py
│   ├── audio_models/                       # 音频模型
│   │   ├── __init__.py
│   │   └── audio_omni_model.py             # 音频全模态模型
│   └── model_hooks.py                      # 模型 Hook
│
├── worker/                                 # Worker 扩展
│   ├── __init__.py
│   ├── multimodal_worker.py                # 多模态 Worker
│   └── worker_hooks.py                     # Worker Hook
│
└── patches/                                # Monkey-patch 管理
    ├── __init__.py
    ├── manager.py                          # Patch 管理器
    ├── pipeline_patches.py                 # 流水线 patches
    └── model_patches.py                    # 模型 patches
```

### 4.2 功能承载点详解

#### 4.2.1 音频推理流水线 (pipelines/audio_pipeline/)

**承载功能**:
- 音频推理
- 音频流式推理
- 音频质量优化

**修改点对应**:
```python
# plugins/vllm_omni/pipelines/audio_pipeline/audio_inference.py
"""vLLM-Omni 中的音频推理流水线"""

from vllm_omni.pipeline.base import InferencePipeline


class AudioInferencePipeline(InferencePipeline):
    """音频推理流水线"""
    
    # 修改点:vllm-omni/pipeline/base.py:InferencePipeline
    # 原类:InferencePipeline
    # 扩展:添加音频推理支持
    
    def __init__(self, config):
        super().__init__(config)
        self.audio_processor = self._init_audio_processor()
    
    def _init_audio_processor(self):
        """初始化音频处理器"""
        # 修改点:vllm-omni/pipeline/base.py:InferencePipeline
        # 原方法:只处理文本和图像
        # 扩展:添加音频处理
        
        from plugins.vllm_omni.models.audio_models import AudioProcessor
        return AudioProcessor(self.config)
    
    async def infer(self, inputs):
        """执行音频推理"""
        # 修改点:vllm-omni/pipeline/base.py:InferencePipeline:infer()
        # 原方法:只处理文本和图像
        # 扩展:处理音频
        
        # 处理音频输入
        if 'audio' in inputs:
            audio_features = self.audio_processor.process(inputs['audio'])
            inputs['audio_features'] = audio_features
        
        # 执行推理
        outputs = await super().infer(inputs)
        
        # 后处理音频输出
        if 'audio' in outputs:
            outputs['audio'] = self.audio_processor.postprocess(outputs['audio'])
        
        return outputs


# plugins/vllm_omni/pipelines/audio_pipeline/audio_streaming.py
"""音频流式推理,支持全双工"""

import asyncio
from typing import AsyncGenerator


class AudioStreamingPipeline:
    """音频流式推理流水线"""
    
    # 修改点:vllm-omni/pipeline/base.py
    # 原方法:批量推理
    # 扩展:流式推理,支持全双工
    
    def __init__(self, config):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.chunk_size = config.get('chunk_size', 1024)
    
    async def stream_infer(self, audio_stream: AsyncGenerator) -> AsyncGenerator:
        """流式推理音频"""
        # 修改点:vllm-omni/pipeline/base.py
        # 原方法:批量处理
        # 扩展:流式处理
        
        buffer = []
        
        async for audio_chunk in audio_stream:
            buffer.append(audio_chunk)
            
            # 积累足够的 chunk 后开始推理
            if len(buffer) >= self.chunk_size:
                # 处理当前 buffer
                audio_segment = torch.cat(buffer, dim=0)
                features = self.audio_processor.process(audio_segment)
                
                # 推理
                output = await self.infer_segment(features)
                
                # 产出结果
                yield output
                
                # 清空 buffer(保留重叠部分)
                buffer = buffer[self.chunk_size // 2:]
    
    async def infer_segment(self, features):
        """推理单个音频段"""
        # 修改点:vllm-omni/pipeline/base.py
        # 原方法:无流式推理
        # 扩展:添加流式推理
        
        # 使用模型进行推理
        output = await self.model.generate(features)
        return output
```

---

## 5. 共享基础设施 (shared/)

跨仓库共享的工具和基础设施。

### 5.1 目录结构

```
shared/
├── patch_manager/                          # 统一的 Patch 管理器
│   ├── __init__.py
│   ├── base.py                             # 基础 Patch 管理器
│   ├── version_check.py                    # 版本检查工具
│   └── patch_registry.py                   # Patch 注册表
│
├── registry_utils/                         # 注册工具
│   ├── __init__.py
│   ├── model_registry.py                   # 模型注册工具
│   ├── reward_registry.py                  # Reward 注册工具
│   └── pipeline_registry.py                # 流水线注册工具
│
├── audio/                                  # 音频处理工具
│   ├── __init__.py
│   ├── audio_processor.py                  # 音频处理器
│   ├── audio_feature_extractor.py          # 音频特征提取器
│   └── audio_quality_model.py              # 音频质量评估模型
│
├── config/                                 # 配置管理
│   ├── __init__.py
│   ├── config_loader.py                    # 配置加载器
│   └── config_validator.py                 # 配置验证器
│
└── utils/                                  # 通用工具
    ├── __init__.py
    ├── logging.py                          # 日志工具
    └── profiling.py                        # 性能分析工具
```

### 5.2 核心组件

#### 5.2.1 统一 Patch 管理器

```python
# shared/patch_manager/base.py
"""统一的 Patch 管理器基类"""

import logging
import sys
from typing import Callable, Optional
from packaging import version

logger = logging.getLogger(__name__)


class BasePatchManager:
    """基础 Patch 管理器"""
    
    _patches = {}
    _applied_patches = set()
    
    @classmethod
    def register_patch(
        cls,
        name: str,
        target_module: str,
        target_attr: str,
        replacement_fn: str,
        version_check: Optional[Callable[[], bool]] = None,
        description: str = "",
    ):
        """注册一个 patch"""
        cls._patches[name] = {
            'module': target_module,
            'attr': target_attr,
            'replacement_fn': replacement_fn,
            'version_check': version_check,
            'description': description,
            'original': None,
        }
        
        logger.info(f"Registered patch: {name} - {description}")
    
    @classmethod
    def apply_all(cls):
        """应用所有注册的 patches"""
        for name, patch_info in cls._patches.items():
            if name in cls._applied_patches:
                logger.warning(f"Patch {name} already applied")
                continue
            
            # 版本检查
            if patch_info['version_check']:
                if not patch_info['version_check']():
                    logger.warning(f"Version check failed for {name}, skipping")
                    continue
            
            # 应用 patch
            try:
                cls._apply_patch(name, patch_info)
                cls._applied_patches.add(name)
                logger.info(f"Applied patch: {name}")
            except Exception as e:
                logger.error(f"Failed to apply patch {name}: {e}")
    
    @classmethod
    def _apply_patch(cls, name: str, patch_info: dict):
        """应用单个 patch"""
        # 导入目标模块
        module = sys.modules.get(patch_info['module'])
        if module is None:
            __import__(patch_info['module'])
            module = sys.modules[patch_info['module']]
        
        # 保存原始实现
        patch_info['original'] = getattr(module, patch_info['attr'])
        
        # 加载替换函数
        module_path, func_name = patch_info['replacement_fn'].rsplit(':', 1)
        replacement_module = __import__(module_path, fromlist=[func_name])
        replacement = getattr(replacement_module, func_name)
        
        # 应用替换
        setattr(module, patch_info['attr'], replacement)
    
    @classmethod
    def unpatch_all(cls):
        """撤销所有 patches"""
        for name in list(cls._applied_patches):
            patch_info = cls._patches[name]
            
            try:
                module = sys.modules[patch_info['module']]
                setattr(module, patch_info['attr'], patch_info['original'])
                cls._applied_patches.remove(name)
                logger.info(f"Unpatched: {name}")
            except Exception as e:
                logger.error(f"Failed to unpatch {name}: {e}")
    
    @classmethod
    def _check_verl_version(cls, required: str) -> bool:
        """检查 verl 版本"""
        try:
            import verl
            current = version.parse(verl.__version__)
            required = version.parse(required.lstrip('>=<'))
            return current >= required
        except Exception:
            return False
    
    @classmethod
    def _check_verl_omni_version(cls, required: str) -> bool:
        """检查 verl-omni 版本"""
        try:
            import verl_omni
            current = version.parse(verl_omni.__version__)
            required = version.parse(required.lstrip('>=<'))
            return current >= required
        except Exception:
            return False
    
    @classmethod
    def _check_vllm_version(cls, required: str) -> bool:
        """检查 vllm 版本"""
        try:
            import vllm
            current = version.parse(vllm.__version__)
            required = version.parse(required.lstrip('>=<'))
            return current >= required
        except Exception:
            return False
```

---

## 6. 使用示例

### 6.1 安装和启用

```bash
# 安装插件
pip install -e your-plugin-project/

# 在训练脚本中启用
export VERL_USE_EXTERNAL_MODULES=your_plugin
export VLLM_PLUGINS=your_plugin
```

### 6.2 训练脚本示例

```python
# examples/audio_omni_training.py
"""音频全模态训练示例"""

# 导入插件(自动注册所有扩展)
import your_plugin

# 使用增强的训练器
from your_plugin.plugins.verl.trainer import FullDuplexTrainer

# 配置
config = {
    'model_type': 'CustomAudioOmniModel',
    'audio_weight': 0.3,
    'duplex_enabled': True,
    # ... 其他配置
}

# 创建训练器
trainer = FullDuplexTrainer(config)

# 运行全双工训练
trainer.run_duplex_training()
```

---

## 7. 总结

这个按仓库分离的目录结构具有以下优势:

1. **清晰的边界**: 每个仓库的扩展独立管理,易于定位和维护
2. **模块化设计**: 每个功能模块独立,便于测试和复用
3. **版本兼容**: 统一的 patch 管理器处理版本检查
4. **可扩展性**: 基于现有 registry 系统,易于添加新功能
5. **零侵入**: 不修改上游代码,通过 plugin + monkey-patch 注入

下一步建议:
1. 先实现 shared/ 基础设施(patch manager, registry utils)
2. 逐步实现各仓库的插件(从最核心的功能开始)
3. 编写完善的测试和文档
4. 效果验证后,向社区提交 PR 将关键 Hook 点下沉
