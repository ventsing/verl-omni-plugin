# 数据处理 Add-on 机制

> 数据处理完全可以通过槽位①和槽位④零侵入扩展，不需要改任何上游代码。

---

## 全景：数据从 parquet 到模型输入的完整链路

```
parquet 文件
    ↓
RLHFDataset.__init__()                         ← 槽位④: 可子类化
    ├─ 加载数据 (datasets.load_dataset)
    ├─ 初始化 tokenizer + processor             ← 槽位①: configure_tokenizer / configure_processor
    └─ maybe_filter_out_long_prompts()           ← ⚠ except Exception 静默失败
        ├─ processor is not None → 多模态分支     ← _process_multi_modal_info() 可覆写
        └─ processor is None → tokenizer 分支
    ↓
DataCollator
    ↓
训练 / rollout
```

---

## 6 个数据处理的扩展点

### 扩展点 1：数据集类（槽位④ data.custom_cls）

```python
# verl/verl/utils/dataset/rl_dataset.py:582
if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
    dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
```

- 支持 `pkg://` 前缀路径（`load_extern_object` → `load_module`）
- 加载的类必须继承 `torch.utils.data.Dataset`
- 默认是 `RLHFDataset`

```yaml
# config.yaml
data:
  custom_cls:
    path: pkg://verl_omni_ext.models.minicpmo_5_0.dataset
    name: MiniCPMOThinkerRLHFDataset
```

**零侵入**：不碰上游文件，只在自己的包里写子类 + config 声明。

### 扩展点 2：maybe_filter_out_long_prompts（覆写过滤逻辑）

上游的静默失败陷阱：

```python
# verl/verl/utils/dataset/rl_dataset.py:241
except Exception:
    print("Error processing one of the samples, skipping...")
    traceback.print_exc()
    return self.max_prompt_length + 1    # ← 超长 → 被过滤掉
```

- 两个分支（processor 分支和 tokenizer 分支）都有这个 `except Exception` 兜底
- processor 一抛异常 → 返回 `max_prompt_length + 1` → 超长 → 被过滤
- **7473 行样本可能全被过滤到 0 行** → 只在日志刷 warning
- 静默失败比崩溃贵得多

**Add-on 做法**：子类化，把吞掉的异常喊出来：

```python
# verl_omni_ext/models/minicpmo_5_0/dataset.py
class MiniCPMOThinkerRLHFDataset(RLHFDataset):
    def filter_long_prompts(self, dataset, max_prompt_length, processor=None):
        total = len(dataset)
        kept = 0
        for item in dataset:
            try:
                input_ids = item.get("input_ids", [])
                if len(input_ids) <= max_prompt_length:
                    kept += 1
            except Exception as e:
                # 不吞——喊出来
                raise RuntimeError(
                    f"Dataset filtering failed: {e}. "
                    f"Upstream would have silently filtered this to 0 rows."
                ) from e

        filtered_ratio = 1 - kept / total
        if filtered_ratio > 0.5:
            raise RuntimeError(
                f"Filter ratio {filtered_ratio:.1%} exceeds threshold "
                f"(kept {kept}/{total}). Check processor."
            )
        return dataset
```

**凡是上游有 `except Exception` 吞掉的路径、而你的模型正好会走进去，就用 custom_cls 子类化它。**

### 扩展点 3：多模态数据处理（_process_multi_modal_info）

verl-omni 自带的 `QwenOmniRLHFDataset` 示范了正确做法：

```python
# verl_omni/utils/dataset/omni_rl_datasets.py
class QwenOmniRLHFDataset(RLHFDataset):
    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        from qwen_omni_utils import process_mm_info
        # Qwen 返回 (audios, images, videos)；verl 期望 (images, videos, audios)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        # 音频 pad 到 hop 倍数——保证 actor recompute 和 vllm-omni rollout 帧数一致
        if audios is not None:
            audios = [pad_audio_to_hop_multiple(a) for a in audios]
        return images, videos, audios
```

**关键经验：音频 pad 到 hop_length 倍数**
- Whisper mel-frame stride at 16kHz = 160 samples
- vllm-omni 在 feature extraction 前 pad 到 hop 倍数
- HF 侧如果不 pad → 帧数不一致 → prompt 长度不一致 → rollout 和 train 对不上
- `pad_audio_to_hop_multiple()` 解决这个问题

换模型时：
- 换音频编码头 → 可能需要换 `process_mm_info` → 必须覆写 `_process_multi_modal_info`
- 换 LLM 主干（不换音频/视觉） → 一般不动

### 扩展点 4：processor 配置（槽位① configure_processor）

```python
# 适配器的 configure_processor 方法
@classmethod
def configure_processor(cls, model_path, model_config):
    processor = AutoProcessor.from_pretrained(model_path)
    # 绑定 RoPE 位置编码
    processor.get_rope_index = types.MethodType(_get_rope_index, processor)
    # 绑定 pad token 去重
    processor.dedup_pad_tokens = types.MethodType(_dedup_pad_tokens, processor)
    return processor
```

**最容易翻车的槽位**：
- `verl.utils.hf_processor` 用白名单 match processor 类名
- 类名不在六个已知里 → 走 `raise ValueError` → 外层 `except` 吞成 `None`
- processor 退化成 `None` → 走 tokenizer 分支 → 静默丢多模态信息

换任何非 Qwen 系模型，这一条先查。

### 扩展点 5：tokenizer 配置（槽位① configure_tokenizer）

```python
@classmethod
def configure_tokenizer(cls, model_path, model_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 加载 chat_template.json
    chat_template_path = os.path.join(model_path, "chat_template.json")
    if os.path.exists(chat_template_path):
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f).get("chat_template")
    return tokenizer
```

### 扩展点 6：数据集 schema 前置自检

在启动脚本里校验 parquet 列名，把"跑到第 40 分钟才炸"变成"启动 3 秒内退出"：

```bash
# examples/qwen3_5_moe/run_*.sh
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_PATH')
assert len(df) > 0, 'Dataset is empty!'
required_cols = ['prompt', 'reward_model']
for col in required_cols:
    assert col in df.columns, f'Missing column: {col}'
print(f'✓ Dataset: {len(df)} rows')
"
```

---

## 换模型时的数据改动矩阵

| 变更 | 槽位④（数据集类） | 槽位①configure_processor | 槽位①configure_tokenizer |
|------|-------------------|-------------------------|--------------------------|
| 换 LLM 主干 | 一般不动 | 标准 hf_processor | 标准（trust_remote_code） |
| 换音频编码头 | 覆写 `_process_multi_modal_info` + `pad_audio_to_hop_multiple` | 可能需要自建 | 一般不动 |
| 换视频编码头 | 覆写 `_process_multi_modal_info` | m-RoPE / position_ids 绑定 | 一般不动 |
| 非 Qwen 系模型 | 覆写 `maybe_filter_out_long_prompts`（防静默失败） | **必须自建**（绕过白名单） | trust_remote_code=True |

---

## 总结

| 层次 | 扩展点 | 机制 | 侵入性 |
|------|--------|------|--------|
| 数据集类 | 槽位④ | `data.custom_cls` + `load_extern_object` | ✅ 零侵入 |
| 数据过滤 | 槽位④子类 | 覆写 `maybe_filter_out_long_prompts` | ✅ 零侵入 |
| 多模态数据处理 | 槽位④子类 | 覆写 `_process_multi_modal_info` | ✅ 零侵入 |
| processor 配置 | 槽位① | `configure_processor` | ✅ 零侵入 |
| tokenizer 配置 | 槽位① | `configure_tokenizer` | ✅ 零侵入 |
| schema 前置自检 | 槽位⑤ | 启动脚本 | ✅ 零侵入 |

**数据处理完全零侵入。** 唯一的陷阱是上游的 `except Exception` 静默吞异常——用 `custom_cls` 子类化把它变成显式报错。
