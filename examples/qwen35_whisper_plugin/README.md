# Qwen3.5 + Whisper 插件 for verl-omni

通过 verl-omni 的 **`external_lib` 机制**（零侵入）注入 Qwen3.5 + Whisper 多模态模型适配器。

## 零侵入机制

verl-omni 自带 `external_lib` 配置项，加载流程：

```
config.yaml 设置 external_lib: qwen35_whisper_plugin
    ↓
verl-omni 的 OmniModelConfig.__post_init__()
    ↓
import_external_libs("qwen35_whisper_plugin")   # importlib.import_module()
    ↓
执行 __init__.py: from .models import qwen35_thinker
    ↓
@OmniModelBase.register("Qwen35OmniForConditionalGeneration", "thinker")
    ↓
注册到 OmniModelBase._registry 字典
    ↓
verl-omni 通过 get_class_by_name() 查找并调用适配器
```

**不修改 verl-omni 任何源码。**

## 安装

```bash
cd examples/qwen35_whisper_plugin
pip install -e .
```

## 使用

### 1. config.yaml 配置

```yaml
actor_rollout_ref:
  model:
    path: /path/to/qwen35-omni
    external_lib: qwen35_whisper_plugin   # 你的插件包名
    model_type: omni_model
    model_stage: thinker
    trust_remote_code: true
```

### 2. 启动训练

```bash
python -m verl_omni.trainer.main_omni --config-path ./config --config-name qwen35_omni
```

### 3. 验证插件

```bash
python verify_plugin.py
```

## 模型 config.json 要求

模型目录的 `config.json` 中 `architectures[0]` 必须与注册键一致：

```json
{
  "architectures": ["Qwen35OmniForConditionalGeneration"]
}
```

## 实现的方法

| 方法 | 必须? | 作用 |
|------|-------|------|
| `get_strip_modules()` | ✅ | 返回需要剥离的模块（talker/codec） |
| `configure_processor()` | ✅ | 加载多模态处理器（含 RoPE/dedup） |
| `configure_tokenizer()` | ✅ | 加载 tokenizer + chat_template |
| `configure_model()` | 可选 | 剥离模块 + 重定向 forward 到 thinker |

## 参考实现

verl-omni 自带的 Qwen3-Omni 适配器：
- 文件：`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`
- 注册：`@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")`

## License

Apache-2.0
