# 示例索引

## 真实模型适配脚本

### qwen3_5_moe/
- `run_qwen35_moe_thinker_gspo_npu.sh` — 启动脚本 + 前置自检（槽位⑤）
- `probes/v0_forward_signature.sh` — 探针：先测量再写适配器

### minicpmo_5_0/
- `run_minicpmo_thinker_gspo_npu.sh` — 从 Qwen3.5 移植，标注 6 处 [MiniCPM] diff 点
- `probes/v0_forward_signature.sh` — 探针：forward 签名 + processor 行为 + AutoClass

## 教学骨架

### qwen35_whisper_plugin/
最小实现：`@OmniModelBase.register` + 3 个抽象方法。适合学习"加新模型只需 3 步"。

## 探针方法论

> 适配决策是被测量出来的，不是被猜出来的。
> 先写探针把 6~10 个关键事实测出来，再写适配器。

推荐探针清单（从 MiniCPM-o 的 v0_*.sh 提炼）：

| 探针 | 测什么 | 为什么要测 |
|------|--------|-----------|
| v0_forward_signature | forward 签名 | 单位置 vs 全关键字不兼容要到第一个 micro-batch 才炸 |
| v0_position_ids | position_ids 计算 | m-RoPE 模式 |
| v0_mrope | m-RoPE 模式 | 多模态位置编码契约 |
| v0_backbone | backbone 加载 | 确认主干（决定能否搬显存/精度结论） |
| v0_chat_template | chat template | 对话格式 |
| v0_mtp | MTP / thinking | 是否有 MTP 头、显存占用 |
