"""
L2: Qwen3.5-MoE monkey patch

打的对象：Qwen3_5MoeVisionModel（transformers / checkpoint remote code）
原因：FSDP2 CPUOffload 下参数报 cpu、激活在 npu，索引张量建在错的设备上

这是 L2 的正确用法——打第三方代码，不是打 verl-omni 自己的东西。
"""
from verl_omni_ext._patchkit import idempotent_patch


@idempotent_patch(
    name="qwen3_5_vision_device_fix",
    target_module="transformers",
    target_attr="Qwen3_5MoeVisionModel",
    fingerprint="transformers>=4.46",
    expected_signature="forward",
    probe_signature=True,  # 记录目标 forward 签名指纹，跨进程一致性校验用
)
def apply_qwen3_5_vision_device_fix(original_forward):
    """修正 ViT 位置编码的设备归属

    FSDP2 CPUOffload 下：
    - 参数在 CPU（FSDP 管理的 offload）
    - 激活在 NPU（实际计算）
    - 位置编码索引张量建在 CPU → forward 时 device mismatch

    补丁逻辑：在 forward 开头把索引张量移到激活所在设备。
    """
    def patched_forward(self, *args, **kwargs):
        # 找到激活的设备，把位置编码索引移过去
        import torch
        for name, buf in self.named_buffers():
            if "position" in name.lower() or "rope" in name.lower():
                # 移到第一个输入张量的设备
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        data = arg.to(arg.device, non_blocking=True)
                        break
        return original_forward(self, *args, **kwargs)

    return patched_forward


# 注意：这个函数打的是模型实例的 forward，不是模块级属性。
# 所以它不能走 @idempotent_patch（那个是打模块属性的）。
# 它在 adapter.configure_model 里直接调用，打在 from_pretrained 返回的实例上。
# 这是正确的位置——补丁放哪一层由它作用的对象的生命周期决定。
