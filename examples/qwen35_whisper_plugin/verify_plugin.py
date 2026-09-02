"""
验证插件是否正确注册

运行方式：
    pip install -e .
    python verify_plugin.py
"""

import sys


def verify_plugin():
    """验证插件注册"""
    print("=" * 60)
    print("验证 Qwen3.5 + Whisper 插件")
    print("=" * 60)

    # 1. 导入插件（触发 @OmniModelBase.register 注册）
    print("\n[1/3] 导入插件...")
    try:
        import qwen35_whisper_plugin
        print(f"  ✓ 插件导入成功, 版本: {qwen35_whisper_plugin.__version__}")
    except ImportError as e:
        print(f"  ✗ 插件导入失败: {e}")
        print("  请先安装: pip install -e .")
        return False

    # 2. 检查 OmniModelBase 注册表
    print("\n[2/3] 检查模型注册表...")
    try:
        from verl_omni.pipelines.model_base import OmniModelBase

        registry = OmniModelBase._registry
        print(f"  ✓ 已注册 {len(registry)} 个适配器:")
        for key, cls in registry.items():
            print(f"    {key} -> {cls.__name__}")

        key = ("Qwen35OmniForConditionalGeneration", "thinker")
        if key not in registry:
            print(f"  ✗ 未找到注册键: {key}")
            return False
        print(f"  ✓ Qwen3.5-Omni Thinker 已注册")
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

    # 3. 检查必须实现的方法
    print("\n[3/3] 检查抽象方法实现...")
    try:
        from qwen35_whisper_plugin.models import Qwen35ThinkerAdapter

        required = [
            "get_strip_modules",
            "configure_processor",
            "configure_tokenizer",
            "configure_model",
        ]
        for method in required:
            if hasattr(Qwen35ThinkerAdapter, method):
                print(f"  ✓ {method} 已实现")
            else:
                print(f"  ✗ {method} 未实现")
                return False

        # 测试 get_strip_modules
        strip = Qwen35ThinkerAdapter.get_strip_modules(None)
        print(f"  ✓ get_strip_modules() = {strip}")

    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ 所有检查通过！插件已正确注册")
    print("=" * 60)
    print("\n下一步：")
    print("  在 config.yaml 中设置:")
    print("    actor_rollout_ref:")
    print("      model:")
    print("        external_lib: qwen35_whisper_plugin")
    print("        model_type: omni_model")
    print("        model_stage: thinker")
    return True


if __name__ == "__main__":
    success = verify_plugin()
    sys.exit(0 if success else 1)
