"""
Qwen3.5-MoE 适配器边界 CPU 测试

目的（应对 L2 静默失效）：
  上游改了被 patch 的函数 → CPU 测试红 → 有人知道 → 而不是等 16 卡启动才发现。
"""
import pytest


class TestQwen35MoeAdapter:
    """测试 Qwen35MoeThinkerAdapter 的注册和方法"""

    def test_adapter_registered(self):
        """适配器注册到 OmniModelBase._registry"""
        import verl_omni_ext  # 触发入口点自动发现
        from verl_omni.pipelines.model_base import OmniModelBase

        key = ("Qwen3_5MoeForConditionalGeneration", "thinker")
        assert key in OmniModelBase._registry, (
            f"Adapter {key} not registered! "
            "Check pyproject.toml entry-points and verl_omni_ext/__init__.py _load_all()"
        )

    def test_strip_modules_empty(self):
        """get_strip_modules 返回空列表（全量训练 ViT）"""
        import verl_omni_ext
        from verl_omni.pipelines.model_base import OmniModelBase

        adapter = OmniModelBase._registry[("Qwen3_5MoeForConditionalGeneration", "thinker")]
        assert adapter.get_strip_modules(None) == []


class TestQwen35MoePatches:
    """测试 L2 monkey patch 的行为"""

    def test_patch_self_check(self):
        """所有补丁都打上了"""
        from verl_omni_ext._patchkit import self_check
        results = self_check()
        if results:  # 如果 transformers 没装，results 为空，这是正常的降级
            assert all(results.values()), f"Patches not applied: {results}"


class TestQwen35MoeRollout:
    """测试 rollout adapter"""

    def test_rollout_registered(self):
        import verl_omni_ext
        from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
        assert "qwen3_5_moe" in OmniRolloutPipelineBase._registry

    def test_pipeline_id_not_equal_to_register_key(self):
        """注册键 ≠ pipeline id"""
        import verl_omni_ext
        from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

        adapter = OmniRolloutPipelineBase._registry["qwen3_5_moe"]
        assert adapter.get_pipeline_id() == "qwen3_5_moe_thinker_only"
        assert adapter.get_pipeline_id() != "qwen3_5_moe"
