"""
MiniCPM-o 适配器边界 CPU 测试
"""
import pytest


class TestMiniCPMOAdapter:
    def test_adapter_registered(self):
        import verl_omni_ext
        from verl_omni.pipelines.model_base import OmniModelBase

        key = ("MiniCPMO", "thinker")
        assert key in OmniModelBase._registry

    def test_strip_modules_empty(self):
        """不剥 tts.*——保持 state_dict 与 checkpoint 逐字相同"""
        import verl_omni_ext
        from verl_omni.pipelines.model_base import OmniModelBase

        adapter = OmniModelBase._registry[("MiniCPMO", "thinker")]
        assert adapter.get_strip_modules(None) == []


class TestMiniCPMODataset:
    def test_filter_ratio_threshold(self):
        """过滤比例超阈值时报错而不是静默继续"""
        from verl_omni_ext.models.minicpmo_5_0.dataset import MiniCPMOThinkerRLHFDataset

        ds = MiniCPMOThinkerRLHFDataset()
        # 模拟全部超长 → 100% 被过滤 → 应该报错
        fake_data = [{"input_ids": [1] * 99999} for _ in range(10)]
        with pytest.raises(RuntimeError, match="exceeds threshold"):
            ds.filter_long_prompts(fake_data, max_prompt_length=100)
