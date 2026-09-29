"""
MiniCPM-o vllm-omni 侧注册（原生插件机制）

加载链路（零 patch，pip install verl-omni-ext 后自动生效）：

  vllm_omni.general_plugins 入口点
    → verl_omni_ext.vllm_omni_plugins.register()
    → 本模块 register()

注意：本模块**只注册 pipeline 拓扑，不注册模型类**。
``MiniCPMO`` 架构上游已注册（minicpmo_4_5 等模型在用），重复注册会
**静默覆盖**上游条目——这是旧 GP-004 方案的隐患之三。我们注册的是
``model_type="minicpmo_5_0"`` 的拓扑（不同的 OMNI_PIPELINES 键），
复用上游模型类。
"""

from .pipeline import MINICPMO_5_0_PIPELINE

_registered = False


def register():
    """注册 pipeline 拓扑。幂等。模型类复用上游，不重注册。"""
    global _registered
    if _registered:
        return

    from vllm_omni.config.pipeline_registry import register_pipeline

    register_pipeline(MINICPMO_5_0_PIPELINE)
    _registered = True
