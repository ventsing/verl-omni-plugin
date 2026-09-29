"""
vllm-omni 侧注册入口（原生插件机制，零 patch，L1）

vllm-omni 自带入口点插件组 ``vllm_omni.general_plugins``
（vllm_omni/plugins/__init__.py）。``load_omni_general_plugins()`` 在所有
进程（主进程 / 引擎核心 / stage 子进程 / worker）发现并**执行**本组的
入口点函数——不只是 import，是调用：

  OmniEngineArgs.__post_init__ → load_omni_general_plugins()   # engine/arg_utils.py:252
  worker mixins / stage_init_utils / diffusion worker           # 同样调用

时序保证：插件函数执行早于 ``ModelConfig.registry`` 首次消费与
``resolve_pipeline_config``——register() 里做的一切注册都能被引擎看到。

pyproject.toml 声明（pip install 后自动被发现）：

  [project.entry-points."vllm_omni.general_plugins"]
  verl_omni_ext = "verl_omni_ext.vllm_omni_plugins:register"

为什么这取代了 GP-004 gate patch（vllm_omni_external_modules.patch）。

旧方案（GP-004：env 驱动 import 外部模块，往 ``_OMNI_MODELS`` 塞元组）
有三个致命问题：

  1. **路径拼接 bug**：元组值被硬编码前缀拼接成
     ``vllm_omni.model_executor.models.{mod_folder}.{mod_relname}``
     ——外部包路径（verl_omni_ext...）拼不进去，惰性导入时必
     ModuleNotFoundError。注册了等于埋雷。
  2. **时机死锁**：注入点在 registry 模块 import 期（:477），早于
     ``OmniModelRegistry`` 实例构造（:484）——想改用实例 API
     ``register_model`` 都不可能，只能塞元组（回到问题 1）。
  3. **静默覆盖**：字典赋值会无告警覆盖上游同名 arch
     （如 ``MiniCPMO``，上游 minicpmo_4_5 已注册）。

新方案全部使用 vllm-omni 公开 API：

  - ``register_pipeline(PipelineConfig)``      # config/pipeline_registry.py:192
    docstring 明说 "Register an out of tree pipeline"——官方树外注册口。
  - ``OmniModelRegistry.register_model(arch, "module:Class")``
    惰性字符串支持任意模块路径（外部包 OK）；重复注册打日志告警。
  - （可选）vllm 上游 ``ModelRegistry.register_model`` 同步——
    vllm-omni 自己在 engine/arg_utils.py:138 就这么干。

要求：verl_omni_ext 必须 pip install（entry_points 发现只看已安装的
distribution）。这与训练侧 verl_omni.models 入口点组的假设一致，无新增
约束。rollout 独立部署时，推理环境也要装本包。

本模块 import 期零副作用；register() 幂等（插件会被多个进程各自加载
一次，进程内也可能多次触发——vllm_omni/plugins/__init__.py 的文档
明确要求插件可重入）。
"""

import logging

logger = logging.getLogger(__name__)

_registered = False


def register():
    """vllm_omni.general_plugins 入口点目标。幂等，可安全多次调用。

    注册内容（全部树外、零 patch）：
      1. 各模型的 pipeline 拓扑（PipelineConfig → OMNI_PIPELINES）
      2. 本包自带的 vllm 模型类实现（如有——见各模型的
         ``_maybe_register_model_class``）
    """
    global _registered
    if _registered:
        return

    # 逐模型注册；单模型失败不拖垮其它模型（与 _load_all 的降级策略一致）
    from verl_omni_ext.models.qwen3_5_moe.vllm_omni import register as _register_qwen

    from verl_omni_ext.models.minicpmo_5_0.vllm_omni import register as _register_cpm

    for name, fn in (
        ("qwen3_5_moe", _register_qwen),
        ("minicpmo_5_0", _register_cpm),
    ):
        try:
            fn()
            logger.debug("vllm-omni 插件注册完成: %s", name)
        except Exception:  # noqa: BLE001
            logger.exception("vllm-omni 插件注册失败（跳过）: %s", name)

    _registered = True
