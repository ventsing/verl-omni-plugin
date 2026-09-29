"""
vllm-omni 原生插件注册链路单测（零 vllm/vllm-omni 依赖）

被测对象：
  verl_omni_ext/vllm_omni_plugins.py          — 入口点目标 register()
  verl_omni_ext/models/*/vllm_omni/__init__.py — 各模型 register()

手法（同 test_patchkit）：sys.modules 塞假 vllm_omni 模块树，
按文件路径加载被测模块（绕过 verl_omni_ext/__init__ 的 verl 依赖），
register() 内的 `from vllm_omni... import ...` 会命中假模块。

验证点：
  1. register_pipeline 被调用，OMNI_PIPELINES 拿到两个 model_type 键
  2. register_model（OmniModelRegistry / 上游 ModelRegistry）未被调用
     ——当前无本包自带模型实现，且不得覆盖上游 MiniCPMO
  3. 幂等：重复调用不重复注册
  4. 单模型失败不拖垮其它模型（降级策略）
  5. pyproject 入口点声明与实现一致（vllm_omni.general_plugins 组）
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


# ============================================================================
# 假 vllm_omni / vllm 模块树
# ============================================================================

class _RecordingRegistry:
    """OmniModelRegistry / ModelRegistry 的假实现：记录 register_model。"""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def register_model(self, arch, model_cls):
        self.calls.append((arch, model_cls))


def _install_fakes():
    calls = {"register_pipeline": []}
    omni_pipelines = {}

    def register_pipeline(pipeline, model_type=None):
        mt = model_type if model_type is not None else getattr(pipeline, "model_type", None)
        calls["register_pipeline"].append(mt)
        omni_pipelines[mt] = pipeline

    omni_model_registry = _RecordingRegistry("OmniModelRegistry")
    upstream_model_registry = _RecordingRegistry("ModelRegistry")

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        # 挂接父属性：sys.modules 命中时 import 系统不会自动 setattr 父模块
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, m)
        return m

    # stage_config 假件：pipeline.py 构造 PipelineConfig 时用
    class StageExecutionType:
        LLM_AR = "llm_ar"
        LLM_GENERATION = "llm_generation"
        DIFFUSION = "diffusion"

    def StagePipelineConfig(**kw):
        return types.SimpleNamespace(**kw)

    def PipelineConfig(**kw):
        return types.SimpleNamespace(**kw)

    # vllm 上游
    _mod("vllm")
    _mod("vllm.model_executor")
    _mod("vllm.model_executor.models")
    _mod("vllm.model_executor.models.registry", ModelRegistry=upstream_model_registry)

    # vllm_omni
    _mod("vllm_omni")
    _mod("vllm_omni.config")
    _mod(
        "vllm_omni.config.stage_config",
        PipelineConfig=PipelineConfig,
        StageExecutionType=StageExecutionType,
        StagePipelineConfig=StagePipelineConfig,
    )
    _mod(
        "vllm_omni.config.pipeline_registry",
        register_pipeline=register_pipeline,
        OMNI_PIPELINES=omni_pipelines,
    )
    _mod("vllm_omni.model_executor")
    _mod("vllm_omni.model_executor.models", OmniModelRegistry=omni_model_registry)

    return {
        "calls": calls,
        "omni_pipelines": omni_pipelines,
        "omni_model_registry": omni_model_registry,
        "upstream_model_registry": upstream_model_registry,
    }


def _load_pkg(fake_name, dir_path):
    """按文件路径加载一个包（带 __path__，让 `from . import pipeline` 可用）。"""
    spec = importlib.util.spec_from_file_location(
        fake_name,
        os.path.join(dir_path, "__init__.py"),
        submodule_search_locations=[dir_path],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fake_name] = mod
    spec.loader.exec_module(mod)
    return mod


FAKES = _install_fakes()

# 加载被测模块（fake 名字避免触碰真实 verl_omni_ext 包名）
_ext_plugins_pkg = _load_pkg(
    "verl_omni_ext_vo_test", os.path.join(ROOT, "verl_omni_ext")
)
# ^ 这会执行 verl_omni_ext/__init__.py 的 _load_all()——入口点组遍历失败
#   （本会话没装 verl/verl-omni）会逐项降级跳过，不影响本测试。

_qwen_vo = _load_pkg(
    "verl_omni_ext_vo_test.models.qwen3_5_moe.vllm_omni",
    os.path.join(ROOT, "verl_omni_ext", "models", "qwen3_5_moe", "vllm_omni"),
)
_cpm_vo = _load_pkg(
    "verl_omni_ext_vo_test.models.minicpmo_5_0.vllm_omni",
    os.path.join(ROOT, "verl_omni_ext", "models", "minicpmo_5_0", "vllm_omni"),
)

# 中央入口点目标：vllm_omni_plugins.py 是平文件，直接 spec 加载
_spec = importlib.util.spec_from_file_location(
    "verl_omni_ext_vo_test.vllm_omni_plugins",
    os.path.join(ROOT, "verl_omni_ext", "vllm_omni_plugins.py"),
)
_plugins_mod = importlib.util.module_from_spec(_spec)
sys.modules["verl_omni_ext_vo_test.vllm_omni_plugins"] = _plugins_mod
_spec.loader.exec_module(_plugins_mod)

# 但 register() 内部 import 的是真实包名 verl_omni_ext.models...
# 测试里没有真实包（verl 依赖），改走各模型 register() 直测 + 中央模块
# 的编排逻辑用 monkey 方式验证：把真实包名指向我们的 fake 树。
sys.modules.setdefault("verl_omni_ext", _ext_plugins_pkg)
_real_models_pkg = types.ModuleType("verl_omni_ext.models")
_real_models_pkg.__path__ = [os.path.join(ROOT, "verl_omni_ext", "models")]
sys.modules.setdefault("verl_omni_ext.models", _real_models_pkg)
_qwen_pkg = types.ModuleType("verl_omni_ext.models.qwen3_5_moe")
_qwen_pkg.__path__ = [os.path.join(ROOT, "verl_omni_ext", "models", "qwen3_5_moe")]
sys.modules.setdefault("verl_omni_ext.models.qwen3_5_moe", _qwen_pkg)
sys.modules.setdefault("verl_omni_ext.models.qwen3_5_moe.vllm_omni", _qwen_vo)
_cpm_pkg = types.ModuleType("verl_omni_ext.models.minicpmo_5_0")
_cpm_pkg.__path__ = [os.path.join(ROOT, "verl_omni_ext", "models", "minicpmo_5_0")]
sys.modules.setdefault("verl_omni_ext.models.minicpmo_5_0", _cpm_pkg)
sys.modules.setdefault("verl_omni_ext.models.minicpmo_5_0.vllm_omni", _cpm_vo)


# ============================================================================
# 1. 各模型 register()：pipeline 注册 + 不碰模型类注册表
# ============================================================================
def test_model_register_pipeline():
    _qwen_vo.register()
    assert "qwen3_5_moe" in FAKES["omni_pipelines"], "qwen3_5_moe 未进 OMNI_PIPELINES"
    assert FAKES["omni_pipelines"]["qwen3_5_moe"] is _qwen_vo.QWEN3_5_MOE_PIPELINE
    print("  ✓ qwen3_5_moe pipeline → OMNI_PIPELINES")


def test_minicpmo_register_pipeline():
    _cpm_vo.register()
    assert "minicpmo_5_0" in FAKES["omni_pipelines"], "minicpmo_5_0 未进 OMNI_PIPELINES"
    assert FAKES["omni_pipelines"]["minicpmo_5_0"] is _cpm_vo.MINICPMO_5_0_PIPELINE
    print("  ✓ minicpmo_5_0 pipeline → OMNI_PIPELINES")


def test_no_model_class_clobber():
    """当前无本包模型实现——两个注册表必须零调用（不覆盖上游 MiniCPMO）。"""
    assert FAKES["omni_model_registry"].calls == [], (
        f"OmniModelRegistry 不应被调用: {FAKES['omni_model_registry'].calls}"
    )
    assert FAKES["upstream_model_registry"].calls == [], (
        f"上游 ModelRegistry 不应被调用: {FAKES['upstream_model_registry'].calls}"
    )
    print("  ✓ 未覆盖上游模型类注册（MiniCPMO 等条目安全）")


def test_model_register_idempotent():
    n = len(FAKES["calls"]["register_pipeline"])
    _qwen_vo.register()
    _cpm_vo.register()
    assert len(FAKES["calls"]["register_pipeline"]) == n, "重复 register 重复注册"
    print("  ✓ 模型级 register() 幂等")


# ============================================================================
# 2. 中央入口点：编排 + 降级 + 幂等
# ============================================================================
def test_central_register_orchestrates():
    _plugins_mod._registered = False
    _plugins_mod.register()
    mts = set(FAKES["calls"]["register_pipeline"])
    assert {"qwen3_5_moe", "minicpmo_5_0"} <= mts, f"中央入口未编排全部模型: {mts}"
    print("  ✓ 中央 register() 编排两个模型")


def test_central_register_idempotent():
    n = len(FAKES["calls"]["register_pipeline"])
    _plugins_mod.register()
    assert len(FAKES["calls"]["register_pipeline"]) == n, "中央重复 register 重复注册"
    print("  ✓ 中央 register() 幂等")


def test_central_register_degrades_gracefully():
    """单模型注册失败不拖垮其它模型。"""
    _plugins_mod._registered = False
    calls = FAKES["calls"]["register_pipeline"]

    def _boom():
        raise RuntimeError("simulated failure")

    # 直接取 fake 树里的模块（import a.b as x 语句需要父属性链，fake 树没有）
    q = sys.modules["verl_omni_ext.models.qwen3_5_moe.vllm_omni"]
    real = q.register
    q.register = _boom
    try:
        _plugins_mod.register()
    finally:
        q.register = real
        _plugins_mod._registered = False
    # minicpmo 仍应注册成功（calls 里至少各一次）
    assert calls.count("minicpmo_5_0") >= 1, "降级失败：好的模型也被拖垮"
    # 恢复后再来一次全量注册，确认自愈
    _plugins_mod.register()
    assert calls.count("qwen3_5_moe") >= 1
    print("  ✓ 单模型失败降级，其它模型照常注册，且可自愈重试")


# ============================================================================
# 3. pyproject 入口点声明一致
# ============================================================================
def test_pyproject_entry_point():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        content = f.read()
    assert '[project.entry-points."vllm_omni.general_plugins"]' in content, (
        "pyproject 缺 vllm_omni.general_plugins 入口点组"
    )
    assert 'verl_omni_ext = "verl_omni_ext.vllm_omni_plugins:register"' in content, (
        "入口点目标应为 verl_omni_ext.vllm_omni_plugins:register"
    )
    assert hasattr(_plugins_mod, "register"), "入口点目标函数不存在"
    print("  ✓ pyproject 入口点声明与实现一致")


# ============================================================================
# runner（无 pytest 环境用 __main__ 模式）
# ============================================================================
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
