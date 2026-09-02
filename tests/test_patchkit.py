"""
_patchkit 增强功能单测：多进程传播证明 + 签名指纹 + watchdog + strict 模式

_patchkit 本身只依赖标准库（importlib/inspect/hashlib/json），
不需要 verl / transformers / torch，可以纯 CPU 单测。

注意：不能 `from verl_omni_ext import _patchkit`——那会执行包 __init__，
触发 verl 依赖的 features import。用 importlib 直接从文件加载。
"""
import importlib
import importlib.util
import json
import os

SPEC = importlib.util.spec_from_file_location(
    "_patchkit_standalone",
    os.path.join(os.path.dirname(__file__), "..", "verl_omni_ext", "_patchkit.py"),
)
pk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pk)

# 假目标模块（同目录）——注册到 sys.modules，供 _patchkit 的 import_module 找到
import sys

TGT_SPEC = importlib.util.spec_from_file_location(
    "_patchkit_test_probe_target",
    os.path.join(os.path.dirname(__file__), "..", "_patchkit_test_probe_target.py"),
)
tgt = importlib.util.module_from_spec(TGT_SPEC)
TGT_SPEC.loader.exec_module(tgt)
sys.modules["_patchkit_test_probe_target"] = tgt


def _reset():
    pk._PATCH_REGISTRY.clear()


# ============================================================================
# 签名指纹
# ============================================================================

def test_signature_fingerprint_stable_and_sensitive():
    f1 = lambda a, b, c=1: None  # noqa: E731
    f2 = lambda a, b, c=1: None  # noqa: E731
    f3 = lambda a, b, **kw: None  # noqa: E731
    assert pk._signature_fingerprint(f1) == pk._signature_fingerprint(f2)
    assert pk._signature_fingerprint(f1) != pk._signature_fingerprint(f3)
    assert len(pk._signature_fingerprint(f1)) == 16


def test_signature_fingerprint_no_introspection():
    assert pk._signature_fingerprint(42) == "N/A"


# ============================================================================
# probe_signature：签名指纹记录 + __patch_signature__ 注入
# ============================================================================

def test_probe_signature_records_and_injects():
    _reset()
    captured = {}

    @pk.idempotent_patch(
        name="test_probe",
        target_module="_patchkit_test_probe_target",
        target_attr="Target",
        fingerprint="test",
        expected_signature="method",
        probe_signature=True,
    )
    def my_patch(original_method, __patch_signature__=None):
        captured["injected"] = __patch_signature__
        return original_method

    my_patch()

    assert captured["injected"] is not None
    assert len(captured["injected"]) == 16
    rec = pk._PATCH_REGISTRY["test_probe"]
    assert rec.applied is True
    assert rec.signature_fp == captured["injected"]
    _reset()


# ============================================================================
# 多进程传播证明
# ============================================================================

def test_get_patch_state_and_line_serializable():
    _reset()
    pk._PATCH_REGISTRY["demo"] = pk.PatchRecord(
        "demo", "mod.Target", "transformers>=4.46", True, "abc1234567890123"
    )
    state = pk.get_patch_state()
    json.dumps(state)  # 必须可 JSON 序列化（Ray 上报前提）
    assert state["demo"]["applied"] is True
    assert state["demo"]["signature_fp"] == "abc1234567890123"

    line = pk.patch_state_line()
    assert line.startswith("PATCH_STATE ")
    parsed = json.loads(line[len("PATCH_STATE "):])
    assert parsed["demo"]["applied"] is True
    _reset()


def _mk_state(name, applied, sig_fp):
    return {
        name: {
            "applied": applied,
            "target": "mod.Target",
            "fingerprint": "t",
            "signature_fp": sig_fp,
        }
    }


def test_consensus_ok():
    states = [_mk_state("p", True, "fp1"), _mk_state("p", True, "fp1")]
    ok, msg = pk.assert_patch_consensus(states)
    assert ok, msg


def test_consensus_missing_patch_in_one_worker():
    ok, msg = pk.assert_patch_consensus(
        [_mk_state("p", True, "fp1"), _mk_state("q", True, "fp1")]
    )
    assert not ok
    assert "patch set differs" in msg


def test_consensus_not_applied():
    ok, msg = pk.assert_patch_consensus(
        [_mk_state("p", True, "fp1"), _mk_state("p", False, "fp1")]
    )
    assert not ok
    assert "NOT applied" in msg


def test_consensus_signature_drift():
    """不同 worker 签名指纹不同（版本倾斜）→ False"""
    ok, msg = pk.assert_patch_consensus(
        [_mk_state("p", True, "fp1"), _mk_state("p", True, "fp2")]
    )
    assert not ok
    assert "fingerprint differs" in msg


def test_consensus_empty():
    ok, _ = pk.assert_patch_consensus([])
    assert not ok


# ============================================================================
# 运行时 watchdog
# ============================================================================

def test_verify_patches_alive():
    _reset()

    @pk.idempotent_patch(
        name="test_watchdog",
        target_module="_patchkit_test_probe_target",
        target_attr="Target",
        fingerprint="test",
        expected_signature="method",
    )
    def my_patch(original_method):
        return original_method

    my_patch()
    assert pk.verify_patches_alive()["test_watchdog"] is True

    # 模拟运行中被覆盖/清 flag
    setattr(tgt.Target, "_verl_omni_ext_patched_test_watchdog", False)
    assert pk.verify_patches_alive()["test_watchdog"] is False
    _reset()


# ============================================================================
# strict 模式
# ============================================================================

def test_strict_mode_raises():
    """VERL_OMNI_EXT_PATCH_STRICT=1 时未打上 → raise"""
    _reset()
    os.environ["VERL_OMNI_EXT_PATCH_STRICT"] = "1"
    try:

        @pk.idempotent_patch(
            name="test_strict",
            target_module="_patchkit_test_probe_target",
            target_attr="DoesNotExist",
            fingerprint="test",
        )
        def my_patch(original):
            return original

        try:
            my_patch()
            assert False, "strict 应 raise"
        except RuntimeError:
            pass  # 预期
    finally:
        os.environ.pop("VERL_OMNI_EXT_PATCH_STRICT", None)
        _reset()


if __name__ == "__main__":
    # 无 pytest 环境时直接运行
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"✅ 全部 {len(fns)} 个测试通过")