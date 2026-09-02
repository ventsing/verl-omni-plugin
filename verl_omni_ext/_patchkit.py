"""
L2 monkey patch 公共基建

所有 monkey patch 必须走本模块提供的装饰器，强制满足 4 条规范：
  (a) 打补丁前断言目标存在且签名符合预期
  (b) 记录被 patch 版本的指纹（transformers 版本区间等）
  (c) 有对应的 CPU 测试验证补丁行为
  (d) 返回 bool，启动脚本自检时断言为 True

这 4 条是应对 monkey patch 最大的风险——"静默失效"的唯一手段：
  上游把被 patch 的函数改名/换签名/删掉时，补丁要么前置断言响，
  要么 CPU 测试红。总之不能让它"打上去了但修的不是原来那个 bug"。

适用对象（L2 的准入规则）：
  只能打第三方代码——transformers / vllm / checkpoint remote code / 模型实例。
  打 verl-omni 自己的东西说明该用 L3 gate patch 或该给上游提扩展点。
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 全局注册表：记录所有已打的补丁，供启动脚本自检
_PATCH_REGISTRY: dict[str, "PatchRecord"] = {}


class PatchRecord:
    """一条补丁的注册记录"""

    __slots__ = ("name", "target", "fingerprint", "applied")

    def __init__(self, name: str, target: str, fingerprint: str, applied: bool):
        self.name = name          # 补丁名（如 "minicpmo_auto_register_guard"）
        self.target = target      # 被打的对象（如 "transformers.AutoImageProcessor"）
        self.fingerprint = fingerprint  # 版本指纹（如 "transformers>=4.46,<4.50"）
        self.applied = applied    # 是否真的打上了

    def __repr__(self):
        status = "✓ applied" if self.applied else "✗ skipped"
        return f"<Patch {self.name} -> {self.target} [{self.fingerprint}] {status}>"


def assert_target(
    module_path: str,
    attr_name: str,
    expected_signature: str | None = None,
) -> Any:
    """前置断言：被 patch 的目标必须存在，签名必须匹配。

    这是防止"静默失效"的第一道防线。如果上游改了名/删了函数，
    这里会直接抛异常，而不是让补丁默默打到一个不存在的东西上。

    Args:
        module_path: 如 "transformers"
        attr_name: 如 "AutoImageProcessor"
        expected_signature: 可选，期望的函数签名片段（如 "register"）

    Returns:
        被断言通过的对象

    Raises:
        AssertionError: 目标不存在或签名不匹配
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise AssertionError(
            f"Patch target module '{module_path}' not importable: {e}"
        ) from e

    target = getattr(module, attr_name, None)
    if target is None:
        raise AssertionError(
            f"Patch target '{module_path}.{attr_name}' does not exist — "
            f"upstream may have renamed or removed it. "
            f"Do NOT proceed: patching a non-existent target is a silent no-op."
        )

    if expected_signature is not None:
        # 检查是否有期望的方法/属性
        if not hasattr(target, expected_signature):
            raise AssertionError(
                f"'{module_path}.{attr_name}' has no attribute '{expected_signature}' — "
                f"upstream API changed. Patch needs updating."
            )

    return target


def idempotent_patch(
    name: str,
    target_module: str,
    target_attr: str,
    fingerprint: str = "",
    expected_signature: str | None = None,
    flag_attr: str | None = None,
):
    """装饰器：把一个函数变成幂等的 monkey patch。

    被装饰的函数接收 original（被替换的原函数/原属性）作为第一个参数，
    返回替换后的函数。

    保证：
      - 幂等：重复调用无害（用 flag_attr 标志位）
      - 前置断言：目标存在且签名匹配
      - 版本指纹：记录在 PatchRecord 里
      - 返回 bool：True=打上了，False=跳过了（目标已不存在是正常的降级）

    Args:
        name: 补丁名（唯一标识）
        target_module: 如 "transformers"
        target_attr: 如 "AutoImageProcessor"
        fingerprint: 版本区间，如 "transformers>=4.46,<4.50"
        expected_signature: 期望的签名片段（用于前置断言）
        flag_attr: 幂等标志位属性名（默认 "_verl_omni_ext_patched_" + name）

    Usage:
        @idempotent_patch(
            name="minicpmo_auto_register_guard",
            target_module="transformers",
            target_attr="AutoImageProcessor",
            fingerprint="transformers>=4.46",
            expected_signature="register",
        )
        def patch_auto_register_guard(original):
            def guarded_register(*args, **kwargs):
                ...
            return guarded_register
    """
    if flag_attr is None:
        flag_attr = f"_verl_omni_ext_patched_{name}"

    def decorator(patch_fn: Callable) -> Callable:
        @functools.wraps(patch_fn)
        def wrapper(*args, **kwargs) -> bool:
            # 幂等检查
            try:
                module = importlib.import_module(target_module)
            except ImportError:
                logger.warning("Patch %r: module %s not importable, skipping", name, target_module)
                _PATCH_REGISTRY[name] = PatchRecord(name, f"{target_module}.{target_attr}", fingerprint, False)
                return False

            target = getattr(module, target_attr, None)
            if target is None:
                logger.warning("Patch %r: target %s.%s not found, skipping", name, target_module, target_attr)
                _PATCH_REGISTRY[name] = PatchRecord(name, f"{target_module}.{target_attr}", fingerprint, False)
                return False

            # 幂等：已打过就跳过
            if getattr(target, flag_attr, False):
                logger.debug("Patch %r: already applied, skipping", name)
                return True

            # 前置断言
            if expected_signature is not None:
                if not hasattr(target, expected_signature):
                    logger.error(
                        "Patch %r: %s.%s missing expected signature %r — "
                        "upstream API changed, NOT patching to avoid silent failure",
                        name, target_module, target_attr, expected_signature,
                    )
                    _PATCH_REGISTRY[name] = PatchRecord(name, f"{target_module}.{target_attr}", fingerprint, False)
                    return False

            # 执行补丁
            original = getattr(target, expected_signature, target) if expected_signature else target
            try:
                replacement = patch_fn(original)
                if expected_signature:
                    setattr(target, expected_signature, replacement)
                else:
                    setattr(module, target_attr, replacement)
                setattr(target, flag_attr, True)
                logger.info("Patch %r applied on %s.%s [%s]", name, target_module, target_attr, fingerprint)
                _PATCH_REGISTRY[name] = PatchRecord(name, f"{target_module}.{target_attr}", fingerprint, True)
                return True
            except Exception as e:
                logger.error("Patch %r failed: %s", name, e)
                _PATCH_REGISTRY[name] = PatchRecord(name, f"{target_module}.{target_attr}", fingerprint, False)
                return False

        return wrapper

    return decorator


def self_check() -> dict[str, bool]:
    """启动脚本自检：断言所有 L2 补丁都已打上。

    在 examples/*/run_*.sh 的前置自检里调用：

        python -c "from verl_omni_ext._patchkit import self_check; \\
                    results = self_check(); \\
                    assert all(results.values()), results"

    如果有补丁没打上（目标不存在/签名变了），这里会响，
    而不是等到第一个 micro-batch 才炸。
    """
    return {name: rec.applied for name, rec in _PATCH_REGISTRY.items()}


def patch_summary() -> str:
    """人类可读的补丁状态摘要"""
    if not _PATCH_REGISTRY:
        return "No patches registered."
    lines = ["L2 monkey patch status:"]
    for name, rec in _PATCH_REGISTRY.items():
        status = "✓" if rec.applied else "✗"
        lines.append(f"  {status} {name} -> {rec.target} [{rec.fingerprint}]")
    return "\n".join(lines)
