"""
Omni-Flow 全双工插件单测：序列化 / TAIL / loss mask / reward / 回放 rollout

覆盖 verl_omni_ext/features/fullduplex/ 的三个无环境依赖模块：
    omniflow_dataset.py — 时间索引 → g_k 序列化（论文 §3.2/§3.4）
    rewards.py          — 四类 reward（论文 §5.4 + RFC #3745）
    duplex_rollout.py   — 回放式 episode + GRPO group（ReplayDuplexClient）

全部纯 CPU，不需要 verl / vllm-omni / torch。
运行：python3 tests/test_omniflow.py
"""

import importlib.util
import os
import sys

# 直接从文件加载（与 test_patchkit.py 相同模式：绕过包 __init__ 的 verl 依赖）
_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ds = _load("omniflow_dataset", "verl_omni_ext/features/fullduplex/omniflow_dataset.py")
rw = _load("omniflow_rewards", "verl_omni_ext/features/fullduplex/rewards.py")
# duplex_rollout 引用 verl_omni_ext 包路径——手动桥接
sys.modules.setdefault("verl_omni_ext", type(sys)("verl_omni_ext"))
sys.modules.setdefault("verl_omni_ext.features", type(sys)("verl_omni_ext.features"))
sys.modules.setdefault("verl_omni_ext.features.fullduplex", type(sys)("verl_omni_ext.features.fullduplex"))
sys.modules["verl_omni_ext.features.fullduplex.omniflow_dataset"] = ds
dr = _load("omniflow_duplex_rollout", "verl_omni_ext/features/fullduplex/duplex_rollout.py")


# ============================================================================
# omniflow_dataset：序列化
# ============================================================================

def test_chunking_and_grouping():
    """1.0s episode @0.2s → 5 块；v/a 按 t_start 归块；LS 控制位"""
    demo = ds.make_episode_row(
        episode_id="t1",
        visual=[
            {"t_start": 0.0, "t_end": 0.2, "frames": ["f0"]},
            {"t_start": 0.4, "t_end": 0.6, "frames": ["f1"]},
        ],
        audio=[{"t_start": 0.0, "t_end": 1.0, "samples": [0] * 16000}],
        text=[{"t_start": 0.4, "t_end": 0.6, "tokens": ["你", "好"]}],
        speech=[{"t_start": 0.4, "t_end": 0.8, "tokens": ["s1", "s2"]}],
    )
    chunks = ds.serialize_episode(demo)
    assert len(chunks) == 5, f"1.0s@0.2s 应 5 块，得到 {len(chunks)}"
    # 视觉归块：f0→chunk0, f1→chunk2
    assert chunks[0].visual_frames == ["f0"]
    assert chunks[2].visual_frames == ["f1"]
    # 音频整段 t_start=0 归 chunk0（流式切分由 processor 完成）
    assert len(chunks[0].audio_chunks) == 1
    # LS 控制位：chunk2 有文本 → speak；其余 listen
    controls = [c.control for c in chunks]
    assert controls == ["<|listen|>", "<|listen|>", "<|speak|>",
                        "<|listen|>", "<|listen|>"], controls
    # 文本按 start time（0.4s）归 chunk2
    assert chunks[2].text_tokens == ["你", "好"]


def test_per_token_timestamps():
    """逐 token 时间戳：token_times 优先于整段时间戳"""
    demo = ds.make_episode_row(
        episode_id="t2",
        text=[{"t_start": 0.0, "t_end": 0.6, "tokens": ["a", "b", "c"],
               "token_times": [0.05, 0.25, 0.45]}],
    )
    chunks = ds.serialize_episode(demo)
    assert chunks[0].text_tokens == ["a"]
    assert chunks[1].text_tokens == ["b"]
    assert chunks[2].text_tokens == ["c"]


def test_tail_lookahead_defers_speech():
    """TAIL：块尾文本的 speech 推迟到下一块（有界 look-ahead）"""
    # 0.4s 处一段 5 token 文本，speech 同段——块尾部分应推迟
    demo = ds.make_episode_row(
        episode_id="t3",
        text=[{"t_start": 0.4, "t_end": 0.6,
               "tokens": ["一", "二", "三", "四", "五"]}],
        speech=[{"t_start": 0.4, "t_end": 0.8, "tokens": ["s1", "s2", "s3", "s4", "s5"]}],
    )
    chunks = ds.serialize_episode(demo, tail_lookahead=2)
    # 文本全在 chunk2
    assert chunks[2].text_tokens == ["一", "二", "三", "四", "五"]
    # speech 段起始 t=0.4 恰在块首（rel≈0）→ 不推迟，归 chunk2
    assert chunks[2].speech_tokens == ["s1", "s2", "s3", "s4", "s5"]
    # 块尾段（rel 接近 1）→ 推迟到 chunk3
    demo2 = ds.make_episode_row(
        episode_id="t3b",
        text=[{"t_start": 0.4, "t_end": 0.6,
               "tokens": ["一", "二", "三", "四", "五"]}],
        speech=[{"t_start": 0.59, "t_end": 0.8, "tokens": ["s1"]}],
    )
    chunks2 = ds.serialize_episode(demo2, tail_lookahead=2)
    assert chunks2[3].speech_tokens == ["s1"], "块尾 speech 应推迟到 chunk3"


def test_loss_mask():
    """loss mask 长度 = 控制位 + text + speech；全部训练位"""
    demo = ds.make_episode_row(
        episode_id="t4",
        text=[{"t_start": 0.2, "t_end": 0.4, "tokens": ["x", "y"]}],
        speech=[{"t_start": 0.2, "t_end": 0.4, "tokens": ["s1"]}],
    )
    chunks = ds.serialize_episode(demo)
    mask = ds.build_loss_mask(chunks)
    expect = sum(1 + len(c.text_tokens) + len(c.speech_tokens) for c in chunks)
    assert len(mask) == expect
    assert all(mask)


def test_collate_fn():
    """omniflow_collate_fn：batch 结构 + GRPO group 默认同 episode"""
    rows = [ds.make_episode_row(episode_id=f"e{i}",
                                text=[{"t_start": 0.2, "t_end": 0.4, "tokens": ["a"]}])
            for i in range(3)]
    out = ds.omniflow_collate_fn(rows)
    assert out["batch_size"] == 3
    assert out["omniflow"] is True
    assert out["chunk_size_s"] == 0.2
    assert len(out["episodes"]) == 3
    for ep in out["episodes"]:
        assert ep["group_id"] == ep["episode_id"]  # GRPO group 默认


# ============================================================================
# rewards：四类信号
# ============================================================================

GT = [{"k": 0, "control": "<|listen|>"},
      {"k": 1, "control": "<|speak|>"},
      {"k": 2, "control": "<|speak|>"},
      {"k": 3, "control": "<|listen|>"}]


def test_decision_reward_perfect():
    traj = [dict(c) for c in GT]
    r, m = rw.decision_reward(traj, GT)
    assert r == 1.0 and m.accuracy == 1.0
    assert (m.true_speak, m.true_listen) == (2, 2)


def test_decision_reward_missed_and_false():
    # 漏说（k=1 listen 但 GT speak）+ 抢话（k=3 speak 但 GT listen）
    traj = [{"k": 0, "control": "<|listen|>"},
            {"k": 1, "control": "<|listen|>"},
            {"k": 2, "control": "<|speak|>"},
            {"k": 3, "control": "<|speak|>"}]
    r, m = rw.decision_reward(traj, GT)
    assert m.missed_speak == 1 and m.false_speak == 1
    # (true_speak=1 + true_listen=1 - 1.5×1 抢话 - 1 漏说) / 4 = -0.125
    assert abs(r - (-0.125)) < 1e-9


def test_timeliness_reward():
    """text/speech 1:1 → 1.0；2:1 积压 → 低分"""
    good = [{"k": 0, "control": "<|speak|>",
             "text_tokens": ["a", "b"], "speech_tokens": ["s1", "s2"]}]
    bad = [{"k": 0, "control": "<|speak|>",
            "text_tokens": ["a", "b", "c", "d"], "speech_tokens": ["s1", "s2"]}]
    assert rw.timeliness_reward(good) == 1.0
    assert rw.timeliness_reward(bad) < 0.5
    assert rw.timeliness_reward([{"k": 0, "control": "<|listen|>"}]) == 0.0


def test_bargein_latency():
    """barge-in 后立即停（同块转 listen）→ 0 罚；隔 3 块才停 → 重罚"""
    fast = [{"k": 0, "control": "<|speak|>", "events": [{"type": "barge_in"}]},
            {"k": 1, "control": "<|listen|>"}]
    slow = [{"k": 0, "control": "<|speak|>", "events": [{"type": "barge_in"}]},
            {"k": 1, "control": "<|speak|>"},
            {"k": 2, "control": "<|speak|>"},
            {"k": 3, "control": "<|listen|>"}]
    assert rw.bargein_latency_penalty(fast) == 0.0
    assert rw.bargein_latency_penalty(slow) < -0.5


def test_interruption_reward():
    """打断后宽限期内转 listen → 奖；继续旧 epoch 内容 → 罚"""
    good = [{"k": 0, "control": "<|speak|>", "events": [{"type": "barge_in", "epoch": 1}]},
            {"k": 1, "control": "<|listen|>", "epoch": 1}]
    bad = [{"k": 0, "control": "<|speak|>", "events": [{"type": "barge_in", "epoch": 1}]},
           {"k": 1, "control": "<|speak|>", "epoch": 1, "text_tokens": ["旧"]}]
    assert rw.interruption_reward(good) == 1.0
    assert rw.interruption_reward(bad) == -1.0


def test_content_score_and_smooth_length():
    # 内容：关键词命中 + speak 位有内容
    s = rw.omniflow_content_score("", "你好世界", "你好",
                                  {"chunks": [{"control": "<|speak|>", "text_tokens": ["你"]}]}
                                  )
    assert 0.9 < s <= 1.0
    # Kimi-K1.5 smooth length：答对短者奖（中位恰为参考长度 → 0 分）、答错不奖短
    ls = rw._smooth_length_reward([10, 20, 30], [True, True, False])
    assert ls[0] > ls[1] >= 0 and ls[2] <= 0


def test_trajectory_reward_composition():
    traj = [{"k": 0, "control": "<|listen|>"},
            {"k": 1, "control": "<|speak|>", "text_tokens": ["你", "好"],
             "speech_tokens": ["s1", "s2"]},
            {"k": 2, "control": "<|speak|>", "text_tokens": ["吗"], "speech_tokens": ["s3"]},
            {"k": 3, "control": "<|listen|>"}]
    out = rw.omniflow_trajectory_reward(traj, GT, "你好吗", "你好吗")
    assert set(out) == {"total", "decision", "timeliness", "interruption", "content"}
    assert out["decision"] == 1.0
    assert -1.0 <= out["total"] <= 1.0


# ============================================================================
# duplex_rollout：回放 episode + GRPO group
# ============================================================================

def test_replay_episode_trajectory():
    """回放 episode：轨迹结构 + 事件注入 + close 契约"""
    demo = ds.make_episode_row(
        episode_id="ep_r",
        audio=[{"t_start": 0.0, "t_end": 1.0, "samples": [0] * 16000}],
        text=[{"t_start": 0.4, "t_end": 0.6, "tokens": ["你", "好"]}],
    )
    env = ds.serialize_episode(demo)
    client = dr.ReplayDuplexClient(ground_truth_chunks=env, barge_at=3)
    traj = dr.run_replay_episode(client, env, "ep_r", weight_version="step_5",
                                 barge_in_ks=[3])
    assert len(traj.chunks) == len(env)
    assert traj.weight_version == "step_5"
    # barge-in 事件落在 k=3
    assert any(e.get("type") == "barge_in" for e in traj.chunks[3]["events"])
    # 会话已关闭（KV lease 释放——权重同步前提）
    assert all(s["closed"] for s in client._sessions.values())
    # 轨迹与 GT 一致时 decision reward 应满分
    out = rw.omniflow_trajectory_reward(traj.to_reward_input(),
                                        [{"k": c.k, "control": c.control} for c in env])
    assert out["decision"] == 1.0


def test_collect_group():
    """GRPO group：同 episode N 条轨迹，group_id 统一"""
    demo = ds.make_episode_row(episode_id="ep_g",
                               text=[{"t_start": 0.2, "t_end": 0.4, "tokens": ["x"]}])
    env = ds.serialize_episode(demo)
    client = dr.ReplayDuplexClient(ground_truth_chunks=env)
    group = dr.collect_group(client, env, "ep_g", group_size=4,
                             weight_version="step_9")
    assert len(group) == 4
    assert all(t.group_id == "grp_ep_g" for t in group)
    assert all(t.weight_version == "step_9" for t in group)
    # group 内轨迹结构一致（同环境回放）
    assert all(len(t.chunks) == len(env) for t in group)


def test_worker_cls_orchestration():
    """worker_cls 协议：generate_sequences 编排 batch → group 轨迹"""
    worker_cls = dr.build_worker_cls()
    demo = ds.make_episode_row(episode_id="ep_w",
                               text=[{"t_start": 0.2, "t_end": 0.4, "tokens": ["x"]}])
    chunks = ds.serialize_episode(demo)
    batch = {"episodes": [{"episode_id": "ep_w", "chunks": chunks,
                           "barge_in_ks": [1]}],
             "weight_version": "step_12"}

    class _FakeCfg:
        actor_rollout_ref = None

    worker = worker_cls(_FakeCfg())
    client = dr.ReplayDuplexClient(ground_truth_chunks=chunks)
    trajs = worker.generate_sequences(batch, client=client)
    assert len(trajs) == 4  # 默认 group_size
    assert all(t.weight_version == "step_12" for t in trajs)


def test_infer_barge_in_ks():
    """speak→listen 转换块推导 barge-in 时刻"""
    from verl_omni_ext.features.fullduplex.omniflow_dataset import (  # noqa: F401
        CONTROL_LISTEN as _L, CONTROL_SPEAK as _S,
    )

    class _C:
        def __init__(self, k, control):
            self.k, self.control = k, control

    env = [_C(0, _L), _C(1, _S), _C(2, _S), _C(3, _L), _C(4, _S), _C(5, _L)]
    assert dr._infer_barge_in_ks(env) == [3, 5]


# ============================================================================
# runner
# ============================================================================

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"✅ 全部 {len(fns)} 个测试通过")