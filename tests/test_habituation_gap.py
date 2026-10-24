# ============================================================
# test_habituation_gap.py · 055 号工作 · 任务 2 + 任务 3
#
# 任务 2：HabituationSignal 从"计数器"改为"同类事件重复度"
#   T055-B1 相似度不足的历史不计入（不同话题 H 不累积）
#   T055-B2 高度相似的历史才累积，且相似度越高贡献越大
#   T055-B3 旧式计数器在同样输入下会给出接近 1 的 H（对照，证明修的是真 bug）
#
# 任务 3：跨天时间感
#   T055-C1 加载会话时按 saved_at 算出 gap_days
#   T055-C2 空档期情绪惯性消退（3 天后 E_prev 明显回落）
#   T055-C3 空档期疲劳恢复、习惯化消退
#   T055-C4 无 saved_at / 刚保存 → gap≈0，状态不动（不误伤）
# ============================================================

import math
import os
import sys
import time
import types

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config                       # noqa: E402
import session_store                # noqa: E402
from hippocampal_evaluator import _habituation   # noqa: E402
from state import reset_state       # noqa: E402

NOW = 1_700_000_000.0


def _mem(ts=NOW):
    return types.SimpleNamespace(timestamp=ts, event_embedding=np.zeros(8))


# ---- 任务 2：习惯化 ----

def test_dissimilar_events_do_not_accumulate():
    """全部相似度低于阈值 → H = 0（不同话题不累积）。"""
    pairs = [(_mem(), 0.55), (_mem(), 0.60), (_mem(), 0.68)]
    assert _habituation(pairs, NOW) == 0.0


def test_similar_events_accumulate():
    """相似度接近 1 的同类事件才累积，越多越高。"""
    one = _habituation([(_mem(), 0.95)], NOW)
    three = _habituation([(_mem(), 0.95)] * 3, NOW)
    assert one > 0.0
    assert three > one
    # 3 条 sim≈1 的权重 ≈3 → H ≈ 3/4
    assert abs(three - 0.75) < 0.05


def test_high_sim_contributes_more_than_borderline():
    """刚过阈值的贡献应远小于完全相同的。"""
    low = _habituation([(_mem(), 0.72)], NOW)
    high = _habituation([(_mem(), 0.99)], NOW)
    assert high > low * 3


def test_old_counter_would_have_saturated():
    """对照：旧式 H = n/(n+1) 在 3 条时就到 0.75，与话题无关。

    这里用同样 3 条"不相似"的历史做对照——旧式给 0.75，新式给 0，
    这就是"聊到第 3 轮就麻木"的数值来源。
    """
    n = 3
    old_h = n / (n + 1.0)
    new_h = _habituation([(_mem(), 0.55)] * n, NOW)
    assert abs(old_h - 0.75) < 1e-9
    assert new_h == 0.0
    assert old_h - new_h > 0.7


# ---- 任务 3：跨天 ----

def _dump_with_age(days):
    st = reset_state(seed=42)
    st["E_prev"] = np.array([-0.8, 0.6, -0.3, 0.4])
    st["w_PFC"] = 1.2
    st["HabituationSignal_prev"] = 0.9
    data = session_store.dump_state(st)
    data["saved_at"] = time.time() - days * 86400.0
    return data


def test_gap_days_computed_from_saved_at():
    data = _dump_with_age(3.0)
    st = session_store.load_state(data, warn=lambda *_: None)
    assert abs(st["session_gap_days"] - 3.0) < 0.05


def test_emotion_decays_during_gap():
    data = _dump_with_age(3.0)
    st = session_store.load_state(data, warn=lambda *_: None)
    expect = 0.8 * math.exp(-3.0 / config.TAU_GAP_DAYS_EMOTION)
    assert abs(abs(float(st["E_prev"][0])) - expect) < 0.02
    # 三天后应该已经消掉九成以上（tau=1 天）
    assert abs(float(st["E_prev"][0])) < 0.8 * 0.1


def test_fatigue_and_habituation_recover_during_gap():
    data = _dump_with_age(3.0)
    st = session_store.load_state(data, warn=lambda *_: None)
    assert st["w_PFC"] < 1.2 * 0.3            # tau=2 天，3 天剩 ~22%
    assert st["HabituationSignal_prev"] < 0.9 * 0.4   # tau=3 天，剩 ~37%


def test_no_gap_keeps_state():
    """刚保存的会话：gap≈0，状态不应被改动。"""
    st0 = reset_state(seed=42)
    st0["E_prev"] = np.array([-0.5, 0.2, 0.0, 0.1])
    st0["w_PFC"] = 0.7
    data = session_store.dump_state(st0)
    data["saved_at"] = time.time()
    st = session_store.load_state(data, warn=lambda *_: None)
    assert st["session_gap_days"] < 1e-4
    assert abs(float(st["E_prev"][0]) - (-0.5)) < 1e-6
    assert abs(float(st["w_PFC"]) - 0.7) < 1e-6


def test_missing_saved_at_is_safe():
    """旧存档没有 saved_at → gap=0，不因缺字段而衰减。"""
    data = _dump_with_age(0.0)
    data.pop("saved_at", None)
    st = session_store.load_state(data, warn=lambda *_: None)
    assert st["session_gap_days"] == 0.0
    assert abs(float(st["E_prev"][0]) - (-0.8)) < 1e-6
