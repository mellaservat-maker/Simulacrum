# ============================================================
# test_resident_gap.py · 058 号工作 · 任务 1
#
# 覆盖常驻进程跨天衰减（修 B16 假死）：
#   T58-A  apply_gap_decay 函数级：gap<=0 逐位不动 / 四条通道公式
#          / session_gap_days 累加 / 明细字段齐全；
#   T58-B  _apply_resident_gap 函数级：阈值内（含等值）不触发、
#          超阈值触发、last_active 缺失/损坏不崩；
#   T58-C  _run_chain 接线：轮末写回 last_active；log 带 resident_gap
#          明细（超阈值）或 None（阈值内）；
#   T58-D  load 后不双重结算：load 按 saved_at 结算一次并把
#          last_active 对齐到加载时刻，紧随其后的同刻轮次不再结算。
#
# 端到端用例全部断网：假 LLM + 固定 S 表 + LLM_BACKEND="mock"
#   （与 test_golden_trajectory / test_relationship 同规格）；
#   时钟用可设假时钟（pipeline.time / session_store.time 打桩）。
# ============================================================

import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import session_store   # noqa: E402
from state import reset_state       # noqa: E402

pipeline.LOG_FILES.setdefault("__t058_gap__", "run_t058_gap_test.jsonl")

T0 = 1_700_000_000.0
INSULT = "你这个AI真蠢，什么都做不好。"
INSULT_S = [-0.88, 0.72, 0.62, 0.82]


class _FixedLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return "（测试固定回复）"


class _SettableClock:
    """可设假时钟：time() 返回当前设定值。"""

    def __init__(self, t=T0):
        self._t = float(t)

    def set(self, t):
        self._t = float(t)

    def time(self):
        return self._t


def _tau(name):
    return float(getattr(config, name, 0.0) or 0.0)


# ---- T58-A：apply_gap_decay 函数级 ----

def test_apply_gap_decay_zero_gap_bit_identical():
    """gap<=0：三条衰减通道不进 if 分支，状态逐位不动。"""
    st = reset_state(seed=42)
    st["E_prev"] = np.array([-0.4, 0.2, -0.1, 0.15])
    st["w_PFC"] = 0.37
    st["HabituationSignal_prev"] = 0.42
    st["session_gap_days"] = 1.5
    snap = {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in st.items()}
    d = session_store.apply_gap_decay(st, 0.0)
    d2 = session_store.apply_gap_decay(st, -3.0)
    for k, v in snap.items():
        cur = st[k]
        if isinstance(v, np.ndarray):
            assert np.array_equal(cur, v), k
        else:
            assert cur == v, k
    assert d["gap_days"] == 0.0 and d2["gap_days"] == 0.0


def test_apply_gap_decay_four_channels_formula():
    """四条通道按公式结算；明细字段齐全。"""
    st = reset_state(seed=42)
    st["E_prev"] = np.array([-0.4, 0.2, -0.1, 0.15])
    st["w_PFC"] = 0.5
    st["HabituationSignal_prev"] = 0.6
    st["session_gap_days"] = 0.0
    gap_days = 0.25
    d = session_store.apply_gap_decay(st, gap_days)
    e_b = np.array([-0.4, 0.2, -0.1, 0.15])
    assert np.allclose(st["E_prev"], e_b * np.exp(-gap_days / _tau("TAU_GAP_DAYS_EMOTION")))
    assert st["w_PFC"] == pytest.approx(0.5 * np.exp(-gap_days / _tau("TAU_GAP_DAYS_FATIGUE")))
    assert st["HabituationSignal_prev"] == pytest.approx(
        0.6 * np.exp(-gap_days / _tau("TAU_GAP_DAYS_HABIT")))
    assert st["session_gap_days"] == pytest.approx(0.25)
    # 明细字段齐全（log 与探针依赖这套键）
    for k in ("gap_days", "E_prev_before", "E_prev_after", "w_PFC_before",
              "w_PFC_after", "H_prev_before", "H_prev_after",
              "session_gap_days_after"):
        assert k in d, k


def test_apply_gap_decay_session_gap_accumulates():
    """session_gap_days 用 +=：同一会话多次空档累加。"""
    st = reset_state(seed=42)
    session_store.apply_gap_decay(st, 0.5)
    session_store.apply_gap_decay(st, 0.25)
    assert st["session_gap_days"] == pytest.approx(0.75)


# ---- T58-B：_apply_resident_gap 函数级 ----

def test_resident_gap_inside_threshold_no_trigger():
    """低于阈值（含等值）：不结算，返回 None，状态不动。"""
    st = reset_state(seed=42)
    now = T0 + 10000.0
    st["E_prev"] = np.array([-0.4, 0.2, -0.1, 0.15])
    st["w_PFC"] = 0.5
    st["last_active"] = now - (config.RESIDENT_GAP_THRESHOLD_SEC - 1.0)
    snap_e, snap_w = st["E_prev"].copy(), st["w_PFC"]
    assert pipeline._apply_resident_gap(st, now) is None
    assert np.array_equal(st["E_prev"], snap_e)
    assert st["w_PFC"] == snap_w

    # 等值：严格大于才触发（黄金轨迹假时钟步长恰 3600 秒，靠这个避开）
    st["last_active"] = now - config.RESIDENT_GAP_THRESHOLD_SEC
    assert pipeline._apply_resident_gap(st, now) is None
    assert np.array_equal(st["E_prev"], snap_e)


def test_resident_gap_over_threshold_triggers():
    """超过阈值：结算并返回带 gap_sec 的明细。"""
    st = reset_state(seed=42)
    now = T0 + 10000.0
    st["E_prev"] = np.array([-0.4, 0.2, -0.1, 0.15])
    st["w_PFC"] = 0.5
    st["HabituationSignal_prev"] = 0.6
    gap_sec = 7200.0
    st["last_active"] = now - gap_sec
    d = pipeline._apply_resident_gap(st, now)
    assert d is not None and d["gap_sec"] == pytest.approx(gap_sec)
    gd = gap_sec / 86400.0
    assert np.allclose(st["E_prev"], np.array([-0.4, 0.2, -0.1, 0.15])
                       * np.exp(-gd / _tau("TAU_GAP_DAYS_EMOTION")))
    assert st["w_PFC"] == pytest.approx(
        0.5 * np.exp(-gd / _tau("TAU_GAP_DAYS_FATIGUE")))


def test_resident_gap_bad_last_active_no_crash():
    """last_active 缺失/非数字：按 now 处理（gap=0），不崩不触发。"""
    st = reset_state(seed=42)
    now = T0 + 10000.0
    st.pop("last_active", None)
    assert pipeline._apply_resident_gap(st, now) is None
    st["last_active"] = "坏值"
    assert pipeline._apply_resident_gap(st, now) is None


# ---- T58-C：_run_chain 接线（端到端） ----

def _patch_turn(monkeypatch, tmp_path, clock):
    def _fake_signals(_text, llm=None):
        return np.asarray(INSULT_S, dtype=float), "test"

    def _fake_fixture(_scn):
        return {"input": INSULT, "event_severity": 0.86, "danger_score": 0.0,
                "importance": 0.5, "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    monkeypatch.setattr(pipeline, "get_llm", lambda: _FixedLLM())
    monkeypatch.setattr(pipeline, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(decompressor, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(pipeline, "get_fixture", _fake_fixture)
    monkeypatch.setattr(pipeline, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")
    monkeypatch.setattr(pipeline, "time", clock)


def test_run_chain_writes_last_active_and_logs_gap(monkeypatch, tmp_path):
    """轮末写回 last_active（= 轮初 now）；间隔 3 小时的第二轮 log 带
    resident_gap 明细，四条通道数值合公式。"""
    clock = _SettableClock(T0)
    _patch_turn(monkeypatch, tmp_path, clock)
    st = reset_state(seed=42)
    log1 = pipeline.run_turn(st, "__t058_gap__", feature_mode="encoder")
    assert st["last_active"] == pytest.approx(T0)      # 轮末推进到轮初
    assert log1.get("resident_gap") is None            # 首轮无更早 last_active

    clock.set(T0 + 10800.0)
    log2 = pipeline.run_turn(st, "__t058_gap__", feature_mode="encoder")
    d = log2.get("resident_gap")
    assert d is not None and d["gap_sec"] == pytest.approx(10800.0)
    gd = 10800.0 / 86400.0
    e_b = np.asarray(log1["E_final"], dtype=float)
    assert np.allclose(np.asarray(d["E_prev_before"], dtype=float), e_b)
    assert np.allclose(np.asarray(d["E_prev_after"], dtype=float),
                       e_b * np.exp(-gd / _tau("TAU_GAP_DAYS_EMOTION")))
    assert d["w_PFC_after"] == pytest.approx(
        d["w_PFC_before"] * np.exp(-gd / _tau("TAU_GAP_DAYS_FATIGUE")))
    assert st["last_active"] == pytest.approx(T0 + 10800.0)


def test_run_chain_inside_threshold_logs_none(monkeypatch, tmp_path):
    """间隔 30 分钟（< 阈值）：log 的 resident_gap 为 None。"""
    clock = _SettableClock(T0)
    _patch_turn(monkeypatch, tmp_path, clock)
    st = reset_state(seed=42)
    pipeline.run_turn(st, "__t058_gap__", feature_mode="encoder")
    clock.set(T0 + 1800.0)
    log2 = pipeline.run_turn(st, "__t058_gap__", feature_mode="encoder")
    assert log2.get("resident_gap") is None


# ---- T58-D：load 后不双重结算 ----

def test_load_then_resident_no_double_settlement(monkeypatch, tmp_path):
    """load 按 saved_at 结算一次并把 last_active 对齐到加载时刻；
    紧随其后的同刻轮次不再结算同一段空档（log 为 None）。"""
    clock = _SettableClock(T0)
    _patch_turn(monkeypatch, tmp_path, clock)
    monkeypatch.setattr(session_store, "time", clock)

    st = reset_state(seed=42)
    pipeline.run_turn(st, "__t058_gap__", feature_mode="encoder")
    clock.set(T0)                                  # dump 的 saved_at = T0
    payload = session_store.dump_state(st)
    w_dumped = float(payload["w_PFC"])

    clock.set(T0 + 10800.0)                        # 隔 3 小时加载
    st2 = session_store.load_state(payload, warn=lambda m: None)
    gd = 10800.0 / 86400.0
    # load 结算一次
    assert st2["w_PFC"] == pytest.approx(
        w_dumped * np.exp(-gd / _tau("TAU_GAP_DAYS_FATIGUE")))
    # last_active 对齐到加载时刻（reset_state 的初值被覆盖）
    assert st2["last_active"] == pytest.approx(T0 + 10800.0)

    # 同刻轮次：gap=0，不二次结算
    log = pipeline.run_turn(st2, "__t058_gap__", feature_mode="encoder")
    assert log.get("resident_gap") is None
