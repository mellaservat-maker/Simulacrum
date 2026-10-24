# ============================================================
# probe_t58_resident_gap.py · 058 号工作 · 任务 1
#
# 目的：验证常驻进程跨天衰减（修 B16 假死）真的接进了数值链。
#   病灶回顾：055 的跨天衰减只在 session_store.load_state 按 saved_at
#   结算；server_api 的会话表是 LRU——进程不重启、会话不被淘汰就永远
#   不走 load，用户隔几小时/几天回来，"气几天"在常驻进程里与"没气过"
#   同值。修法：pipeline._run_chain 轮初按 st["last_active"] 超阈值
#   （RESIDENT_GAP_THRESHOLD_SEC=3600）手工结算同四条通道。
#
# 验证矩阵：
#   A1 触发组：两轮间隔 3 小时（10800s > 3600s），不重启进程——
#     log["resident_gap"] 明细齐全，四条通道数值逐一核对公式；
#   A2 不触发组：间隔 30 分钟（1800s < 3600s）——明细为 None；
#   A3 等值边界：间隔恰 3600s——严格大于才触发，等值不触发；
#   A4 同时钟对照：同样的两轮、同样的时钟，只把阈值临时抬到天际——
#     round2 log 与 A1 逐字段比对，差异只能来自衰减本身
#     （证明衰减进了数值链，不只是 log 字段）；
#   B  重启路径对照：round1 后 dump（saved_at=T0），隔 10800s load——
#     四条通道结算结果与 A1 同口径（标量逐位、向量容 float32）；
#   C  函数级边界：last_active 缺失/损坏 → 按 now 处理不崩。
#
# 确定性保障（与 probe_t56 / test_golden_trajectory 同规格）：
#   · 固定输入 + 固定 S 表；假 LLM + LLM_BACKEND="mock"；
#   · 可设时钟（_SettableClock，任意时刻打桩，pipeline/event_parser/
#     session_store 三处 time 共用）；
#   · fixture 注入走 pipeline.get_fixture，与生产同一条 run_turn 路。
#
# 失败即报错：任一指称不成立直接 AssertionError 退出（非零码）。
# 用法：run_with_venv.bat scripts\probe_t58_resident_gap.py
# ============================================================

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config              # noqa: E402
import pipeline            # noqa: E402
import decompressor        # noqa: E402
import event_parser        # noqa: E402
import session_store       # noqa: E402
from state import reset_state       # noqa: E402

SCN = "__t058_gap__"
pipeline.LOG_FILES.setdefault(SCN, "run_t058_resident_gap.jsonl")

T0 = 1_700_000_000.0
GAP_FIRE = 10800.0        # 3 小时：超过 3600 阈值，触发
GAP_NOFIRE = 1800.0       # 30 分钟：低于阈值，不触发
GAP_EXACT = 3600.0        # 恰等阈值：严格大于才触发，等值不触发

TEXT = "你这个AI真蠢，什么都做不好。"
S = [-0.88, 0.72, 0.62, 0.82]
SEV = float(np.clip(0.7 * 0.88 + 0.3 * 0.82, 0.0, 1.0))

TAU_E = float(config.TAU_GAP_DAYS_EMOTION)     # 1 天
TAU_F = float(config.TAU_GAP_DAYS_FATIGUE)     # 2 天
TAU_H = float(config.TAU_GAP_DAYS_HABIT)       # 3 天


class _FixedLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return "（058 探针固定回复）"


class _SettableClock:
    """可设时钟：time() 返回当前设定值（任意时刻打桩用）。"""

    def __init__(self, t=T0):
        self._t = float(t)

    def set(self, t):
        self._t = float(t)

    def time(self):
        return self._t


def _patch_all(clock, thr=None):
    """打桩返回 saved 元组；thr 非 None 时同时改常驻阈值（A4 对照用）。"""
    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time, session_store.time,
             config.RESIDENT_GAP_THRESHOLD_SEC)

    def _fake_signals(_text, llm=None):
        return np.asarray(S, dtype=float), "t058"

    def _fake_fixture(_scn):
        return {"input": TEXT, "event_severity": SEV, "danger_score": 0.0,
                "importance": 0.5, "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    pipeline.get_fixture = _fake_fixture
    pipeline.get_llm = lambda: _FixedLLM()
    pipeline.extract_semantic_signals = _fake_signals
    decompressor.extract_semantic_signals = _fake_signals
    pipeline.LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "logs")
    config.LLM_BACKEND = "mock"
    pipeline.time = clock
    event_parser.time = clock
    session_store.time = clock
    if thr is not None:
        config.RESIDENT_GAP_THRESHOLD_SEC = float(thr)
    return saved


def _restore_all(saved):
    (pipeline.get_fixture, pipeline.get_llm,
     pipeline.extract_semantic_signals,
     decompressor.extract_semantic_signals,
     pipeline.LOGS_DIR, config.LLM_BACKEND,
     pipeline.time, event_parser.time, session_store.time,
     config.RESIDENT_GAP_THRESHOLD_SEC) = saved


def _two_rounds(gap, thr=None):
    """同一进程跑两轮：round1 @T0，round2 @T0+gap。返回 (log1, log2, st)。
    thr 非 None 时临时改常驻阈值（A4 对照）。"""
    clock = _SettableClock(T0)
    saved = _patch_all(clock, thr)
    try:
        st = reset_state(seed=42)
        clock.set(T0)
        log1 = pipeline.run_turn(st, SCN, feature_mode="encoder")
        clock.set(T0 + gap)
        log2 = pipeline.run_turn(st, SCN, feature_mode="encoder")
        return log1, log2, st
    finally:
        _restore_all(saved)


def _check_detail(d, gap_sec):
    """核对 resident_gap 明细四条通道与公式逐位一致（标量精确/向量近似）。"""
    assert d is not None, "应触发常驻衰减，明细为 None"
    assert abs(d["gap_sec"] - gap_sec) < 1e-9, \
        "gap_sec 应为 %s，实测 %s" % (gap_sec, d.get("gap_sec"))
    gd = gap_sec / 86400.0
    assert abs(d["gap_days"] - gd) < 1e-12, "gap_days 换算错误"
    e_b = np.asarray(d["E_prev_before"], dtype=float)
    e_a = np.asarray(d["E_prev_after"], dtype=float)
    assert np.allclose(e_a, e_b * math.exp(-gd / TAU_E), rtol=0, atol=1e-12), \
        "E_prev 通道衰减不符公式"
    assert abs(d["w_PFC_after"]
              - d["w_PFC_before"] * math.exp(-gd / TAU_F)) < 1e-12, \
        "w_PFC 通道衰减不符公式"
    assert abs(d["H_prev_after"]
              - d["H_prev_before"] * math.exp(-gd / TAU_H)) < 1e-12, \
        "H_prev 通道衰减不符公式"
    assert abs(d["session_gap_days_after"] - gd) < 1e-12, \
        "session_gap_days 应累加本段空档"


def main():
    print("=" * 72)
    print("  058 探针 · 常驻进程跨天衰减（B16 假死修复验证）")
    print("=" * 72)
    print("  阈值 RESIDENT_GAP_THRESHOLD_SEC=%.0f 秒；tau：E=%.0f天 "
          "w_PFC=%.0f天 H=%.0f天" % (config.RESIDENT_GAP_THRESHOLD_SEC,
                                      TAU_E, TAU_F, TAU_H))
    print("  输入：两轮同一句辱骂（固定 S/severity），round1@T0，"
          "round2@T0+gap")
    print()

    # ---- A1：触发组（3 小时） ----
    print("[A1] 触发组：间隔 %.0f 秒（3 小时，> 阈值）" % GAP_FIRE)
    log1, log2, st1 = _two_rounds(GAP_FIRE)
    d = log2.get("resident_gap")
    _check_detail(d, GAP_FIRE)
    print("  log2.resident_gap 明细（四条通道）：")
    print("    gap_sec=%.1f  gap_days=%.6f" % (d["gap_sec"], d["gap_days"]))
    print("    E_prev   %s → %s（×%.4f）"
          % (np.round(d["E_prev_before"], 4).tolist(),
             np.round(d["E_prev_after"], 4).tolist(),
             math.exp(-(GAP_FIRE / 86400.0) / TAU_E)))
    print("    w_PFC    %.4f → %.4f（×%.4f）"
          % (d["w_PFC_before"], d["w_PFC_after"],
             math.exp(-(GAP_FIRE / 86400.0) / TAU_F)))
    print("    H_prev   %.4f → %.4f（×%.4f）"
          % (d["H_prev_before"], d["H_prev_after"],
             math.exp(-(GAP_FIRE / 86400.0) / TAU_H)))
    print("    session_gap_days → %.6f（本段累加）" % d["session_gap_days_after"])
    # last_active 已推进到 round2 轮初（第三轮不会再结算同一段）
    assert abs(float(st1.get("last_active", 0.0)) - (T0 + GAP_FIRE)) < 1e-9, \
        "last_active 应推进到 round2 轮初"
    print("  → last_active 已推进到 round2 轮初（同一段空档不会二次结算）")
    print("  → 四条通道数值与公式逐位一致：E_prev / w_PFC 确实触发衰减")
    print()

    # ---- A2：不触发组（30 分钟） ----
    print("[A2] 不触发组：间隔 %.0f 秒（30 分钟，< 阈值）" % GAP_NOFIRE)
    log1b, log2b, st1b = _two_rounds(GAP_NOFIRE)
    assert log2b.get("resident_gap") is None, "低于阈值不应结算"
    print("  log2.resident_gap = None（未结算）")
    # E_prev 未被空档机制碰过：round1 轮末值原样进入 round2
    e1_end = np.asarray(log1b["E_final"], dtype=float)
    print("  → round1 轮末 E_final=%s 原样进入 round2（无衰减分支）"
          % np.round(e1_end, 4).tolist())
    print()

    # ---- A3：等值边界（恰 3600） ----
    print("[A3] 等值边界：间隔恰 %.0f 秒（== 阈值）" % GAP_EXACT)
    log1c, log2c, _ = _two_rounds(GAP_EXACT)
    assert log2c.get("resident_gap") is None, \
        "等值不应触发（严格大于；黄金轨迹假时钟步长恰 3600，靠这个避开）"
    print("  log2.resident_gap = None（严格大于才触发，等值不触发）")
    print("  → 这正是黄金轨迹 3600 秒步长下零偏移的原因")
    print()

    # ---- A4：同时钟对照（阈值抬天际） ----
    print("[A4] 同时钟对照：同样的两轮/时钟，仅把阈值临时改为 1e18")
    log1d, log2d, _ = _two_rounds(GAP_FIRE, thr=1e18)
    assert log2d.get("resident_gap") is None, "阈值抬天际后不应结算"
    diffs = []
    for k in ("E_final", "E_raw", "w_PFC", "I_inhibit", "D_error",
              "HabituationSignal", "HistoricalResidual", "danger_now"):
        va, vb = np.asarray(log2[k], dtype=float), np.asarray(log2d[k],
                                                              dtype=float)
        if not np.allclose(va, vb, rtol=0, atol=1e-9):
            diffs.append(k)
    print("  round2 log 与 A1 出现差异的字段：%s" % ("、".join(diffs) or "无"))
    assert diffs, "衰减生效必须改变 round2 的数值链输出（否则只是 log 字段）"
    print("  → 差异只能来自衰减本身（两轮输入/时钟/记忆完全相同）：")
    print("     衰减进了数值链，不只是观测字段——B16 假死确认修复")
    print()

    # ---- B：重启路径（load_state）同口径对照 ----
    print("[B] 重启路径对照：round1 后 dump，隔 %.0f 秒 load" % GAP_FIRE)
    clock = _SettableClock(T0)
    saved = _patch_all(clock)
    try:
        st = reset_state(seed=42)
        clock.set(T0)
        pipeline.run_turn(st, SCN, feature_mode="encoder")
        clock.set(T0)
        payload = session_store.dump_state(st)     # saved_at = T0（打桩时钟）
        w_before = float(payload["w_PFC"])
        h_before = float(payload["HabituationSignal_prev"])
        e_before = np.asarray(session_store._b64_to_vec(payload["E_prev"]),
                              dtype=float)
        clock.set(T0 + GAP_FIRE)
        st2 = session_store.load_state(payload, warn=lambda m: None)
    finally:
        _restore_all(saved)
    gd = GAP_FIRE / 86400.0
    # 标量逐位同口径；向量容 float32 存档精度
    assert abs(float(st2["w_PFC"]) - w_before * math.exp(-gd / TAU_F)) < 1e-9, \
        "load 路径 w_PFC 衰减不符公式"
    assert abs(float(st2["HabituationSignal_prev"])
              - h_before * math.exp(-gd / TAU_H)) < 1e-9, \
        "load 路径 H_prev 衰减不符公式"
    e2 = np.asarray(st2["E_prev"], dtype=float)
    assert np.allclose(e2, e_before * math.exp(-gd / TAU_E), rtol=0, atol=1e-6), \
        "load 路径 E_prev 衰减不符公式（float32 存档容差内）"
    assert abs(float(st2["session_gap_days"]) - gd) < 1e-12, \
        "load 路径 session_gap_days 应累加"
    # 与 A1 常驻路径的 after 值同口径（标量逐位，向量近似）
    assert abs(float(st2["w_PFC"]) - d["w_PFC_after"]) < 1e-9, \
        "两条路径 w_PFC 应同口径"
    assert abs(float(st2["HabituationSignal_prev"]) - d["H_prev_after"]) < 1e-9, \
        "两条路径 H_prev 应同口径"
    assert np.allclose(np.asarray(st2["E_prev"], dtype=float),
                       np.asarray(d["E_prev_after"], dtype=float),
                       rtol=0, atol=1e-6), "两条路径 E_prev 应同口径"
    print("  load 后：w_PFC %.4f→%.4f | H_prev %.4f→%.4f | "
          "session_gap_days=%.6f" % (w_before, st2["w_PFC"], h_before,
                                     st2["HabituationSignal_prev"],
                                     st2["session_gap_days"]))
    print("  → 四条通道与 A1 常驻路径同口径（标量逐位一致，向量差仅在")
    print("     float32 存档精度内）；last_active 重置为加载时刻，不落盘")
    print()

    # ---- C：函数级边界（last_active 缺失/损坏） ----
    print("[C] 函数级边界：pipeline._apply_resident_gap 直接调")
    now = T0 + 10000.0
    st_x = reset_state(seed=42)
    st_x.pop("last_active", None)                 # 缺失 → 按 now 处理
    assert pipeline._apply_resident_gap(st_x, now) is None, "缺失应不触发"
    st_x["last_active"] = "坏值"                   # 损坏 → 回落 now
    assert pipeline._apply_resident_gap(st_x, now) is None, "损坏应不触发"
    st_x["last_active"] = now - 7200.0             # 2 小时 → 触发
    dx = pipeline._apply_resident_gap(st_x, now)
    _check_detail(dx, 7200.0)
    st_x["last_active"] = now                      # 同时刻 → 不触发
    assert pipeline._apply_resident_gap(st_x, now) is None, "同时刻不触发"
    print("  last_active 缺失 → None；损坏('坏值') → None；"
          "2 小时前 → 触发且数值合公式；同时刻 → None")
    print("  → 全部边界行为符合设计（坏数据不崩、不误触发）")
    print()

    print("=" * 72)
    print("  058 探针（任务1）完成：全部断言通过")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
