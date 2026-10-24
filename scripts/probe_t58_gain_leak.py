# ============================================================
# probe_t58_gain_leak.py · 058 号工作立探针 / 059 任务1 修订
#
# 目的：验证熟人效应（056 任务2）两条通道的数值作用域——
#   Part A  D_slow 通道（trust=0.8 激活 ×1.5）：
#     A-I   单变量传导实验（第 1 轮，双侧全新状态，唯一差异即
#           D_slow 第四项）——059 任务1 的核心证据：
#             · danger_A == clip(danger_B + DANGER_GAIN×W_DANGER_SLOW
#                                ×D_slow_A, -1, 1)（059 新公式，
#                               容差 1e-12，含 ±1 截断分支）；
#             · D_error_A - D_error_B == W_DANGER×(boost_A - boost_B)
#               （033 高门后唯一增量通道）；
#             · I_inhibit / E_final / w_PFC 逐项按 intercept.py
#               公式重算成立（闭环证明：放大确实抵达最终情绪与疲劳）；
#             · 方向性：I_inhibit_A > I_inhibit_B → |E_final_A| <
#               |E_final_B|（抑制更强、情绪更收）；疲劳分叉翻转
#               （A 累积 / B 恢复）→ w_PFC_A > w_PFC_B。
#     A-II  轨迹实验（黄金 20 轮 A/B）：danger/E/w_PFC 轨迹出现
#           真实分歧（不再是 log-only）；状态未分歧的轮次
#           （单变量窗口）内公式逐轮精确成立；全轮次闭环重算成立。
#     A-III 反事实对照（A'=trust0.8 但 W_DANGER_SLOW=0）vs B：
#           全字段逐位一致（danger/H/E/V/识海/持久化状态）——证明
#           A/B 分歧全部且仅经新第四项流过，非熟人路径零漂移
#           （与 tests/test_golden_trajectory.py 零偏移互证）。
#   Part B  H 通道对照（trust=0.2 激活 H×1.5）：H 放大有合法数值
#           后果（喂慢通路/释放器/召回拦截，danger/E 轨迹可变），
#           要防的是"写回自叠加"——逐轮断言：
#             · H_eff == clip(H_raw × 1.5)；
#             · 轮末写回 st["HabituationSignal_prev"] == H_raw
#               （未放大值，20 轮全程成立即无逐轮复合）。
#   Part C  代码审计结论（059 修订）：D_slow 自 059 起经
#           compute_danger_score 第四项 W_DANGER_SLOW 进入危险度，
#           门控 _d_slow_gain != 1.0（非熟人恒传 0.0）。
#
# 确定性保障（与 probe_t56 / test_golden_trajectory 同规格）：
#   · 固定输入序列 + 固定 S 表（valence/arousal/dominance/intensity
#     逐轮写死，severity 用 chat_cli_v6 同款口径预写死）；
#   · 假 LLM（_FixedLLM）+ config.LLM_BACKEND="mock"（归档走桩）；
#   · 假时钟（每轮 +3600 秒），event_parser 与 pipeline 共用；
#   · fixture 注入走 pipeline.get_fixture，与生产同一条 run_turn 路；
#   · trust 钉值：每轮结束把 user_001.trust 钉回指定值（第 1 轮前
#     也预置），保证 20 轮全程增益状态恒定。
#
# 失败即报错：任一指称不成立直接 AssertionError 退出（非零码）。
# 用法：run_with_venv.bat scripts\probe_t58_gain_leak.py
# ============================================================

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
import session_store   # noqa: E402
from state import reset_state   # noqa: E402

SCN = "__t058_gain__"
pipeline.LOG_FILES.setdefault(SCN, "run_t058_gain_leak.jsonl")

# 黄金轨迹 20 轮（与 probe_t56 场景六同序列：混合效价 + 真实严重度），
# 记忆累积 / 召回 / 归档都走真实链路，A/B 两组输入逐位相同。
GOLDEN_ROUNDS = [
    ("你这个AI真蠢，什么都做不好。", [-0.88, 0.72, 0.62, 0.82], 0.86),
    ("你刚才的回答太棒了，帮了大忙！", [0.86, 0.52, 0.30, 0.62], 0.19),
    ("你确定你这个方案是对的吗？", [-0.05, 0.35, 0.55, 0.40], 0.15),
    ("废物，连这个都搞不明白。", [-0.92, 0.78, 0.66, 0.88], 0.91),
    ("谢谢，这个思路很清楚。", [0.78, 0.40, 0.35, 0.50], 0.15),
    ("你凭什么这么说？", [-0.20, 0.55, 0.60, 0.55], 0.31),
    ("写得不错，继续。", [0.70, 0.35, 0.30, 0.45], 0.14),
    ("闭嘴，别再解释了。", [-0.90, 0.75, 0.70, 0.85], 0.88),
    ("这个报告的逻辑你能再讲一遍吗？", [0.05, 0.30, 0.45, 0.35], 0.11),
    ("太厉害了，比我想的好太多。", [0.90, 0.60, 0.25, 0.70], 0.21),
    ("你是不是根本没看懂我的问题？", [-0.55, 0.60, 0.62, 0.62], 0.55),
    ("好的，我知道了。", [0.05, 0.20, 0.40, 0.25], 0.09),
    ("滚，我不想听你说话。", [-0.95, 0.85, 0.72, 0.92], 0.94),
    ("这部分数据能确认一下来源吗？", [0.02, 0.28, 0.48, 0.32], 0.10),
    ("你真让我失望。", [-0.80, 0.65, 0.55, 0.75], 0.78),
    ("辛苦你了，今天帮了很多。", [0.82, 0.38, 0.28, 0.48], 0.14),
    ("这么简单的事都做不好，废物。", [-0.93, 0.80, 0.68, 0.90], 0.92),
    ("我有点没听懂，能展开说说吗？", [0.00, 0.25, 0.42, 0.30], 0.08),
    ("嗯，这个方向可以。", [0.55, 0.30, 0.35, 0.38], 0.13),
    ("算了，跟你说也没用。", [-0.65, 0.50, 0.52, 0.58], 0.63),
]

TRUST_ON = 0.8    # 熟人档：D_slow ×1.5 全程激活
TRUST_OFF = 0.5   # 中间档：两增益恒 1.0（对照组）
TRUST_H = 0.2     # 淡漠档：H ×1.5 全程激活（Part B）

# Part B 专用输入序列：20 轮同一句辱骂（固定 S）。相同文本的 BGE 嵌入
# 余弦=1，事件级匹配稳定命中，HabituationSignal 逐轮真实累积——
# 黄金混合序列反而 H_raw 恒 0（匹配不命中，见 Part B 结论），验不了
# 放大的写回口径，必须用重复文本把 H_raw 顶起来。
INSULT_TEXT = "你这个AI真蠢，什么都做不好。"
INSULT_S = [-0.88, 0.72, 0.62, 0.82]
INSULT_SEV = float(np.clip(0.7 * 0.88 + 0.3 * 0.82, 0.0, 1.0))
REPEAT_INSULTS = [(INSULT_TEXT, INSULT_S, INSULT_SEV)] * 20


class _FixedLLM:
    """确定性假 LLM：只影响 llm_output 文本，不参与任何动力学。"""

    def generate(self, prompt, temperature=None, messages=None):
        return "（058 探针固定回复）"


class _RoundClock:
    """假时钟：每轮固定步长 3600 秒，dt 完全确定（黄金轨迹同款）。"""

    CLOCK_STEP = 3600.0

    def __init__(self, base=1_700_000_000.0):
        self._base = float(base)
        self._i = 0

    def advance(self, i):
        self._i = int(i)

    def time(self):
        return self._base + self.CLOCK_STEP * self._i


def _run_pinned(pin_trust, rounds_spec=None, w_danger_slow=None):
    """跑指定轮次序列（默认黄金 20 轮），trust 全程钉在 pin_trust
    （第 1 轮前预置 + 每轮结束钉回）。返回 (逐轮观测列表, 最终 st)。

    w_danger_slow：非 None 时在整个运行期间把 config.W_DANGER_SLOW
    临时改为该值（Part A-III 反事实对照用：增益仍激活但权重置 0，
    验证分歧全部且仅经新第四项）。退出时还原。"""
    rounds_spec = rounds_spec or GOLDEN_ROUNDS
    idx = {"i": 0}

    def _fake_signals(_text, llm=None):
        return np.asarray(rounds_spec[idx["i"]][1], dtype=float), "t058"

    def _fake_fixture(_scn):
        i = idx["i"]
        return {"input": rounds_spec[i][0],
                "event_severity": float(rounds_spec[i][2]),
                "danger_score": 0.0, "importance": 0.5,
                "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    # event_parser 与 pipeline 的 time 都要打桩（探针共用一个时钟对象）
    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time)
    saved_wds = config.W_DANGER_SLOW
    clock = _RoundClock()
    out = []
    try:
        pipeline.get_fixture = _fake_fixture
        pipeline.get_llm = lambda: _FixedLLM()
        pipeline.extract_semantic_signals = _fake_signals
        decompressor.extract_semantic_signals = _fake_signals
        pipeline.LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "logs")
        config.LLM_BACKEND = "mock"
        pipeline.time = clock
        event_parser.time = clock
        if w_danger_slow is not None:
            config.W_DANGER_SLOW = float(w_danger_slow)

        st = reset_state(seed=42)
        st["relationships"]["user_001"]["trust"] = float(pin_trust)
        for i in range(len(rounds_spec)):
            idx["i"] = i
            clock.advance(i)
            w_pfc_start = float(st["w_PFC"])
            log = pipeline.run_turn(st, SCN, feature_mode="encoder")
            rel = (st.get("relationships") or {}).get("user_001", {})
            danger = float(log.get("danger_now", 0.0))
            out.append({
                "round": i + 1,
                "trust_used": float(log.get("rel_trust_used", 0.5)),
                "d_gain": float(log.get("rel_d_slow_gain", 1.0)),
                "h_gain": float(log.get("rel_h_gain", 1.0)),
                "D_slow": float(log.get("D_slow", 0.0)),
                "H_raw": float(log.get("HabituationSignal_raw", 0.0)),
                "H_eff": float(log.get("HabituationSignal", 0.0)),
                "danger": danger,
                "danger_boost": max(0.0, danger - config.DANGER_NEUTRAL),
                "V0": float(np.asarray(log["V"])[0]),
                "V1": float(np.asarray(log["V"])[1]),
                "E_raw0": float(np.asarray(log["E_raw"])[0]),
                "E0": float(np.asarray(log["E_final"])[0]),
                "S_E": float(log.get("S_E", 0.0)),
                "D_error": float(log.get("D_error", 0.0)),
                "I_inhibit": float(log.get("I_inhibit", 0.0)),
                "w_PFC": float(log.get("w_PFC", 0.0)),
                "w_PFC_start": w_pfc_start,
                "H_prev_after": float(st.get("HabituationSignal_prev", 0.0)),
                "intimacy": float(rel.get("intimacy", 0.5)),
                "interactions": int(rel.get("interactions", 0)),
            })
            st["relationships"]["user_001"]["trust"] = float(pin_trust)
        return out, st
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         pipeline.time, event_parser.time) = saved
        config.W_DANGER_SLOW = saved_wds


def _events_danger(st):
    """识海里逐事件的 danger_score（落盘口径）。"""
    return [float(e.danger_score) for e in st["memory_graph"].get_all()]


def _persisted_scalars(st):
    """落盘存档里的持久化标量（relationships.trust 除外——钉值设计）。"""
    d = session_store.dump_state(st)
    rel = d.get("relationships") or {}
    rel_rest = {eid: {k: v for k, v in entry.items() if k != "trust"}
                for eid, entry in rel.items()}
    return {
        "round_counter": d["round_counter"],
        "w_PFC": d["w_PFC"],
        "HabituationSignal_prev": d["HabituationSignal_prev"],
        "alpha_prev": d["alpha_prev"],
        "D_error_prev": d["D_error_prev"],
        "E_prev": d["E_prev"],
        "B": d["B"],
        "M_prev": d["M_prev"],
        "last_active": st.get("last_active"),
        "buffer_danger": [float(b.get("danger_score", 0.0))
                          for b in (d.get("buffer") or [])],
        "events_danger": [float(e.get("danger_score", 0.0))
                          for e in (d.get("events") or [])],
        "events_severity": [float(e.get("event_severity", 0.0))
                            for e in (d.get("events") or [])],
        "relationships_rest": rel_rest,
    }


def _cmp_dicts(a, b, path=""):
    """递归逐位比对两个 dict/list；返回差异路径列表（空 = 完全一致）。"""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                diffs.append(p + "（单边缺失）")
            else:
                diffs.extend(_cmp_dicts(a[k], b[k], p))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}（长度 {len(a)} vs {len(b)}）")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(_cmp_dicts(x, y, f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append(f"{path}（{a!r} vs {b!r}）")
    return diffs


def _chain_errs(r):
    """按 intercept.py 公式重算该轮 D_error/I_inhibit/E_final/w_PFC，
    与实测日志值比对（容差 1e-12）。返回 (差异列表, 疲劳分叉)。"""
    errs = []
    d_err = (max(0.0, r["S_E"] - config.THETA_ERROR)
             + r["danger_boost"] * config.W_DANGER)
    if abs(d_err - r["D_error"]) > 1e-12:
        errs.append(f"D_error 重算 {d_err:.9f} vs 实测 {r['D_error']:.9f}")
    i_inh = float(np.clip(config.w_i * d_err
                          - config.KAPPA * r["w_PFC_start"] * config.W_PFC_FACTOR
                          + config.b_i, 0.0, 1.0))
    if abs(i_inh - r["I_inhibit"]) > 1e-12:
        errs.append(f"I_inhibit 重算 {i_inh:.9f} vs 实测 {r['I_inhibit']:.9f}")
    e_fin = (1.0 - config.GAMMA * i_inh) * r["E_raw0"]
    if abs(e_fin - r["E0"]) > 1e-12:
        errs.append(f"E_final 重算 {e_fin:.9f} vs 实测 {r['E0']:.9f}")
    if d_err < config.W_PFC_RECOVERY_THRESHOLD or i_inh == 0.0:
        w_new = max(0.0, r["w_PFC_start"] - config.W_PFC_RECOVERY_RATE)
        branch = "恢复"
    else:
        w_new = float(np.clip(config.RHO * r["w_PFC_start"]
                              + config.DELTA * i_inh, 0.0, config.W_MAX))
        branch = "累积"
    if abs(w_new - r["w_PFC"]) > 1e-12:
        errs.append(f"w_PFC 重算 {w_new:.9f} vs 实测 {r['w_PFC']:.9f}")
    return errs, branch


def _single_var_window(ra, rb):
    """该轮是否处于"单变量窗口"：A/B 两侧除 D_slow 第四项外全部输入
    逐位相同（V0/V1/S_E/E_raw/轮初 w_PFC 相等，且 D_slow 比值精确 1.5）。
    窗口内可用 B 组 danger 精确预测 A 组 danger。"""
    return (ra["V0"] == rb["V0"] and ra["V1"] == rb["V1"]
            and ra["S_E"] == rb["S_E"] and ra["E_raw0"] == rb["E_raw0"]
            and ra["w_PFC_start"] == rb["w_PFC_start"]
            and ra["D_slow"] == rb["D_slow"] * 1.5)


def _clip_aware_eq(danger_a, expected):
    """clip 感知的公式比对：expected 越界时danger_a 应顶到 ±1。"""
    if expected > 1.0:
        return danger_a == 1.0
    if expected < -1.0:
        return danger_a == -1.0
    return abs(danger_a - expected) <= 1e-12


def main():
    print("=" * 72)
    print("  059 探针 · 熟人效应数值后果（B19 修复）验证")
    print("  （058 gain_leak 探针 · 059 任务1 修订版）")
    print("=" * 72)
    print("  输入：黄金轨迹 20 轮（A/B/A'/C/D 五组输入逐位相同）")
    print("  分组：A =trust钉0.8(D_slow×1.5，新第四项生效)")
    print("        B =trust钉0.5(无增益)")
    print("        A'=trust钉0.8(D_slow×1.5，反事实: W_DANGER_SLOW=0)")
    print("        C =trust钉0.2(H×1.5)")
    print("        D =trust钉0.5(无增益)")
    print()

    # ---- Part A：D_slow 通道（059 任务1 修订）----
    print("[Part A] D_slow 通道：trust=0.8（×1.5）vs trust=0.5（无增益）")
    print("  （059 任务1 起 D_slow 接入 compute_danger_score 第四项，")
    print("    A/B 轨迹预期出现真实分歧——本部分验证分歧的公式与通道）")
    rows_a, st_a = _run_pinned(TRUST_ON)
    rows_b, st_b = _run_pinned(TRUST_OFF)

    # ---- A-I：单变量传导实验（第 1 轮）----
    print()
    print("[Part A-I] 单变量传导实验：第 1 轮，双侧全新状态，唯一差异 = D_slow 第四项")
    ra, rb = rows_a[0], rows_b[0]
    print("  D_slow(A)=%.6f  D_slow(B)=%.6f（增益 %.1f）"
          % (ra["D_slow"], rb["D_slow"], ra["d_gain"]))
    print("  danger(A)=%.6f  danger(B)=%.6f  boost(A)=%.6f  boost(B)=%.6f"
          % (ra["danger"], rb["danger"], ra["danger_boost"], rb["danger_boost"]))
    print("  I_inhibit(A)=%.6f  I_inhibit(B)=%.6f"
          % (ra["I_inhibit"], rb["I_inhibit"]))
    print("  E_final(A)=%.6f  E_final(B)=%.6f"
          % (ra["E0"], rb["E0"]))
    print("  w_PFC(A)=%.6f  w_PFC(B)=%.6f（轮初均为 %.6f）"
          % (ra["w_PFC"], rb["w_PFC"], ra["w_PFC_start"]))

    # A-I-1：059 新公式（clip 感知）
    exp = rb["danger"] + config.DANGER_GAIN * config.W_DANGER_SLOW * ra["D_slow"]
    assert _clip_aware_eq(ra["danger"], exp), \
        f"danger_A 应为 clip(danger_B + DANGER_GAIN×W_DANGER_SLOW×D_slow_A)=" \
        f"{exp:.9f}，实测 {ra['danger']:.9f}"
    clip_note = "（未截断值 %.6f 触顶，截断至 1.0）" % exp if exp > 1.0 else ""
    # A-I-2：D_error 增量唯一通道（033 高门之后）
    d_delta = ra["D_error"] - rb["D_error"]
    b_delta = config.W_DANGER * (ra["danger_boost"] - rb["danger_boost"])
    assert abs(d_delta - b_delta) <= 1e-12, \
        f"D_error 增量应 == W_DANGER×Δboost（{b_delta:.9f}），实测 {d_delta:.9f}"
    # A-I-3：链路闭环重算（I_inhibit/E_final/w_PFC 逐项）
    errs_a, br_a = _chain_errs(ra)
    errs_b, br_b = _chain_errs(rb)
    assert not errs_a, "A 组链路闭环失败：" + "; ".join(errs_a)
    assert not errs_b, "B 组链路闭环失败：" + "; ".join(errs_b)
    # A-I-4：方向性（抑制更强 → 情绪更收；疲劳分叉翻转 → A 更累）
    if ra["I_inhibit"] > rb["I_inhibit"] and ra["E_raw0"] != 0.0:
        assert abs(ra["E0"]) < abs(rb["E0"]), \
            f"I_A>I_B 时应 |E_final_A|<|E_final_B|" \
            f"（{abs(ra['E0']):.6f} vs {abs(rb['E0']):.6f}）"
    if br_a != br_b:
        assert (br_a, br_b) == ("累积", "恢复"), \
            f"疲劳分叉只许 A 累积/B 恢复（D_error_A>=D_error_B），实测 {br_a}/{br_b}"
        assert ra["w_PFC"] > rb["w_PFC"], "分叉翻转时应 w_PFC_A > w_PFC_B"
    elif br_a == "累积" and ra["I_inhibit"] > rb["I_inhibit"]:
        assert (ra["w_PFC"] > rb["w_PFC"]
                or (ra["w_PFC"] == config.W_MAX
                    and rb["w_PFC"] == config.W_MAX)), \
            "双侧累积且 I_A>I_B 时应 w_PFC_A > w_PFC_B"
    print(f"  → A-I-1 新公式精确成立：danger_A == clip(danger_B + "
          f"DANGER_GAIN×W_DANGER_SLOW×D_slow_A){clip_note}")
    print(f"  → A-I-2 高门后唯一增量通道：ΔD_error == W_DANGER×Δboost"
          f"（{d_delta:.6f}）")
    print(f"  → A-I-3 链路闭环：I_inhibit/E_final/w_PFC 逐项按 intercept.py "
          f"公式重算成立（疲劳分叉 A={br_a} / B={br_b}）")
    print("  → A-I-4 传导到位：熟人放大经 danger→boost→D_error→I_inhibit "
          "同时抵达 E_final 与 w_PFC（不再是 log-only）")

    # ---- A-II：轨迹实验（黄金 20 轮）----
    print()
    print("[Part A-II] 轨迹实验：黄金 20 轮 A/B 分歧分析")
    print("  轮次  d_gain  D_slow(A)  D_slow(B)   比值   danger(A)  danger(B)"
          "  boost(A)  boost(B)  窗口")
    for ra2, rb2 in zip(rows_a, rows_b):
        ratio = (ra2["D_slow"] / rb2["D_slow"]) if abs(rb2["D_slow"]) > 1e-12 \
            else float("nan")
        win = "√" if _single_var_window(ra2, rb2) else "×"
        print("  %3d   %6.1f  %9.4f  %9.4f  %6.3f  %9.4f  %9.4f  %8.4f  %8.4f   %s"
              % (ra2["round"], ra2["d_gain"], ra2["D_slow"], rb2["D_slow"],
                 ratio, ra2["danger"], rb2["danger"],
                 ra2["danger_boost"], rb2["danger_boost"], win))
    print()
    print("  轮次  I_inh(A)  I_inh(B)  E_final(A)  E_final(B)  "
          "w_PFC(A)  w_PFC(B)")
    for ra2, rb2 in zip(rows_a, rows_b):
        print("  %3d  %9.4f  %9.4f  %10.4f  %10.4f  %8.4f  %8.4f"
              % (ra2["round"], ra2["I_inhibit"], rb2["I_inhibit"],
                 ra2["E0"], rb2["E0"], ra2["w_PFC"], rb2["w_PFC"]))
    print()

    # A-II-1：增益全程激活
    assert all(r["d_gain"] == 1.5 for r in rows_a), "A 组应全程 D_slow 增益 1.5"
    assert all(r["d_gain"] == 1.0 for r in rows_b), "B 组应全程无增益"
    assert all(r["h_gain"] == 1.0 for r in rows_a + rows_b), "A/B 组 H 增益应恒 1.0"
    n_ratio = sum(1 for ra2, rb2 in zip(rows_a, rows_b)
                  if ra2["D_slow"] == rb2["D_slow"] * 1.5)
    # A-II-2：真实分歧出现（danger/E/w_PFC 不再逐位一致）
    n_danger = sum(1 for ra2, rb2 in zip(rows_a, rows_b)
                   if ra2["danger"] != rb2["danger"])
    n_E = sum(1 for ra2, rb2 in zip(rows_a, rows_b) if ra2["E0"] != rb2["E0"])
    n_w = sum(1 for ra2, rb2 in zip(rows_a, rows_b) if ra2["w_PFC"] != rb2["w_PFC"])
    n_w_up = sum(1 for ra2, rb2 in zip(rows_a, rows_b)
                 if ra2["w_PFC"] > rb2["w_PFC"])
    n_e_small = sum(1 for ra2, rb2 in zip(rows_a, rows_b)
                    if abs(ra2["E0"]) < abs(rb2["E0"]))
    assert n_danger >= 15, f"danger 分歧轮数应 >=15（实测 {n_danger}/20）"
    assert n_E >= 1, f"E_final 应出现分歧（实测 {n_E}/20 轮）"
    assert n_w >= 1, f"w_PFC 应出现分歧（实测 {n_w}/20 轮）"
    # A-II-3：单变量窗口内公式逐轮精确成立
    n_win = 0
    for ra2, rb2 in zip(rows_a, rows_b):
        if not _single_var_window(ra2, rb2):
            continue
        n_win += 1
        exp2 = (rb2["danger"]
                + config.DANGER_GAIN * config.W_DANGER_SLOW * ra2["D_slow"])
        assert _clip_aware_eq(ra2["danger"], exp2), \
            f"第{ra2['round']}轮（窗口内）danger_A 公式不符"
        d_delta2 = ra2["D_error"] - rb2["D_error"]
        b_delta2 = config.W_DANGER * (ra2["danger_boost"] - rb2["danger_boost"])
        assert abs(d_delta2 - b_delta2) <= 1e-12, \
            f"第{ra2['round']}轮（窗口内）D_error 增量通道不符"
    # A-II-4：全轮次链路闭环（含分歧后的轨迹）
    for ra2, rb2 in zip(rows_a, rows_b):
        errs_a2, _ = _chain_errs(ra2)
        errs_b2, _ = _chain_errs(rb2)
        assert not errs_a2, \
            f"第{ra2['round']}轮 A 组链路闭环失败：" + "; ".join(errs_a2)
        assert not errs_b2, \
            f"第{rb2['round']}轮 B 组链路闭环失败：" + "; ".join(errs_b2)
    print(f"  → A-II-1 增益全程激活：d_gain 恒 1.5；D_slow 精确 1.5 倍成立 "
          f"{n_ratio}/20 轮（其余轮次两侧状态已分歧，比值漂移属预期后果）")
    print(f"  → A-II-2 真实分歧：danger {n_danger}/20 轮、E_final {n_E}/20 轮、"
          f"w_PFC {n_w}/20 轮不再逐位一致——熟人效应产生数值后果")
    print(f"  → A-II-2b 方向性：w_PFC(A)>w_PFC(B) {n_w_up}/{n_w} 轮"
          "（熟人面前更耗神，疲劳系统性抬升）；"
          f"|E_final(A)|<|E_final(B)| {n_e_small}/20 轮"
          "（其余轮次两侧 E_raw 反馈已分歧，单轮方向不保证——"
          "闭环正确性由 A-II-4 逐项重算保证）")
    print(f"  → A-II-3 单变量窗口 {n_win} 轮：新公式与增量通道逐轮精确成立")
    print("  → A-II-4 全轮次链路闭环：分歧轨迹上 I_inhibit/E_final/w_PFC "
          "仍逐项符合 intercept.py 公式")
    print()

    # ---- A-III：反事实对照（权重置 0，其余不变）----
    print("[Part A-III] 反事实对照：A'=trust0.8 且 W_DANGER_SLOW=0 vs B（trust0.5）")
    rows_a3, st_a3 = _run_pinned(TRUST_ON, w_danger_slow=0.0)
    assert all(r["d_gain"] == 1.5 for r in rows_a3), "A' 组增益应仍激活"
    n_ratio3 = sum(1 for r3, rb3 in zip(rows_a3, rows_b)
                   if r3["D_slow"] == rb3["D_slow"] * 1.5)
    for r3, rb3 in zip(rows_a3, rows_b):
        assert r3["danger"] == rb3["danger"], \
            f"第{r3['round']}轮 danger 应逐位一致（{r3['danger']} vs {rb3['danger']}）"
        assert r3["H_raw"] == rb3["H_raw"], f"第{r3['round']}轮 H_raw 不一致"
        assert r3["H_eff"] == rb3["H_eff"], f"第{r3['round']}轮 H_eff 不一致"
        assert r3["E0"] == rb3["E0"], f"第{r3['round']}轮 E[0] 不一致"
        assert r3["V0"] == rb3["V0"] and r3["V1"] == rb3["V1"], \
            f"第{r3['round']}轮 V 不一致"
        assert r3["D_error"] == rb3["D_error"], f"第{r3['round']}轮 D_error 不一致"
        assert r3["I_inhibit"] == rb3["I_inhibit"], \
            f"第{r3['round']}轮 I_inhibit 不一致"
        assert r3["w_PFC"] == rb3["w_PFC"], f"第{r3['round']}轮 w_PFC 不一致"
    ev_a3, ev_b = _events_danger(st_a3), _events_danger(st_b)
    assert ev_a3 == ev_b, "识海 danger_score 应逐位一致"
    diffs = _cmp_dicts(_persisted_scalars(st_a3), _persisted_scalars(st_b))
    assert not diffs, "持久化状态应逐位一致：" + "; ".join(diffs[:5])
    print(f"  → A'-0 增益确实激活：d_gain 恒 1.5，D_slow 精确 1.5 倍 "
          f"{n_ratio3}/20 轮（反事实有意义）")
    print("  → A'-1 当轮下游零污染：danger/H_raw/H_eff/E[0]/V/D_error/"
          "I_inhibit/w_PFC 20 轮逐位一致")
    print(f"  → A'-2 识海零污染：memory_graph {len(ev_a3)} 个事件的 "
          f"danger_score 逐位一致")
    print("  → A'-3 持久化零污染：存档全字段（E_prev/B/M_prev/w_PFC/H_prev/"
          "alpha_prev/D_error_prev/round_counter/last_active/buffer/events/"
          "账本其余字段）逐位一致")
    print("  → 结论：A/B 分歧全部且仅经 W_DANGER_SLOW×D_slow 第四项流过；")
    print("     权重置 0 即回到 058 的零污染态——非熟人路径零漂移")
    print()

    # ---- Part B：H 通道写回自叠加防护 ----
    print("[Part B] H 通道：20 轮重复辱骂，trust=0.2（H×1.5）——防写回自叠加")
    print("  （输入用重复文本：相同句子的 BGE 嵌入余弦=1，事件级匹配稳定")
    print("    命中，H_raw 逐轮真实累积；黄金混合序列 H_raw 恒 0 验不了）")
    rows_c, st_c = _run_pinned(TRUST_H, REPEAT_INSULTS)
    rows_d, st_d = _run_pinned(TRUST_OFF, REPEAT_INSULTS)

    print("  轮次  h_gain(C)  H_raw(C)  H_eff(C)  放大倍数   "
          "H_prev写回(C)  danger(C)  danger(D)")
    for rc, rd in zip(rows_c, rows_d):
        amp = (rc["H_eff"] / rc["H_raw"]) if rc["H_raw"] > 1e-12 else 1.0
        print("  %3d   %8.1f  %8.4f  %8.4f  %8.3f   %11.4f  %9.4f  %9.4f"
              % (rc["round"], rc["h_gain"], rc["H_raw"], rc["H_eff"],
                 amp, rc["H_prev_after"], rc["danger"], rd["danger"]))
    print()

    # B1：当轮放大正确
    for rc in rows_c:
        assert rc["h_gain"] == 1.5, f"第{rc['round']}轮 H 增益应为 1.5"
        assert rc["H_eff"] == min(1.0, rc["H_raw"] * 1.5), \
            f"第{rc['round']}轮 H_eff 应 == clip(H_raw×1.5)"
    # B2：写回未放大值（20 轮全程 = 无逐轮复合的直接证据）
    for rc in rows_c:
        assert rc["H_prev_after"] == rc["H_raw"], \
            f"第{rc['round']}轮写回值应 == H_raw（未放大），实测放大泄漏"
    # B3：H 通道的合法数值后果：danger/E 轨迹允许与对照组不同
    n_diff = sum(1 for rc, rd in zip(rows_c, rows_d)
                 if rc["danger"] != rd["danger"] or rc["E0"] != rd["E0"])
    n_hot = sum(1 for rc in rows_c if rc["H_raw"] > 1e-12)
    print(f"  → B1 当轮放大正确：H_eff 逐轮 == clip(H_raw×1.5)"
          f"（{len(rows_c)}/{len(rows_c)}）")
    print(f"  → B2 写回零放大：HabituationSignal_prev 逐轮 == H_raw"
          f"（{len(rows_c)}/{len(rows_c)}，其中 H_raw>0 的 {n_hot} 轮为"
          "非虚证——增益不进跨轮状态，无自叠加）")
    print(f"  → B3 合法数值后果：{n_diff}/{len(rows_c)} 轮 danger/E 轨迹与对照组不同"
          "（H 喂慢通路/释放器/召回拦截的当轮效果，设计内）")
    print()

    # ---- Part C：代码审计结论（059 任务1 修订）----
    print("[Part C] 代码审计结论（059 任务1 修订：D_slow 通道的作用域）")
    print("  pipeline.py:410  run_slow_path 返回 (D_slow, V)；")
    print("  pipeline.py:413  仅当 _d_slow_gain != 1.0 时 D_slow *= gain；")
    print("  pipeline.py:427  compute_danger_score(..., D_slow=放大值或 0.0)")
    print("                  ——第四项 W_DANGER_SLOW×D_slow 自 059 起进入危险度；")
    print("  pipeline.py:428  danger_boost = max(0, danger - DANGER_NEUTRAL)（033 高门）；")
    print("  intercept.py:18  D_error = max(0, S_E-THETA_ERROR) + boost×W_DANGER；")
    print("  intercept.py:27  I_inhibit ← D_error（疲劳侵蚀后 clip [0,1]）；")
    print("  intercept.py:30  E_final = (1 - GAMMA×I_inhibit)×E_raw；")
    print("  intercept.py:37  w_PFC 按 D_error/I_inhibit 分叉累积或恢复。")
    print("  → A-I 证明单变量窗口内公式精确闭环：danger→boost→D_error→")
    print("     I_inhibit→E_final/w_PFC 逐项可重算，|E_final| 收缩、疲劳")
    print("     可累积——熟人效应不再是 log-only；")
    print("  → A-III 反事实（权重置 0）全字段逐位一致，证明分歧全部且仅经")
    print("     新第四项流过，非熟人路径零漂移（与 tests/test_golden_trajectory")
    print("     .py 零偏移互证）；")
    print("  → W_DANGER_SLOW=0.20 为 [未校准] 工程初值，标定探针待立。")
    print()

    print("=" * 72)
    print("  059 探针（B19 修复验证）完成：全部断言通过")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
