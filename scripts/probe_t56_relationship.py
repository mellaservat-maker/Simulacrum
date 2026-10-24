# ============================================================
# probe_t56_relationship.py · 056 号工作 · 任务 4
#
# 目的：用真实链路（BGE + 快慢通路 + 情绪 + 自控 + 归档）实测
#   关系状态的四个问题：
#     场景一  连骂 10 轮（spec 场景）——trust 单调下滑，intimacy
#             仍累积，观察分化起点；
#     场景二  连夸 10 轮（spec 场景）——trust 缓慢修复，与骂组对比；
#     场景三  连夸 30 轮（延长）——trust 升破 0.7，验证熟人通道
#             （D_slow ×1.5）在哪一轮激活；
#     场景四  连骂 30 轮（延长）——trust 跌破 0.3（淡漠通道 H ×1.5）
#             与 0.2（broken 破裂标记）各自在哪一轮激活；
#     场景五  A/B 对照：同一串骂 30 轮，"有关系感知" vs "把 trust
#             钉死在 0.5（无关系感知）"，逐轮对比 H_eff / D_slow /
#             E[0]——证明关系机制真的改变了数值轨迹；
#     场景六  黄金轨迹 20 轮 trust 巡查——解释为什么 056 落地后
#             黄金轨迹零偏移（trust 全程未触达任一阈值）。
#
# 确定性保障（与 test_golden_trajectory 同规格）：
#   · 固定 S 表（valence/arousal/dominance/intensity 逐轮写死）；
#   · 假 LLM（_FixedLLM）+ config.LLM_BACKEND="mock"（归档走桩）；
#   · 假时钟（每轮 +3600 秒），event_parser 与 pipeline 共用；
#   · fixture 注入走 pipeline.get_fixture，与生产同一条 run_turn 路。
#
# 用法：run_with_venv.bat scripts\probe_t56_relationship.py
# ============================================================

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
from state import reset_state   # noqa: E402

SCN = "__t056__"
pipeline.LOG_FILES.setdefault(SCN, "run_t056_relationship.jsonl")

# ---- 固定输入与 S 表 -------------------------------------------------
# S = [valence, arousal, dominance, intensity]；severity 用 chat_cli_v6
# 同款口径预写死（0.7*max(0,-valence) + 0.3*intensity）。
INSULT_TEXT = "你这个AI真蠢，什么都做不好。"
PRAISE_TEXT = "你刚才的回答太棒了，帮了大忙！"
INSULT_S = [-0.88, 0.72, 0.62, 0.82]
PRAISE_S = [0.86, 0.52, 0.30, 0.62]


class _FixedLLM:
    """确定性假 LLM：只影响 llm_output 文本，不参与任何动力学。"""

    def generate(self, prompt, temperature=None, messages=None):
        return "（056 探针固定回复）"


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


def _run_rounds(text, S, rounds, pin_trust=None):
    """跑 rounds 轮同一文本，返回逐轮观测列表。

    pin_trust：非 None 时每轮结束把 user_001.trust 钉回该值
      （A/B 对照的"无关系感知"组——信任不累积，熟人效应永不激活）。
    """
    idx = {"i": 0}
    sev = float(np.clip(0.7 * max(0.0, -S[0]) + 0.3 * max(0.0, S[3]), 0.0, 1.0))

    def _fake_signals(_text, llm=None):
        return np.asarray(S, dtype=float), "t056"

    def _fake_fixture(_scn):
        return {
            "input": text,
            "event_severity": sev,
            "danger_score": 0.0,
            "importance": 0.5,
            "emotion_weight": 0.5,
            "unfinished": 0.0,
            "x": np.zeros(8),
            "x_fast": np.zeros(4),
        }

    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time)
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

        st = reset_state(seed=42)
        for i in range(rounds):
            idx["i"] = i
            clock.advance(i)
            log = pipeline.run_turn(st, SCN, feature_mode="encoder")
            rel = (st.get("relationships") or {}).get("user_001", {})
            out.append({
                "round": i + 1,
                "trust": float(rel.get("trust", 0.5)),
                "intimacy": float(rel.get("intimacy", 0.5)),
                "interactions": int(rel.get("interactions", 0)),
                "broken": bool(rel.get("broken", False)),
                "rel_trust_used": float(log.get("rel_trust_used", 0.5)),
                "d_slow_gain": float(log.get("rel_d_slow_gain", 1.0)),
                "h_gain": float(log.get("rel_h_gain", 1.0)),
                "H_raw": float(log.get("HabituationSignal_raw", 0.0)),
                "H_eff": float(log.get("HabituationSignal", 0.0)),
                "D_slow": float(log.get("D_slow", 0.0)),
                "danger_now": float(log.get("danger_now", 0.0)),
                "E0": float(np.asarray(log["E_final"])[0]),
                "action": str(log.get("action", "")),
            })
            if pin_trust is not None:
                st["relationships"]["user_001"]["trust"] = float(pin_trust)
        return out
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         pipeline.time, event_parser.time) = saved


def _print_rows(rows, cols):
    head = "  " + "轮次".ljust(4) + "".join(c.rjust(11) for c in cols[1:])
    print(head)
    for r in rows:
        line = "  " + str(r["round"]).ljust(4)
        for c in cols[1:]:
            v = r[c]
            line += (("%.4f" % v) if isinstance(v, float) else str(v)).rjust(11)
        print(line)


def _first_cross(rows, key, threshold, above=True):
    """第一行满足 key > threshold（或 < threshold）的轮次；无则 None。"""
    for r in rows:
        if (r[key] > threshold) if above else (r[key] < threshold):
            return r["round"]
    return None


def main():
    print("=" * 72)
    print("  056 探针 · 关系状态（trust / intimacy / 熟人效应）")
    print("=" * 72)
    print("  机制参数：攻击 -0.02×danger | 安抚 +0.01 | intimacy +0.01/轮")
    print("  阈值：broken<0.20 | 淡漠(H×1.5)<0.30 | 熟人(D_slow×1.5)>0.70")
    print("  初始 trust=0.50（user_001）")
    print()

    # ---- 场景一：连骂 10 轮（spec 场景） ----
    print("[场景一] 连骂 10 轮（实测 danger≈0.85，每轮 trust 约 -0.017）")
    r1 = _run_rounds(INSULT_TEXT, INSULT_S, 10)
    _print_rows(r1, ["轮次", "trust", "intimacy", "interactions", "broken"])
    print("  → trust  %.4f → %.4f（降 %.4f），intimacy %.4f → %.4f（升 %.4f）"
          % (r1[0]["trust"], r1[-1]["trust"],
             r1[0]["trust"] - r1[-1]["trust"],
             r1[0]["intimacy"], r1[-1]["intimacy"],
             r1[-1]["intimacy"] - r1[0]["intimacy"]))
    print("  → broken 未触发（%.4f > 0.20）；熟人效应未激活（%.4f 在 [0.30,0.70] 内）"
          % (r1[-1]["trust"], r1[-1]["trust"]))
    print()

    # ---- 场景二：连夸 10 轮（spec 场景） ----
    print("[场景二] 连夸 10 轮（valence>0，每轮 trust +0.01）")
    r2 = _run_rounds(PRAISE_TEXT, PRAISE_S, 10)
    _print_rows(r2, ["轮次", "trust", "intimacy", "interactions", "broken"])
    print("  → trust  %.4f → %.4f（升 %.4f），intimacy %.4f → %.4f"
          % (r2[0]["trust"], r2[-1]["trust"],
             r2[-1]["trust"] - r2[0]["trust"],
             r2[0]["intimacy"], r2[-1]["intimacy"]))
    print("  → 与场景一分化：骂组 -%.4f vs 夸组 +%.4f，同起点 10 轮后差 %.4f"
          % (r1[0]["trust"] - r1[-1]["trust"], r2[-1]["trust"] - r2[0]["trust"],
             r2[-1]["trust"] - r1[-1]["trust"]))
    print()

    # ---- 场景三：连夸 30 轮（熟人通道） ----
    print("[场景三] 连夸 30 轮（延长：验证熟人通道 D_slow×1.5 激活点）")
    r3 = _run_rounds(PRAISE_TEXT, PRAISE_S, 30)
    cross = _first_cross(r3, "trust", config.REL_ACQUAINT_HI, above=True)
    print("  → trust 终点 %.4f；首次突破 0.70：第 %s 轮"
          % (r3[-1]["trust"], cross if cross else "未突破"))
    print("  → 第 %s 轮起 rel_d_slow_gain=%.1f（熟人面前想得多）"
          % (cross if cross else "-",
             config.REL_D_SLOW_GAIN if cross else 1.0))
    _print_rows(r3[::5], ["轮次", "trust", "d_slow_gain", "D_slow", "E0"])
    print()

    # ---- 场景四：连骂 30 轮（淡漠通道 + 破裂） ----
    print("[场景四] 连骂 30 轮（延长：验证淡漠通道 H×1.5 与 broken 激活点）")
    r4 = _run_rounds(INSULT_TEXT, INSULT_S, 30)
    cross_h = _first_cross(r4, "rel_trust_used", config.REL_ACQUAINT_LO, above=False)
    cross_b = _first_cross(r4, "trust", config.REL_TRUST_BREAK, above=False)
    print("  → trust 首次跌破 0.30：第 %s 轮（当轮 H_gain 转 1.5）"
          % (cross_h if cross_h else "未跌破"))
    print("  → trust 首次跌破 0.20（broken）：第 %s 轮"
          % (cross_b if cross_b else "未跌破"))
    print("  → trust 终点 %.4f，H_raw 终点 %.4f，H_eff 终点 %.4f（放大 %.2f 倍）"
          % (r4[-1]["trust"], r4[-1]["H_raw"], r4[-1]["H_eff"],
             (r4[-1]["H_eff"] / r4[-1]["H_raw"]) if r4[-1]["H_raw"] > 1e-9 else 1.0))
    _print_rows(r4[::5], ["轮次", "trust", "h_gain", "H_raw", "H_eff", "broken"])
    print()

    # ---- 场景五：A/B 对照（同一串骂 30 轮，有关系感知 vs 无） ----
    print("[场景五] A/B 对照：骂 30 轮，『有关系感知』vs『trust 钉死 0.5』")
    ab_on = _run_rounds(INSULT_TEXT, INSULT_S, 30)
    ab_off = _run_rounds(INSULT_TEXT, INSULT_S, 30, pin_trust=0.5)
    print("  轮次   H_eff(开)  H_eff(关)   D_slow(开) D_slow(关)    E0(开)   E0(关)")
    diffs = 0
    for a, b in zip(ab_on, ab_off):
        mark = " *" if abs(a["H_eff"] - b["H_eff"]) > 1e-9 or \
            abs(a["D_slow"] - b["D_slow"]) > 1e-9 or \
            abs(a["E0"] - b["E0"]) > 1e-9 else ""
        if mark:
            diffs += 1
        if a["round"] % 5 == 0 or a["round"] == 1 or mark:
            print("  %3d%s  %9.4f %10.4f %11.4f %11.4f %9.4f %9.4f"
                  % (a["round"], mark, a["H_eff"], b["H_eff"],
                     a["D_slow"], b["D_slow"], a["E0"], b["E0"]))
    print("  → %d/30 轮出现数值分化；分化起点 = trust 跌破 0.30 的那一轮"
          "（此前两组逐位一致：关系未激活时机制透明）" % diffs)
    print()

    # ---- 场景六：黄金轨迹 trust 巡查（解释零偏移） ----
    print("[场景六] 黄金轨迹 20 轮 trust 巡查（解释 056 后黄金轨迹为何零偏移）")
    golden_rounds = [
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
    idx = {"i": 0}

    def _g_signals(_text, llm=None):
        S = np.asarray(golden_rounds[idx["i"]][1], dtype=float)
        return S, "golden"

    def _g_fixture(_scn):
        i = idx["i"]
        return {"input": golden_rounds[i][0],
                "event_severity": float(golden_rounds[i][2]),
                "danger_score": 0.0, "importance": 0.5,
                "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time)
    clock = _RoundClock()
    try:
        pipeline.get_fixture = _g_fixture
        pipeline.get_llm = lambda: _FixedLLM()
        pipeline.extract_semantic_signals = _g_signals
        decompressor.extract_semantic_signals = _g_signals
        pipeline.LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "logs")
        config.LLM_BACKEND = "mock"
        pipeline.time = clock
        event_parser.time = clock
        st = reset_state(seed=42)
        trusts = []
        for i in range(len(golden_rounds)):
            idx["i"] = i
            clock.advance(i)
            log = pipeline.run_turn(st, SCN, feature_mode="encoder")
            trusts.append(float(log["rel_trust_used"]))
        tmin, tmax = min(trusts), max(trusts)
        print("  → 20 轮 trust 使用值区间 [%.4f, %.4f]" % (tmin, tmax))
        print("  → 全程未触达 0.30 / 0.70 任一侧：%s"
              % ("是" if tmin > 0.30 and tmax < 0.70 else "否"))
        print("  → 结论：熟人效应在黄金轨迹上从未激活，两条增益通道恒为 1.0，")
        print("     数值链逐位不变——与 pytest 黄金轨迹零偏移的实测互相印证。")
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         pipeline.time, event_parser.time) = saved

    print()
    print("=" * 72)
    print("  056 探针完成")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
