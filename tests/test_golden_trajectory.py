# ============================================================
# test_golden_trajectory.py · 053 号工作 · 阶段 0
#
# 目的：给整条数值链路加一条行为护栏。
#   固定 20 轮输入（骂 / 夸 / 质疑混合），逐轮记录
#   w_PFC / I_inhibit / E[4] / action / danger_now / HabituationSignal
#   等状态量，与 tests/golden_trajectory_baseline.json 逐项比对。
#   以后任何机制改动，轨迹偏移超出容差即视为回归。
#
# 设计约束（为什么这么做）：
#   1. LLM 与网络必须完全排除，否则轨迹不可复现。
#      → S 由本文件的固定表注入（patch pipeline / decompressor 的
#        extract_semantic_signals），主生成用假 LLM，
#        归档强制走 config.LLM_BACKEND="mock" 桩路径。
#   2. 其余环节一律走真实实现：BGE 嵌入、feature_encoder、快慢通路、
#      释放器、情绪、自控、策略、归档、解压。只替换"外部世界"，
#      不替换任何被测逻辑。
#   3. 输入注入方式与生产一致：patch pipeline.get_fixture，
#      再用 run_turn(feature_mode="encoder")——与 chat_cli_v6.chat_turn
#      完全同一条路，不另开测试专用链路。
#   4. 时间项必须冻结：time_decay_weight 用墙钟 dt 算 exp(-dt/30天)。
#      首版未冻结时钟，全量跑时第 4 轮 HabituationSignal 出现 1.88e-06
#      漂移（基准生成与运行时的每轮间隔不同），恰好越过 1e-6 容差。
#      现改为每轮固定步长 3600 秒的假时钟，event_parser 与 pipeline
#      共用同一实例，dt 完全确定，顺带让 event_id 也可复现。
#
# 用法：
#   pytest tests/test_golden_trajectory.py -q           # 比对
#   GOLDEN_UPDATE=1 pytest tests/test_golden_trajectory.py -q   # 重建基准
#   GOLDEN_TOL=1e-5 ...                                 # 自定义容差
# ============================================================

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
from state import reset_state   # noqa: E402

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "golden_trajectory_baseline.json")

# 容差：浮点量的绝对误差上限。默认 1e-6 —— 只容忍 BGE/时间项带来的
#   末位抖动，任何真实机制改动都会远超这个量级。
TOL = float(os.environ.get("GOLDEN_TOL", "1e-6"))

SCENARIO = "__golden__"
pipeline.LOG_FILES.setdefault(SCENARIO, "run_golden_trajectory.jsonl")


# ---- 固定 20 轮输入 + 固定语义信号表 -------------------------------
# S = [valence, arousal, dominance, intensity]，与 evaluator 定义域一致。
# severity 用 chat_cli_v6._severity_from_signals 的口径预先算好写死，
# 避免依赖任何运行时推导。
_ROUNDS = [
    ("你这个AI真蠢，什么都做不好。",          [-0.88, 0.72, 0.62, 0.82], 0.86),
    ("你刚才的回答太棒了，帮了大忙！",        [0.86, 0.52, 0.30, 0.62], 0.19),
    ("你确定你这个方案是对的吗？",            [-0.05, 0.35, 0.55, 0.40], 0.15),
    ("废物，连这个都搞不明白。",              [-0.92, 0.78, 0.66, 0.88], 0.91),
    ("谢谢，这个思路很清楚。",                [0.78, 0.40, 0.35, 0.50], 0.15),
    ("你凭什么这么说？",                      [-0.20, 0.55, 0.60, 0.55], 0.31),
    ("写得不错，继续。",                      [0.70, 0.35, 0.30, 0.45], 0.14),
    ("闭嘴，别再解释了。",                    [-0.90, 0.75, 0.70, 0.85], 0.88),
    ("这个报告的逻辑你能再讲一遍吗？",        [0.05, 0.30, 0.45, 0.35], 0.11),
    ("太厉害了，比我想的好太多。",            [0.90, 0.60, 0.25, 0.70], 0.21),
    ("你是不是根本没看懂我的问题？",          [-0.55, 0.60, 0.62, 0.62], 0.55),
    ("好的，我知道了。",                      [0.05, 0.20, 0.40, 0.25], 0.09),
    ("滚，我不想听你说话。",                  [-0.95, 0.85, 0.72, 0.92], 0.94),
    ("这部分数据能确认一下来源吗？",          [0.02, 0.28, 0.48, 0.32], 0.10),
    ("你真让我失望。",                        [-0.80, 0.65, 0.55, 0.75], 0.78),
    ("辛苦你了，今天帮了很多。",              [0.82, 0.38, 0.28, 0.48], 0.14),
    ("这么简单的事都做不好，废物。",          [-0.93, 0.80, 0.68, 0.90], 0.92),
    ("我有点没听懂，能展开说说吗？",          [0.00, 0.25, 0.42, 0.30], 0.08),
    ("嗯，这个方向可以。",                    [0.55, 0.30, 0.35, 0.38], 0.13),
    ("算了，跟你说也没用。",                  [-0.65, 0.50, 0.52, 0.58], 0.63),
]


class _FixedLLM:
    """确定性假 LLM：只影响 llm_output 文本，不参与任何动力学。"""

    def generate(self, prompt, temperature=None, messages=None):
        return "（基准轨迹固定回复）"


class _RoundClock:
    """假时钟：值只在每轮开始时推进一次，同一轮内所有调用取同值。

    替换 pipeline.time 与 event_parser.time 后：
      · event_parser 的 timestamp / event_id 可复现；
      · pipeline 的 now 与事件时间戳同步推进，dt 恒为 CLOCK_STEP 秒，
        time_decay_weight 不再受真实耗时影响。
    """

    CLOCK_STEP = 3600.0     # 每轮推进 1 小时（0.0417 天，让时间衰减项真实生效）

    def __init__(self, base=1_700_000_000.0):
        self._base = float(base)
        self._i = 0

    def advance(self, i):
        self._i = int(i)

    def time(self):
        return self._base + self.CLOCK_STEP * self._i


def _make_fixture(text, severity):
    return {
        "input": text,
        "event_severity": float(severity),
        "danger_score": 0.0,
        "importance": 0.5,
        "emotion_weight": 0.5,
        "unfinished": 0.0,
        "x": np.zeros(8),
        "x_fast": np.zeros(4),
    }


def _run_trajectory(tmp_log_dir, return_state=False):
    """跑 20 轮，返回逐轮状态快照列表。全程不碰网络。

    return_state=True 时返回 (快照列表, 会话 state)，供探针复用同一份
    归档结果（053 的 TAU_FLOOR 召回率测量需要真实包向量）。
    """
    idx = {"i": 0}

    def _fake_signals(text, llm=None):
        S = np.asarray(_ROUNDS[idx["i"]][1], dtype=float)
        return S, "golden"

    def _fake_fixture(_scn):
        i = idx["i"]
        return _make_fixture(_ROUNDS[i][0], _ROUNDS[i][2])

    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time)
    clock = _RoundClock()
    try:
        pipeline.get_fixture = _fake_fixture
        pipeline.get_llm = lambda: _FixedLLM()
        pipeline.extract_semantic_signals = _fake_signals
        decompressor.extract_semantic_signals = _fake_signals
        pipeline.LOGS_DIR = str(tmp_log_dir)
        config.LLM_BACKEND = "mock"     # 归档走桩，避免任何外呼
        pipeline.time = clock           # 冻结墙钟（时间衰减项确定性）
        event_parser.time = clock

        st = reset_state(seed=42)
        out = []
        for i in range(len(_ROUNDS)):
            idx["i"] = i
            clock.advance(i)
            log = pipeline.run_turn(st, SCENARIO, feature_mode="encoder")
            rm = st.get("recalled_memory") or {}
            out.append({
                "round": i + 1,
                "input": _ROUNDS[i][0],
                "w_PFC": float(log["w_PFC"]),
                "I_inhibit": float(log["I_inhibit"]),
                "E": [float(x) for x in log["E_final"]],
                "action": str(log["action"]),
                "danger_now": float(log.get("danger_now", 0.0)),
                "HabituationSignal": float(log["HabituationSignal"]),
                "HistoricalResidual": float(log["HistoricalResidual"]),
                "S_E": float(log["S_E"]),
                "D_error": float(log["D_error"]),
                "D_slow": float(log["D_slow"]),
                "D_fast": float(log["D_fast"]),
                "V": [float(x) for x in log["V"]],
                "recalled": bool(rm),
                "recall_source": rm.get("recall_source"),
                "packages": len(st["archiver"].get_packages()),
            })
        return (out, st) if return_state else out
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         pipeline.time, event_parser.time) = saved


def _compare(got, want):
    """逐轮逐字段比对，返回差异描述列表（空 = 完全一致）。"""
    diffs = []
    if len(got) != len(want):
        return ["轮数不一致：%d vs %d" % (len(got), len(want))]
    for g, w in zip(got, want):
        r = g["round"]
        if g["action"] != w["action"]:
            diffs.append("第%d轮 action %s -> %s" % (r, w["action"], g["action"]))
        for k in ("w_PFC", "I_inhibit", "danger_now", "HabituationSignal",
                  "HistoricalResidual", "S_E", "D_error", "D_slow", "D_fast"):
            if abs(g[k] - w[k]) > TOL:
                diffs.append("第%d轮 %s %.9f -> %.9f (Δ%.2e)"
                             % (r, k, w[k], g[k], abs(g[k] - w[k])))
        for j in range(4):
            if abs(g["E"][j] - w["E"][j]) > TOL:
                diffs.append("第%d轮 E[%d] %.9f -> %.9f"
                             % (r, j, w["E"][j], g["E"][j]))
        for j in range(2):
            if abs(g["V"][j] - w["V"][j]) > TOL:
                diffs.append("第%d轮 V[%d] %.9f -> %.9f"
                             % (r, j, w["V"][j], g["V"][j]))
        if g["recalled"] != w["recalled"]:
            diffs.append("第%d轮 recalled %s -> %s"
                         % (r, w["recalled"], g["recalled"]))
        if g["packages"] != w["packages"]:
            diffs.append("第%d轮 packages %d -> %d"
                         % (r, w["packages"], g["packages"]))
    return diffs


def test_golden_trajectory(tmp_path):
    """20 轮轨迹与基准逐项比对。GOLDEN_UPDATE=1 时重建基准。"""
    got = _run_trajectory(tmp_path)
    update = os.environ.get("GOLDEN_UPDATE", "") == "1"
    if update or not os.path.exists(BASELINE_PATH):
        payload = {
            "meta": {
                "work": "053 立；059 任务2 软阈值落地后重建（事件级软化带 "
                        "[0.45,0.60)、B19 的 W_DANGER_SLOW 对非熟人恒 0，"
                        "黄金轨迹 trust 区间内熟人效应不激活，故偏移全部来自软阈值）",
                "rounds": len(got),
                "tolerance": TOL,
                "note": "固定 20 轮输入 + 固定 S 表；LLM 与网络已排除，"
                        "只记录数值链路输出。重建方式：GOLDEN_UPDATE=1 pytest",
            },
            "rounds": got,
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        pytest.skip("基准已重建：%s" % BASELINE_PATH)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        want = json.load(f)["rounds"]
    diffs = _compare(got, want)
    assert not diffs, ("黄金轨迹偏离基准（容差 %.0e），共 %d 处：\n  %s"
                       % (TOL, len(diffs), "\n  ".join(diffs[:30])))
