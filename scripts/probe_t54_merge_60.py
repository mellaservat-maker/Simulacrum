# ============================================================
# probe_t54_merge_60.py · 054 号工作 · 任务 5
#
# 目的：验证 053 落地的「写入侧去重合并」在长会话里真的生效。
#   053 的黄金轨迹只有 20 轮、4 个主题互不相同的包，一次都没触发合并，
#   等于只有单测证明、没有端到端数据。本探针补这一段。
#
# 做法：60 轮同主题对话（一直聊"代码报错"，句式轮换），
#   同一批输入跑两遍：
#     merge_on  = True   去重合并开启
#     merge_off = False  去重合并关闭（053 之前的行为）
#   对比：包数量、包内轮数、总归档轮数、合并次数、平均包龄。
#
# 全程不碰网络：mock 归档后端 + 固定 S 表 + 假 LLM + 冻结时钟。
#
# 用法：python scripts/probe_t54_merge_60.py
# ============================================================

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
from state import reset_state   # noqa: E402

ROUNDS = 60
SCENARIO = "__probe_merge60__"
pipeline.LOG_FILES.setdefault(SCENARIO, "probe_t54_merge60.jsonl")

# 同主题语料：全部围绕"代码报错"，句式与情绪强度轮换
_BASE = [
    "这段代码报错了，说 KeyError。",
    "为什么我的脚本一运行就抛异常？",
    "这个函数返回 None，是不是哪里写错了。",
    "安装依赖的时候报版本冲突。",
    "调试了半天还是找不到 bug。",
    "控制台一堆红色报错，看不懂。",
    "这个报错说索引越界，怎么改？",
    "跑测试的时候挂了一个用例。",
    "日志里只有一行 traceback，信息太少。",
    "同样的代码在别人机器上能跑。",
]
INPUTS = [_BASE[i % len(_BASE)] for i in range(ROUNDS)]

# 固定 S：负效价、中低强度（"报错很烦但不至于暴怒"），轮换一点点起伏
def _s_for(i):
    val = -0.45 - 0.05 * ((i * 7) % 4)
    aro = 0.35 + 0.05 * ((i * 3) % 5)
    dom = 0.50
    itn = 0.40 + 0.05 * ((i * 5) % 5)
    return np.array([val, aro, dom, itn], dtype=float)


class _FixedLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return "（探针固定回复）"


class _Clock:
    STEP = 3600.0

    def __init__(self):
        self._base = 1_700_000_000.0
        self._i = 0

    def advance(self, i):
        self._i = int(i)

    def time(self):
        return self._base + self.STEP * self._i


def _run(merge_enabled, tmpdir):
    """跑 60 轮，返回统计 dict。"""
    idx = {"i": 0}
    clock = _Clock()

    def _fake_signals(text, llm=None):
        return _s_for(idx["i"]), "probe"

    def _fake_fixture(_scn):
        i = idx["i"]
        S = _s_for(i)
        sev = float(np.clip(0.7 * max(0.0, -S[0]) + 0.3 * max(0.0, S[3]), 0.0, 1.0))
        return {
            "input": INPUTS[i], "event_severity": sev, "danger_score": 0.0,
            "importance": 0.5, "emotion_weight": 0.5, "unfinished": 0.0,
            "x": np.zeros(8), "x_fast": np.zeros(4),
        }

    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             config.PACKAGE_MERGE_ENABLED,
             pipeline.time, event_parser.time)
    try:
        pipeline.get_fixture = _fake_fixture
        pipeline.get_llm = lambda: _FixedLLM()
        pipeline.extract_semantic_signals = _fake_signals
        decompressor.extract_semantic_signals = _fake_signals
        pipeline.LOGS_DIR = str(tmpdir)
        config.LLM_BACKEND = "mock"
        config.PACKAGE_MERGE_ENABLED = bool(merge_enabled)
        pipeline.time = clock
        event_parser.time = clock

        st = reset_state(seed=42)
        recalls = 0
        for i in range(ROUNDS):
            idx["i"] = i
            clock.advance(i)
            pipeline.run_turn(st, SCENARIO, feature_mode="encoder")
            if st.get("recalled_memory"):
                recalls += 1

        pkgs = st["archiver"].get_packages()
        stats = st["archiver"].stats()
        return {
            "packages": len(pkgs),
            "n_rounds": [int(p.n_rounds or 0) for p in pkgs],
            "pointers": [p.raw_pointer for p in pkgs],
            "merged_into": stats["merged_into"],
            "narrative_fallback": stats["narrative_fallback"],
            "round_counter": int(st["round_counter"]),
            "raw_log": len(st["raw_log"]),
            "recalls": recalls,
        }
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         config.PACKAGE_MERGE_ENABLED,
         pipeline.time, event_parser.time) = saved


def _show(tag, r):
    print("\n[%s]" % tag)
    print("  对话轮数            : %d" % r["round_counter"])
    print("  记忆包数量          : %d" % r["packages"])
    print("  包内轮数            : %s" % (r["n_rounds"] or "-"))
    print("  归档轮次合计        : %d" % sum(r["n_rounds"]))
    print("  raw_log 轮数        : %d" % r["raw_log"])
    print("  去重合并次数        : %d" % r["merged_into"])
    print("  叙事降级次数        : %d" % r["narrative_fallback"])
    print("  发生召回的轮数      : %d / %d" % (r["recalls"], r["round_counter"]))
    if r["pointers"]:
        print("  pointer 列表        :")
        for p in r["pointers"]:
            print("    %s" % p)


def main():
    with tempfile.TemporaryDirectory() as td1:
        on = _run(True, td1)
    with tempfile.TemporaryDirectory() as td2:
        off = _run(False, td2)

    _show("去重合并 开（054 现状）", on)
    _show("去重合并 关（053 之前）", off)

    print("\n对比")
    print("-" * 62)
    print("  包数量        %3d  ->  %3d   （压缩 %d 个）"
          % (off["packages"], on["packages"], off["packages"] - on["packages"]))
    if off["packages"]:
        print("  压缩率        %.0f%%"
              % (100.0 * (off["packages"] - on["packages"]) / off["packages"]))
    print("  归档轮次合计  %3d  ->  %3d   （应相等：合并只改归属不改总量）"
          % (sum(off["n_rounds"]), sum(on["n_rounds"])))
    print("  合并次数      %3d  ->  %3d" % (off["merged_into"], on["merged_into"]))
    print("  召回轮数      %3d  ->  %3d" % (off["recalls"], on["recalls"]))
    if on["n_rounds"]:
        print("  平均每包轮数  %.1f  ->  %.1f"
              % (np.mean(off["n_rounds"]) if off["n_rounds"] else 0,
                 np.mean(on["n_rounds"])))

    print("\n判定：合并开启后包数应明显下降、归档轮次总量不变（不漏轮）。")
    if on["packages"] < off["packages"] and sum(on["n_rounds"]) == sum(off["n_rounds"]):
        print("结论：合并生效，且没有丢轮。")
    elif on["packages"] >= off["packages"]:
        print("结论：合并未生效（60 轮同主题语料的余弦没有超过阈值），需复查阈值。")
    else:
        print("结论：合并生效但轮次总量不等，存在丢轮，需排查。")


if __name__ == "__main__":
    main()
