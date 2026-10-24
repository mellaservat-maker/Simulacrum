# ============================================================
# probe_t55_history.py · 055 号工作 · 任务 1
#
# 验证"让 LLM 看到对话历史"是否真的打通。分两段：
#
# A 确定性验证（不联网，必然可复现）
#   用"只认 messages 里写了什么"的桩模型——它不看 prompt，只在收到的
#   messages 里找「我叫X」，找不到就说不知道。这样：
#     带历史 → 必须答出"张三"
#     不带历史（HISTORY_MAX_TURNS=0）→ 必须答"不知道"
#   这是对数据通路的严格证明：模型能答出名字，只可能是因为历史真的到了。
#
# B 真模型验证（联网，可能因网络/额度失败）
#   走完整链路连聊两轮，问"我叫什么名字"，检查回答里是否出现"张三"。
#   失败会明确报告原因，不静默跳过。
#
# 用法：python scripts/probe_t55_history.py
# ============================================================

import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
from prompt_builder import collect_history   # noqa: E402
from state import reset_state                # noqa: E402

SCENARIO = "__probe_t55__"
pipeline.LOG_FILES.setdefault(SCENARIO, "probe_t55_history.jsonl")

TURNS = [
    ("我叫张三。", [-0.10, 0.30, 0.50, 0.30]),
    ("我叫什么名字？", [0.00, 0.30, 0.50, 0.30]),
]


class _NameOnlyStub:
    """只从 messages 里读信息、完全不看 prompt 的桩模型。"""

    def __init__(self):
        self.seen = []

    def generate(self, prompt, temperature=None, messages=None):
        self.seen.append(messages)
        text = "\n".join(m.get("content", "") for m in (messages or [])
                         if m.get("role") == "user")
        m = re.search(r"我叫([\u4e00-\u9fa5]{1,4})", text)
        return ("你叫%s。" % m.group(1)) if m else "我不知道你叫什么。"


def _run_turns(llm, max_turns, tmpdir):
    idx = {"i": 0}

    def _fake_signals(text, llm=None):
        return np.asarray(TURNS[idx["i"]][1], dtype=float), "probe"

    def _fake_fixture(_scn):
        return {"input": TURNS[idx["i"]][0], "event_severity": 0.1,
                "danger_score": 0.0, "importance": 0.5, "emotion_weight": 0.5,
                "unfinished": 0.0, "x": np.zeros(8), "x_fast": np.zeros(4)}

    saved = (pipeline.get_llm, pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals, pipeline.get_fixture,
             pipeline.LOGS_DIR, config.LLM_BACKEND, config.HISTORY_MAX_TURNS)
    try:
        pipeline.get_llm = lambda: llm
        pipeline.extract_semantic_signals = _fake_signals
        decompressor.extract_semantic_signals = _fake_signals
        pipeline.get_fixture = _fake_fixture
        pipeline.LOGS_DIR = str(tmpdir)
        config.LLM_BACKEND = "mock"
        config.HISTORY_MAX_TURNS = max_turns
        st = reset_state(seed=42)
        outs = []
        for i in range(len(TURNS)):
            idx["i"] = i
            log = pipeline.run_turn(st, SCENARIO, feature_mode="encoder")
            outs.append(log)
        return outs, st
    finally:
        (pipeline.get_llm, pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals, pipeline.get_fixture,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         config.HISTORY_MAX_TURNS) = saved


def part_a():
    print("A 确定性验证（桩模型只看 messages）")
    print("-" * 66)
    with tempfile.TemporaryDirectory() as td:
        stub_on = _NameOnlyStub()
        outs_on, st = _run_turns(stub_on, config.HISTORY_MAX_TURNS, td)
    with tempfile.TemporaryDirectory() as td:
        stub_off = _NameOnlyStub()
        outs_off, _ = _run_turns(stub_off, 0, td)

    print("  第二轮发给模型的 messages 条数：带历史 %d / 不带历史 %d"
          % (len(stub_on.seen[1]), len(stub_off.seen[1])))
    print("  第二轮实际 messages：")
    for m in stub_on.seen[1]:
        c = m["content"].replace("\n", " ")[:60]
        print("    %-9s %s" % (m["role"], c))
    a_on = outs_on[1]["llm_output"].strip()
    a_off = outs_off[1]["llm_output"].strip()
    print("  带历史   答：%s" % a_on)
    print("  不带历史 答：%s" % a_off)
    print("  历史消息数（log.history_msgs）：第1轮 %d / 第2轮 %d"
          % (outs_on[0].get("history_msgs", -1),
             outs_on[1].get("history_msgs", -1)))
    ok = ("张三" in a_on) and ("张三" not in a_off)
    print("  判定：%s" % ("通过（信息确实经 messages 到达模型）" if ok
                        else "失败（历史没进 payload 或桩读不到）"))
    return ok


def main():
    ok_a = part_a()

    print("\nB 真模型验证（联网，可能因网络/额度失败）")
    print("-" * 66)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "scripts"))
        from chat_cli_v6 import chat_turn   # noqa: E402
        st = reset_state(seed=42)
        a1 = chat_turn(st, TURNS[0][0])[0]["llm_output"].strip()
        print("  第1轮 问：%s" % TURNS[0][0])
        print("        答：%s" % a1.replace("\n", " ")[:60])
        log2 = chat_turn(st, TURNS[1][0])[0]
        a2 = log2["llm_output"].strip()
        print("  第2轮 问：%s" % TURNS[1][0])
        print("        答：%s" % a2.replace("\n", " ")[:60])
        print("  本轮带给模型的历史条数：%d" % log2.get("history_msgs", -1))
        if "张三" in a2:
            print("  判定：通过（真模型答出了名字）")
        else:
            print("  判定：未通过（模型没答出名字，见上方原文）")
    except Exception as e:
        print("  真模型调用失败：%s: %s" % (type(e).__name__, e))
        print("  （网络/额度问题，不影响 A 段的确定性结论）")

    print("\n总判定：%s" % ("A 段通过，历史注入链路成立" if ok_a
                          else "A 段未通过，需排查"))


if __name__ == "__main__":
    main()
