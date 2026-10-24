# ============================================================
# test_dialogue_history.py · 055 号工作 · 任务 1
#
# 覆盖"让 LLM 看到对话历史"：
#   T055-A1 build_messages：历史在前、本轮 prompt 在最后一条 user
#   T055-A2 collect_history：raw_log 与 memory_buffer 合并去重、按轮数截断
#   T055-A3 collect_history：按 token 预算从新到旧截断，最近一轮必留
#   T055-A4 端到端：连聊两轮，第二轮真的把第一轮的问答带给了模型
#   T055-A5 无历史（首轮）→ 只有一条 user 消息，行为与旧版一致
# ============================================================

import os
import sys
import types

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config                       # noqa: E402
import pipeline                     # noqa: E402
import decompressor                 # noqa: E402
import event_parser                 # noqa: E402
from prompt_builder import (build_messages, collect_history,   # noqa: E402
                            estimate_tokens)
from raw_log import RawLog          # noqa: E402
from short_term_buffer import ShortTermBuffer   # noqa: E402
from state import reset_state       # noqa: E402

# run_turn 用场景名索引日志文件，测试场景需注册独立日志名
pipeline.LOG_FILES.setdefault("__t055__", "run_t055_history.jsonl")


def _fake_state(records=None, buffered=None):
    st = {"raw_log": RawLog(), "memory_buffer": ShortTermBuffer(default_keep=2)}
    for rid, u, a in (records or []):
        st["raw_log"].append({"round_id": rid, "user_input": u, "ai_output": a,
                              "timestamp": 1700000000.0 + rid})
    for rid, u, a in (buffered or []):
        st["memory_buffer"].append({"round_id": rid, "user_input": u,
                                    "ai_output": a,
                                    "timestamp": 1700000000.0 + rid})
    return st


# ---- T055-A1 ----

def test_build_messages_history_then_current():
    hist = [{"role": "user", "content": "我叫张三"},
            {"role": "assistant", "content": "你好张三"}]
    msgs = build_messages("【状态】平静", "我叫什么名字？", "task_first", hist)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "我叫张三"}
    assert msgs[1] == {"role": "assistant", "content": "你好张三"}
    assert msgs[-1]["role"] == "user"
    # 独白与当前输入仍在最后一条里（状态注入方式不变）
    assert "我叫什么名字？" in msgs[-1]["content"]
    assert "【状态】平静" in msgs[-1]["content"]


def test_build_messages_without_history_is_single_user():
    msgs = build_messages("【状态】平静", "你好", "task_first", None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# ---- T055-A2 ----

def test_collect_history_merges_and_orders():
    st = _fake_state(records=[(1, "第一轮", "回复一"), (2, "第二轮", "回复二")],
                     buffered=[(3, "第三轮", "回复三")])
    msgs = collect_history(st, max_turns=6, max_tokens=1000)
    assert [m["content"] for m in msgs] == ["第一轮", "回复一",
                                            "第二轮", "回复二",
                                            "第三轮", "回复三"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3


def test_collect_history_dedupes_same_round():
    """同一 round_id 同时出现在 raw_log 与 buffer 时只保留一份。"""
    st = _fake_state(records=[(1, "第一轮", "回复一")],
                     buffered=[(1, "第一轮", "回复一")])
    msgs = collect_history(st, max_turns=6, max_tokens=1000)
    assert len(msgs) == 2


def test_collect_history_respects_max_turns():
    st = _fake_state(records=[(i, "第%d轮" % i, "回复%d" % i) for i in range(1, 9)])
    msgs = collect_history(st, max_turns=2, max_tokens=100000)
    assert [m["content"] for m in msgs] == ["第7轮", "回复7", "第8轮", "回复8"]


# ---- T055-A3 ----

def test_collect_history_token_budget_keeps_newest():
    long_text = "很长的内容" * 200       # 估算约 1000 token
    st = _fake_state(records=[(1, long_text, long_text),
                              (2, "短的", "短的"),
                              (3, "最新一句", "最新回复")])
    msgs = collect_history(st, max_turns=6, max_tokens=100)
    # 预算只够最近一轮；旧的被裁掉，但最近一轮一定在
    assert any("最新一句" in m["content"] for m in msgs)
    assert not any("很长的内容" in m["content"] for m in msgs)


def test_estimate_tokens_cjk_vs_ascii():
    assert estimate_tokens("你好") >= 2
    assert estimate_tokens("abcdefghij") < estimate_tokens("一二三四五六七八九十")


# ---- T055-A4：端到端（不联网）----

class _Recorder:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, temperature=None, messages=None):
        self.calls.append(messages)
        return "收到"


def test_pipeline_passes_history_to_llm(monkeypatch, tmp_path):
    """连聊两轮，第二轮必须把第一轮的 user/assistant 一起带给模型。"""
    rec = _Recorder()
    idx = {"i": 0}
    inputs = [("我叫张三。", [-0.1, 0.3, 0.5, 0.3]),
              ("我叫什么名字？", [0.0, 0.3, 0.5, 0.3])]

    def _fake_signals(text, llm=None):
        return np.asarray(inputs[idx["i"]][1], dtype=float), "test"

    def _fake_fixture(_scn):
        return {"input": inputs[idx["i"]][0], "event_severity": 0.1,
                "danger_score": 0.0, "importance": 0.5, "emotion_weight": 0.5,
                "unfinished": 0.0, "x": np.zeros(8), "x_fast": np.zeros(4)}

    monkeypatch.setattr(pipeline, "get_llm", lambda: rec)
    monkeypatch.setattr(pipeline, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(decompressor, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(pipeline, "get_fixture", _fake_fixture)
    monkeypatch.setattr(pipeline, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")

    st = reset_state(seed=42)
    idx["i"] = 0
    pipeline.run_turn(st, "__t055__", feature_mode="encoder")
    idx["i"] = 1
    pipeline.run_turn(st, "__t055__", feature_mode="encoder")

    assert len(rec.calls) == 2
    # 第一轮没有历史
    assert len(rec.calls[0]) == 1
    # 第二轮带上了第一轮的问答
    second = rec.calls[1]
    assert len(second) >= 3
    assert second[0]["role"] == "user" and "我叫张三" in second[0]["content"]
    assert second[1]["role"] == "assistant"
    assert "我叫什么名字" in second[-1]["content"]
