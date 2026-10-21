# ============================================================
# auto_chat.py · 自动对话脚本
# 自动灌入预设对话，跑完整链路，打印每轮输出
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import reset_state
from event_parser import parse as event_parse
from feature_encoder import encode as feature_encode
from xinguang import update_xinguang_budget
from hippocampal_evaluator import evaluate as hippocampal_evaluate
from fast_path import run_fast_path
from slow_path import run_slow_path
from emotion_releaser import run_releaser
from emotion import run_emotion
from intercept import run_intercept
from policy import policy_minimal
from state_translator import state_to_prompt
from prompt_builder import build_prompt
from llm_client import get_llm
from config import MEMORY_WRITE_ENABLED, MEMORY_READ_ENABLED

import numpy as np


def chat_turn(st, user_input):
    """跑一轮完整链路，返回 (llm_output, action)。"""
    import time
    now = time.time()

    # 【M2 临时默认值】待后续从输入推导
    event_severity = 0.5
    importance = 0.5
    emotion_weight = 0.5
    unfinished = 0.0
    danger_score = 0.0

    event = event_parse(
        user_input,
        event_severity=event_severity,
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
        danger_score=danger_score,
    )

    x, x_fast = feature_encode(
        event, st["M_prev"], st["HabituationSignal_prev"], st["B"], st["memory_graph"]
    )

    B = st["B"]
    B_xg = update_xinguang_budget(B[0], B[1])

    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, now)
    M = hippo["M"]
    ReframedContext = hippo["ReframedContext"]
    HabituationSignal = hippo["HabituationSignal"]
    HistoricalResidual = hippo["HistoricalResidual"]
    M_text = hippo["M_text"]

    if MEMORY_READ_ENABLED:
        packages = st["archiver"].get_packages()
        if packages:
            latest = packages[-1]
            recalled = st["decompressor"].for_emotion_evaluator(latest)
            st["recalled_memory"] = recalled
    else:
        st["recalled_memory"] = None

    D_fast, B = run_fast_path(x_fast, B)
    D_slow, V = run_slow_path(x, ReframedContext, M, B, st["w_PFC"], HabituationSignal, np.zeros(4))  # [012] 无语义信号，补全 0
    raw_emotion_impact, released_impact, release_rate = run_releaser(V, D_fast, HabituationSignal, 0.0)  # [012] S_valence=0
    E_raw = run_emotion(st["E_prev"], released_impact, B)
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(E_raw, st["w_PFC"])
    action = policy_minimal(E_final, I_inhibit, event_severity)

    monologue = state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal)
    prompt = build_prompt(monologue, user_input)
    llm_output = get_llm().generate(prompt)

    event.danger_score = float(np.clip(V[1] * 2 - 0.5, -1.0, 1.0))
    st["memory_graph"].add(event)

    st["E_prev"] = E_final
    st["w_PFC"] = w_PFC_new
    st["B"] = B
    st["M_prev"] = M
    st["HabituationSignal_prev"] = HabituationSignal

    if MEMORY_WRITE_ENABLED:
        st["round_counter"] += 1
        st["memory_buffer"].append({
            "round_id": st["round_counter"],
            "user_input": user_input,
            "ai_output": llm_output,
            "timestamp": now,
        })
        if st["archiver"].should_archive():
            st["archiver"].archive()

    return llm_output, action


DIALOG = [
    "你好啊",
    "你觉得我今天心情怎么样",
    "你这个回答太烂了，真蠢",
    "算了，不指望你了",
    "你根本不懂我在说什么",
    "我再问你一次，你确定你的方案对吗",
    "你上次说的那个问题，后来怎么处理的",
    "好吧，其实你做得还行",
    "谢谢你，帮了大忙",
    "你还记得我刚才骂过你吗",
]


def main():
    st = reset_state(seed=42)
    print("=== 自动对话测试 ===")
    print(f"共 {len(DIALOG)} 轮预设对话")
    print()

    recall_rounds = []

    for i, user_input in enumerate(DIALOG):
        print(f"轮次 {i+1}")
        print(f"你：{user_input}")

        try:
            llm_output, action = chat_turn(st, user_input)
            print(f"AI：{llm_output}")

            buf_len = len(st["memory_buffer"])
            pkg_count = len(st["archiver"].get_packages())
            recalled = st.get("recalled_memory")
            if recalled is None:
                recalled_str = "无"
            else:
                recalled_str = f"有(emo={recalled.get('emotion_value')})"
                recall_rounds.append(i + 1)

            print(f"[缓冲区: {buf_len} 条 | 已归档: {pkg_count} 个包 | 召回: {recalled_str}]")
        except Exception as e:
            print(f"（本轮出错：{type(e).__name__}: {e}）")
        print()

    # 汇总
    total_packages = len(st["archiver"].get_packages())
    print(f"总共归档了 {total_packages} 个包")
    print()

    if total_packages > 0:
        last_pkg = st["archiver"].get_packages()[-1]
        print("最后一个包的完整内容：")
        print(f"  seq: {last_pkg.seq}")
        print(f"  emotion_value: {last_pkg.emotion_value}")
        print(f"  emotion_polarity: {last_pkg.emotion_polarity}")
        print(f"  keywords: {last_pkg.keywords}")
        print(f"  raw_pointer: {last_pkg.raw_pointer}")
        print(f"  narrative_cause: {last_pkg.narrative_cause}")
        print(f"  narrative_trigger: {last_pkg.narrative_trigger}")
        print(f"  narrative_consequence: {last_pkg.narrative_consequence}")
        print(f"  narrative_evaluation: {last_pkg.narrative_evaluation}")
        print(f"  narrative_emotion: {last_pkg.narrative_emotion}")
        print(f"  narrative_thought: {last_pkg.narrative_thought}")
    print()

    if recall_rounds:
        first_recall = min(recall_rounds)
        print(f"第 7 轮之后是否出现'召回：有'：{'是' if first_recall <= 10 and first_recall >= 7 else '否'}")
        print(f"首次出现在第 {first_recall} 轮")
    else:
        print("第 7 轮之后是否出现'召回：有'：否")

    print()
    print("自动对话完成")


if __name__ == "__main__":
    main()
