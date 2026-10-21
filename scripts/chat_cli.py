# ============================================================
# chat_cli.py · 交互式命令行对话
# 让所有者在终端里与小凌对话，记忆通路全程工作
# 链路完全照 pipeline.run_turn，唯一区别：输入来自 input()，feature_mode 用 encoder
# ============================================================

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MEMORY_WRITE_ENABLED, MEMORY_READ_ENABLED
from state import reset_state
from fast_path import run_fast_path
from slow_path import run_slow_path
from emotion_releaser import run_releaser
from emotion import run_emotion
from intercept import run_intercept
from policy import policy_minimal
from xinguang import update_xinguang_budget
from hippocampal_evaluator import evaluate as hippocampal_evaluate
from event_parser import parse as event_parse
from feature_encoder import encode as feature_encode
from state_translator import state_to_prompt
from prompt_builder import build_prompt
from llm_client import get_llm


def chat_turn(st, user_input):
    """跑一轮完整链路，返回 AI 输出。照 run_turn 逻辑，去掉 fixture 与 reset。"""
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

    # encoder 模式：真实输入走特征编码
    x, x_fast = feature_encode(
        event, st["M_prev"], st["HabituationSignal_prev"], st["B"], st["memory_graph"]
    )

    # 心光预算（用回合开始的 B）
    B = st["B"]
    B_xg = update_xinguang_budget(B[0], B[1])

    # 海马体评估
    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, now)
    M = hippo["M"]
    ReframedContext = hippo["ReframedContext"]
    HabituationSignal = hippo["HabituationSignal"]
    HistoricalResidual = hippo["HistoricalResidual"]
    M_text = hippo["M_text"]

    # 读取侧（M2-D，默认关闭，只记录不影响评估）
    if MEMORY_READ_ENABLED:
        packages = st["archiver"].get_packages()
        if packages:
            latest = packages[-1]
            recalled = st["decompressor"].for_emotion_evaluator(latest)
            st["recalled_memory"] = recalled
    else:
        st["recalled_memory"] = None

    # 快通路（可能更新 B[1]）
    D_fast, B = run_fast_path(x_fast, B)

    # 慢通路（用更新后的 B）
    D_slow, V = run_slow_path(x, ReframedContext, M, B, st["w_PFC"], HabituationSignal, np.zeros(4))  # [012] 无语义信号，补全 0

    # 情绪释放器
    raw_emotion_impact, released_impact, release_rate = run_releaser(V, D_fast, HabituationSignal, 0.0)  # [012] S_valence=0

    # 情绪加点
    E_raw = run_emotion(st["E_prev"], released_impact, B)

    # 拦截层
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(E_raw, st["w_PFC"])

    # 策略
    action = policy_minimal(E_final, I_inhibit, event_severity)

    # 状态翻译 + prompt + LLM
    monologue = state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal)
    prompt = build_prompt(monologue, user_input)
    llm_output = get_llm().generate(prompt)

    # 回写识海（5 字段，5.5.14）
    event.event_severity = event_severity
    event.danger_score = float(np.clip(V[1] * 2 - 0.5, -1.0, 1.0))
    event.importance = importance
    event.emotion_weight = emotion_weight
    event.unfinished = unfinished
    st["memory_graph"].add(event)

    # 写入侧（M2-D）：每轮结束写缓冲区，达阈值归档
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


def main():
    st = reset_state(seed=42)
    print("=== 小凌对话测试 ===")
    print("输入 '退出' 结束")
    print()

    while True:
        user_input = input("你：").strip()

        if user_input in ("退出", "quit", "exit"):
            break
        if not user_input:
            print("（空输入，请再说一次）")
            print()
            continue

        try:
            llm_output, action = chat_turn(st, user_input)
            print(f"AI：{llm_output}")
            buf_len = len(st["memory_buffer"])
            pkg_count = len(st["archiver"].get_packages())
            recalled = st.get("recalled_memory")
            if recalled is None:
                recalled_str = "无"
            else:
                recalled_str = str(recalled)
            print(f"[缓冲区: {buf_len} 条 | 已归档: {pkg_count} 个包 | 召回: {recalled_str}]")
        except Exception as e:
            print(f"（本轮出错：{type(e).__name__}: {e}）")
            print("（你可以继续输入下一句）")
        print()

    print(f"\n会话结束。共归档 {len(st['archiver'].get_packages())} 个记忆包。")


if __name__ == "__main__":
    main()
