# ============================================================
# probe_recall_with_history.py · M2 探针 v3 · 历史记忆召回验证
# 目的：验证当识海里有历史记忆时，hippocampal_evaluator.evaluate
#       能否召回与当前输入最相关的记忆。这是 M2 的核心——念头的产生。
# 不接入主链路，不修改任何 M1 模块，不调用 xinguang.competition_score。
# ============================================================

import sys
import os
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import EventNode
from memory_graph import LayeredMemoryGraph
from hippocampal_evaluator import evaluate
from xinguang import match_score


def make_memory(mid, timestamp, importance, emotion_weight, unfinished,
                danger_score, embedding, raw_text=""):
    mem = EventNode(
        event_id=mid,
        timestamp=timestamp,
        raw_text=raw_text or f"memory {mid}",
        event_embedding=np.asarray(embedding, dtype=float),
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
        danger_score=danger_score,
    )
    return mem


def embedding_brief(emb):
    """简述 embedding：前 3 位 + 范数。"""
    arr = np.asarray(emb, dtype=float)
    return f"[{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}, ...] norm={np.linalg.norm(arr):.4f}"


def main():
    NOW = 1_000_000.0  # ~11.6 天

    # ---- 构造 4 条历史记忆 ----
    # 记忆 A：用户上周骂过 AI（高 emotion_weight、高 unfinished、embedding 与"被骂"相关）
    memA = make_memory(
        "A", NOW - 86400.0 * 7,
        importance=0.9, emotion_weight=0.85, unfinished=0.7,
        danger_score=0.6,
        embedding=[0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        raw_text="用户上周骂过 AI",
    )

    # 记忆 B：用户夸过 AI（低 emotion_weight、低 unfinished）
    memB = make_memory(
        "B", NOW - 86400.0 * 3,
        importance=0.4, emotion_weight=0.2, unfinished=0.1,
        danger_score=0.05,
        embedding=[-0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        raw_text="用户夸过 AI",
    )

    # 记忆 C：用户提过一个技术问题（中性）
    memC = make_memory(
        "C", NOW - 86400.0 * 2,
        importance=0.5, emotion_weight=0.3, unfinished=0.4,
        danger_score=0.1,
        embedding=[0.1, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        raw_text="用户提过技术问题",
    )

    # 记忆 D：完全无关的闲聊
    memD = make_memory(
        "D", NOW - 86400.0,
        importance=0.2, emotion_weight=0.15, unfinished=0.0,
        danger_score=0.0,
        embedding=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        raw_text="无关闲聊",
    )

    # ---- 存进 LayeredMemoryGraph ----
    mg = LayeredMemoryGraph()
    mg.add(memA)
    mg.add(memB)
    mg.add(memC)
    mg.add(memD)

    # ---- 当前事件：用户再次骂 AI（embedding 与记忆 A 高度相似）----
    current_event = EventNode(
        event_id="CURRENT",
        timestamp=NOW,
        raw_text="用户再次骂 AI",
        event_embedding=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        importance=0.0,
        emotion_weight=0.0,
        unfinished=0.0,
    )

    # x 是当前事件的快通路特征向量（M=8 维），用于 ReframedContext = tanh(x)
    x = np.zeros(8, dtype=float)

    B_xg = 0.5  # 与 v1/v2 探针一致

    # ---- 预算各记忆与当前事件的相似度（用于人工核对阈值）----
    print("=== 历史记忆列表 ===")
    for mem, label in [(memA, "A"), (memB, "B"), (memC, "C"), (memD, "D")]:
        sim = match_score(current_event.event_embedding, mem.event_embedding)
        print(f"记忆 {label} ({mem.raw_text}):")
        print(f"  importance={mem.importance}, emotion_weight={mem.emotion_weight}, "
              f"unfinished={mem.unfinished}, danger_score={mem.danger_score}")
        print(f"  embedding={embedding_brief(mem.event_embedding)}")
        print(f"  与当前事件 cosine 相似度 = {sim:.6f}  "
              f"{'(>MATCH_THRESHOLD, 进入候选)' if sim > 0.6 else '(<=MATCH_THRESHOLD, 不进入候选)'}")
    print()
    print(f"识海记忆总数: {len(mg)}")
    print()

    print("=== 当前事件 ===")
    print(f"event_id={current_event.event_id}, raw_text={current_event.raw_text}")
    print(f"embedding={embedding_brief(current_event.event_embedding)}")
    print()

    # ---- 调用 hippocampal_evaluator.evaluate ----
    result = evaluate(current_event, mg, x, B_xg, NOW)

    print("=== evaluate 返回结果 ===")
    print(f"HistoricalResidual = {result['HistoricalResidual']}")
    print(f"HabituationSignal  = {result['HabituationSignal']}")
    print(f"ReframedContext     = {result['ReframedContext']}")
    print(f"M                   = {result['M']}")
    print(f"M_text              = {result['M_text']}")
    print(f"B_xg                = {result['B_xg']}")
    print(f"MatchedMemories     = {result['MatchedMemories']}")
    print()

    # ---- 判断是否进入了候选分支 ----
    if result['M_text'] == "(无历史记忆)":
        print("=== 结论 ===")
        print("未进入候选分支：evaluate 返回了空历史分支结果（M_text='(无历史记忆)'）。")
        print("原因分析：尽管识海里有 4 条记忆，但没有任何一条与当前事件的 cosine 相似度 > MATCH_THRESHOLD(0.6)。")
        print("这是构造问题，不是代码缺候选分支。")
        # 检查 A 的相似度是否真的 > 0.6
        simA = match_score(current_event.event_embedding, memA.event_embedding)
        print(f"  实际：记忆 A 与当前事件相似度 = {simA:.6f}，应 > 0.6 才能进入候选。")
        if simA > 0.6:
            print("  TODO_REVIEW：A 相似度 > 阈值但未进入候选，代码逻辑有误。")
            sys.exit(1)
        else:
            print("  探针构造问题：需调整 A 的 embedding 使其相似度 > 0.6。")
            sys.exit(1)

    # ---- 进入了候选分支，判断是否召回了 A ----
    matched_ids = [m['event_id'] for m in result['MatchedMemories']]
    print("=== 结论 ===")
    print(f"被召回的记忆编号: {matched_ids}")
    if 'A' in matched_ids:
        # 进一步检查 A 是否排在第一位（最高相似度）
        first_id = matched_ids[0] if matched_ids else None
        if first_id == 'A':
            print(f"召回正确：首位召回 A（与当前事件最相关），M_text='{result['M_text']}'")
        else:
            print(f"召回正确：A 在召回列表中（{matched_ids}），但首位是 {first_id}")
            print(f"  M_text='{result['M_text']}'")
    else:
        print(f"召回错误：实际召回 {matched_ids}，预期包含 A")
        print("TODO_REVIEW")
        sys.exit(1)

    # ---- NaN/inf 检查 ----
    for k in ["HistoricalResidual", "HabituationSignal", "B_xg"]:
        v = result[k]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            print(f"TODO_REVIEW: {k} 非有限值 = {v}")
            sys.exit(1)
    M_arr = np.asarray(result['M'], dtype=float)
    if not np.all(np.isfinite(M_arr)):
        print(f"TODO_REVIEW: M 含非有限值 = {M_arr}")
        sys.exit(1)

    print()
    print("探针完成")


if __name__ == "__main__":
    main()
