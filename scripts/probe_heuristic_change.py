# ============================================================
# probe_heuristic_change.py · V4.3 验证探针
# 目的：验证 hippocampal_evaluator.evaluate 的排序口径
#       已从 match_score 切换为 competition_score。
# 构造 4 条记忆，保证 match_score 第一名 ≠ competition_score 第一名，
# 调用 evaluate 后检查排序第一位是否为 A（competition_score 口径）。
# ============================================================

import sys
import os
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import EventNode
from memory_graph import LayeredMemoryGraph
from hippocampal_evaluator import evaluate
from xinguang import match_score, competition_score


def make_memory(mid, timestamp, importance, emotion_weight, unfinished,
                danger_score, embedding, persons, tasks, raw_text=""):
    mem = EventNode(
        event_id=mid,
        timestamp=timestamp,
        raw_text=raw_text or f"memory {mid}",
        event_embedding=np.asarray(embedding, dtype=float),
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
        danger_score=danger_score,
        persons=set(persons),
        tasks=set(tasks),
    )
    return mem


def main():
    NOW = 1_000_000.0
    B_xg = 0.5
    cur_persons = {"用户"}
    cur_tasks = {"骂 AI"}

    # ---- 4 条候选记忆（全部 > MATCH_THRESHOLD=0.6，且 match_score 第一名 ≠ competition_score 第一名）----
    # 当前事件 embedding = [1, 0, 0, 0, 0, 0, 0, 0]

    # A：上次骂过 AI。相似度中等偏高，权重全高 → competition_score 第一
    memA = make_memory(
        "A", NOW - 86400.0,
        importance=0.95, emotion_weight=0.9, unfinished=0.85, danger_score=0.7,
        embedding=[0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"骂 AI"},
        raw_text="上次骂过 AI",
    )

    # B：提过技术问题。相似度最高，权重全低 → match_score 第一
    memB = make_memory(
        "B", NOW - 86400.0 * 2,
        importance=0.3, emotion_weight=0.15, unfinished=0.1, danger_score=0.05,
        embedding=[0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"技术问题"},
        raw_text="提过技术问题",
    )

    # C：夸过 AI。相似度中等，权重中等
    memC = make_memory(
        "C", NOW - 86400.0 * 3,
        importance=0.4, emotion_weight=0.5, unfinished=0.2, danger_score=0.1,
        embedding=[0.7, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"夸 AI"},
        raw_text="夸过 AI",
    )

    # D：闲聊。相似度刚过线，权重低
    memD = make_memory(
        "D", NOW - 86400.0 * 5,
        importance=0.2, emotion_weight=0.1, unfinished=0.0, danger_score=0.0,
        embedding=[0.65, 0.6, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"闲聊"},
        raw_text="闲聊",
    )

    mg = LayeredMemoryGraph()
    mg.add(memA)
    mg.add(memB)
    mg.add(memC)
    mg.add(memD)

    # ---- 当前事件 ----
    current_event = EventNode(
        event_id="CURRENT",
        timestamp=NOW,
        raw_text="用户再次骂 AI",
        event_embedding=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        importance=0.0,
        emotion_weight=0.0,
        unfinished=0.0,
        persons=set(cur_persons),
        tasks=set(cur_tasks),
    )

    x = np.zeros(8, dtype=float)

    # ---- 先打印各记忆的 match_score 与 competition_score 对照（便于核对）----
    print("=== 各记忆对照 ===")
    print("记忆 | match_score | competition_score")
    for mem in mg.get_all():
        ms = match_score(current_event.event_embedding, mem.event_embedding)
        cs = competition_score(
            mem, current_event, B_xg, NOW,
            current_event.persons, mem.persons,
            current_event.tasks, mem.tasks,
        )
        print(f"{mem.event_id}   | {ms:.6f}   | {cs:.6f}")
    print()

    # ---- 调用 evaluate ----
    result = evaluate(current_event, mg, x, B_xg, NOW)

    print("=== evaluate 返回 MatchedMemories ===")
    for item in result["MatchedMemories"]:
        print(f"  event_id={item['event_id']}, score={item['score']:.6f}, residual={item['residual']}")
    print()

    print(f"M_text = {result['M_text']}")
    print()

    # ---- 验收：排序第一位是否为 A ----
    matched = result["MatchedMemories"]
    if not matched:
        print("TODO_REVIEW: 没有任何候选被召回，无法判断排序口径")
        sys.exit(1)

    first_id = matched[0]["event_id"]
    print(f"排序第一位 = {first_id}")
    if first_id == "A":
        print("排序正确：第一位是 A（competition_score 口径）")
    elif first_id == "B":
        print("TODO_REVIEW: 排序第一位是 B（match_score 口径），接入失败")
        sys.exit(1)
    else:
        print(f"TODO_REVIEW: 排序第一位是 {first_id}，既不是 A 也不是 B")
        sys.exit(1)

    print()
    print("探针完成")


if __name__ == "__main__":
    main()
