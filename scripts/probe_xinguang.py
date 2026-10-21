# ============================================================
# probe_xinguang.py · M2 探针 · 心光竞争差分化验证
# 目的：在手工构造的 5 条候选记忆上验证 xinguang.competition_score
#       能否算出差分化的分数。不接入主链路，不修改任何 M1 模块。
# ============================================================

import sys
import os
import time
import math

import numpy as np

# 让脚本能从 scripts/ 目录直接 import emotion_agent 根目录的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import EventNode
from xinguang import competition_score


def make_memory(mid, timestamp, importance, emotion_weight, unfinished,
                embedding, persons, tasks):
    """手工构造一条候选记忆。persons/tasks 是集合，单独保存供 competition_score 使用。"""
    mem = EventNode(
        event_id=mid,
        timestamp=timestamp,
        raw_text=f"memory {mid}",
        event_embedding=np.asarray(embedding, dtype=float),
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
    )
    return mem, set(persons), set(tasks)


def main():
    # 设定"现在"时间锚点（用浮点秒，便于构造不同时间差）
    NOW = 1_000_000.0  # ~11.6 天

    # B_xg 取明确数值 0.5
    B_xg = 0.5

    # ---- 5 条候选记忆（差异化构造，便于观察分数差异）----
    # 故意让每条在某一维度上突出，便于核对竞争分排序
    candidates = []

    # M1：高重要性、近时间、高情绪权重、有未完成
    mem, persons, tasks = make_memory(
        "M1", NOW - 86400.0, importance=0.9, emotion_weight=0.8, unfinished=0.7,
        embedding=[0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"写报告"},
    )
    candidates.append((mem, persons, tasks))

    # M2：中等重要、远时间（时间衰减）
    mem, persons, tasks = make_memory(
        "M2", NOW - 86400.0 * 30, importance=0.5, emotion_weight=0.4, unfinished=0.3,
        embedding=[0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"写报告"},
    )
    candidates.append((mem, persons, tasks))

    # M3：低重要性但语义高度相似（embedding 与 current_event 几乎一致）
    mem, persons, tasks = make_memory(
        "M3", NOW - 3600.0, importance=0.2, emotion_weight=0.1, unfinished=0.0,
        embedding=[0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"李四"}, tasks={"开会"},
    )
    candidates.append((mem, persons, tasks))

    # M4：人物匹配但任务不匹配
    mem, persons, tasks = make_memory(
        "M4", NOW - 7200.0, importance=0.6, emotion_weight=0.5, unfinished=0.4,
        embedding=[0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"别的任务"},
    )
    candidates.append((mem, persons, tasks))

    # M5：完全正交（embedding 几乎 0 相似）
    mem, persons, tasks = make_memory(
        "M5", NOW - 1800.0, importance=0.3, emotion_weight=0.2, unfinished=0.1,
        embedding=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
        persons={"王五"}, tasks={"完全无关"},
    )
    candidates.append((mem, persons, tasks))

    # ---- 当前事件 ----
    current_event = EventNode(
        event_id="CURRENT",
        timestamp=NOW,
        raw_text="当前事件",
        event_embedding=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        importance=0.0,
        emotion_weight=0.0,
        unfinished=0.0,
    )
    cur_persons = {"张三"}
    cur_tasks = {"写报告"}

    # ---- 对 5 条候选记忆调用 competition_score ----
    rows = []
    for idx, (mem, mem_persons, mem_tasks) in enumerate(candidates, start=1):
        # 拆开调用以便分项记录
        # 为得到分项值，复算一次各分量（与 competition_score 内部一致）
        from xinguang import time_decay_weight, match_score
        from config import XG_W_ACT, XG_W_MATCH, XG_COMPLEXITY_BASE

        importance_term = mem.importance
        recency_term = time_decay_weight(mem, NOW)
        emotion_weight_term = mem.emotion_weight
        relation_weight_term = 0.0
        unfinished_term = mem.unfinished

        semantic_sim_term = match_score(current_event.event_embedding, mem.event_embedding)
        person_match_term = 1.0 if (cur_persons & mem_persons) else 0.0
        time_match_term = recency_term
        task_match_term = 1.0 if (cur_tasks & mem_tasks) else 0.0

        activation = (
            XG_W_ACT[0] * importance_term +
            XG_W_ACT[1] * recency_term +
            XG_W_ACT[2] * emotion_weight_term +
            XG_W_ACT[3] * relation_weight_term +
            XG_W_ACT[4] * unfinished_term
        )
        match_val = (
            XG_W_MATCH[0] * semantic_sim_term +
            XG_W_MATCH[1] * person_match_term +
            XG_W_MATCH[2] * time_match_term +
            XG_W_MATCH[3] * task_match_term
        )
        required_xg = XG_COMPLEXITY_BASE * activation
        resource_fit = min(1.0, B_xg / max(required_xg, 1e-6))

        # 真值调用（与上面分项算的应当一致）
        score = competition_score(
            mem, current_event, B_xg, NOW,
            cur_persons, mem_persons, cur_tasks, mem_tasks
        )

        # NaN / inf 检查
        if not math.isfinite(score):
            print(f"TODO_REVIEW: 记忆 {mem.event_id} 竞争分非有限值 = {score}")
            sys.exit(1)
        if not math.isfinite(activation):
            print(f"TODO_REVIEW: 记忆 {mem.event_id} 激活值非有限值 = {activation}")
            sys.exit(1)
        if not math.isfinite(match_val):
            print(f"TODO_REVIEW: 记忆 {mem.event_id} 匹配度非有限值 = {match_val}")
            sys.exit(1)
        if not math.isfinite(resource_fit):
            print(f"TODO_REVIEW: 记忆 {mem.event_id} 资源适配度非有限值 = {resource_fit}")
            sys.exit(1)

        rows.append({
            "idx": idx,
            "mid": mem.event_id,
            "activation": activation,
            "match_val": match_val,
            "resource_fit": resource_fit,
            "score": score,
        })

    # ---- 验证分数有差异 ----
    scores = [r["score"] for r in rows]
    if len(set(scores)) == 1:
        print("TODO_REVIEW: 5 条竞争分全部相同，无差异")
        sys.exit(1)

    # ---- 打印分数表 ----
    print("记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分")
    print("-" * 60)
    for r in rows:
        print(f"{r['mid']} | {r['activation']:.6f} | {r['match_val']:.6f} | "
              f"{r['resource_fit']:.6f} | {r['score']:.6f}")

    # ---- 最高分记忆 ----
    best = max(rows, key=lambda r: r["score"])
    print("-" * 60)
    print(f"竞争分最高: {best['mid']}  分数 = {best['score']:.6f}")

    print("探针完成")


if __name__ == "__main__":
    main()
