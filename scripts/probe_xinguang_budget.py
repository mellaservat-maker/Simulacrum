# ============================================================
# probe_xinguang_budget.py · M2 探针 v2 · 资源适配度压制验证
# 目的：在 B_xg 紧张档位下验证 resource_fit 能否真正压制高消耗记忆。
# 沿用 v1 的 5 条记忆构造（完全相同）。
# 不接入主链路，不修改任何 M1 模块。
# ============================================================

import sys
import os
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import EventNode
from xinguang import competition_score, time_decay_weight, match_score
from config import XG_W_ACT, XG_W_MATCH, XG_COMPLEXITY_BASE


def make_memory(mid, timestamp, importance, emotion_weight, unfinished,
                embedding, persons, tasks):
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


def build_candidates_and_current():
    """沿用 v1 探针构造，完全相同。"""
    NOW = 1_000_000.0
    candidates = []

    mem, persons, tasks = make_memory(
        "M1", NOW - 86400.0, importance=0.9, emotion_weight=0.8, unfinished=0.7,
        embedding=[0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"写报告"},
    )
    candidates.append((mem, persons, tasks))

    mem, persons, tasks = make_memory(
        "M2", NOW - 86400.0 * 30, importance=0.5, emotion_weight=0.4, unfinished=0.3,
        embedding=[0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"写报告"},
    )
    candidates.append((mem, persons, tasks))

    mem, persons, tasks = make_memory(
        "M3", NOW - 3600.0, importance=0.2, emotion_weight=0.1, unfinished=0.0,
        embedding=[0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"李四"}, tasks={"开会"},
    )
    candidates.append((mem, persons, tasks))

    mem, persons, tasks = make_memory(
        "M4", NOW - 7200.0, importance=0.6, emotion_weight=0.5, unfinished=0.4,
        embedding=[0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"张三"}, tasks={"别的任务"},
    )
    candidates.append((mem, persons, tasks))

    mem, persons, tasks = make_memory(
        "M5", NOW - 1800.0, importance=0.3, emotion_weight=0.2, unfinished=0.1,
        embedding=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
        persons={"王五"}, tasks={"完全无关"},
    )
    candidates.append((mem, persons, tasks))

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
    return NOW, candidates, current_event, cur_persons, cur_tasks


def compute_terms(mem, current_event, B_xg, NOW, cur_persons, mem_persons, cur_tasks, mem_tasks):
    """分项复算，与 competition_score 内部一致。"""
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
    return activation, match_val, required_xg, resource_fit


def main():
    NOW, candidates, current_event, cur_persons, cur_tasks = build_candidates_and_current()

    print(f"XG_COMPLEXITY_BASE = {XG_COMPLEXITY_BASE}")
    print()

    # ---- required_xg 分布（与 B_xg 无关，先算一次）----
    print("=== required_xg 分布 ===")
    print("记忆编号 | 激活值 | required_xg")
    for mem, mem_persons, mem_tasks in candidates:
        act, _, req, _ = compute_terms(
            mem, current_event, 1.0, NOW,
            cur_persons, mem_persons, cur_tasks, mem_tasks
        )
        print(f"{mem.event_id} | {act:.6f} | {req:.6f}")
    print()

    B_xg_levels = [1.0, 0.5, 0.2, 0.1, 0.05]
    summary = []  # (B_xg, best_mid, best_score)

    for B_xg in B_xg_levels:
        print(f"=== B_xg = {B_xg} ===")
        print("记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分")
        print("-" * 64)
        rows = []
        for mem, mem_persons, mem_tasks in candidates:
            act, mat, req, rfit = compute_terms(
                mem, current_event, B_xg, NOW,
                cur_persons, mem_persons, cur_tasks, mem_tasks
            )
            score_true = competition_score(
                mem, current_event, B_xg, NOW,
                cur_persons, mem_persons, cur_tasks, mem_tasks
            )
            # 一致性强校验
            score_recomputed = act * mat * rfit
            if abs(score_recomputed - score_true) > 1e-9:
                print(f"TODO_REVIEW: 一致性失败 {mem.event_id} "
                      f"复算={score_recomputed} 真值={score_true} "
                      f"误差={abs(score_recomputed - score_true)}")
                sys.exit(1)
            # NaN/inf 检查
            for name, val in [("activation", act), ("match", mat),
                              ("resource_fit", rfit), ("score", score_true)]:
                if not math.isfinite(val):
                    print(f"TODO_REVIEW: {mem.event_id} {name} 非有限值 = {val}")
                    sys.exit(1)
            rows.append((mem.event_id, act, mat, rfit, score_true))
            print(f"{mem.event_id} | {act:.6f} | {mat:.6f} | {rfit:.6f} | {score_true:.6f}")
        best = max(rows, key=lambda r: r[4])
        print("-" * 64)
        print(f"最高分: {best[0]}  分数 = {best[4]:.6f}")
        summary.append((B_xg, best[0], best[4]))
        print()

    # ---- 汇总表 ----
    print("=== 最高分记忆随 B_xg 变化 ===")
    print("B_xg | 最高分记忆 | 分数")
    print("-" * 40)
    for B_xg, mid, sc in summary:
        print(f"{B_xg} | {mid} | {sc:.6f}")

    # ---- 硬性停止条件检查：B_xg=0.05 时 5 条资源适配度是否全为 1 ----
    last_B = B_xg_levels[-1]
    rfit_at_tightest = []
    for mem, mem_persons, mem_tasks in candidates:
        _, _, _, rfit = compute_terms(
            mem, current_event, last_B, NOW,
            cur_persons, mem_persons, cur_tasks, mem_tasks
        )
        rfit_at_tightest.append(rfit)
    if all(abs(r - 1.0) < 1e-12 for r in rfit_at_tightest):
        print()
        print("TODO_REVIEW: XG_COMPLEXITY_BASE 过小或 B_xg 档位范围不足，"
              "资源适配度因子在现有参数下无法体现差异化。"
              "需要所有者决策是否调整参数或档位。")
        sys.exit(1)

    print()
    print("探针完成")


if __name__ == "__main__":
    main()
