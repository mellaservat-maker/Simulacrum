# M2 第二步探针回应 · probe_xinguang_budget.py

回应日期：2026-10-06
状态：探针通过，资源适配度因子被验证到，等待裁决。

---

## 1. scripts/probe_xinguang_budget.py 完整原文

```python
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
```

---

## 2. 运行输出原文

```
XG_COMPLEXITY_BASE = 0.5

=== required_xg 分布 ===
记忆编号 | 激活值 | required_xg
M1 | 0.786804 | 0.393402
M2 | 0.376970 | 0.188485
M3 | 0.339653 | 0.169827
M4 | 0.599307 | 0.299653
M5 | 0.404826 | 0.202413

=== B_xg = 1.0 ===
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
----------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 1.000000 | 0.779215
M2 | 0.376970 | 0.869556 | 1.000000 | 0.327796
M3 | 0.339653 | 0.599702 | 1.000000 | 0.203691
M4 | 0.599307 | 0.782265 | 1.000000 | 0.468817
M5 | 0.404826 | 0.199861 | 1.000000 | 0.080909
----------------------------------------------------------------
最高分: M1  分数 = 0.779215

=== B_xg = 0.5 ===
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
----------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 1.000000 | 0.779215
M2 | 0.376970 | 0.869556 | 1.000000 | 0.327796
M3 | 0.339653 | 0.599702 | 1.000000 | 0.203691
M4 | 0.599307 | 0.782265 | 1.000000 | 0.468817
M5 | 0.404826 | 0.199861 | 1.000000 | 0.080909
----------------------------------------------------------------
最高分: M1  分数 = 0.779215

=== B_xg = 0.2 ===
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
----------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 0.508386 | 0.396142
M2 | 0.376970 | 0.869556 | 1.000000 | 0.327796
M3 | 0.339653 | 0.599702 | 1.000000 | 0.203691
M4 | 0.599307 | 0.782265 | 0.667438 | 0.312906
M5 | 0.404826 | 0.199861 | 0.988078 | 0.079944
----------------------------------------------------------------
最高分: M1  分数 = 0.396142

=== B_xg = 0.1 ===
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
----------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 0.254193 | 0.198071
M2 | 0.376970 | 0.869556 | 0.530546 | 0.173911
M3 | 0.339653 | 0.599702 | 0.588836 | 0.119940
M4 | 0.599307 | 0.782265 | 0.333719 | 0.156453
M5 | 0.404826 | 0.199861 | 0.494039 | 0.039972
----------------------------------------------------------------
最高分: M1  分数 = 0.198071

=== B_xg = 0.05 ===
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
----------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 0.127096 | 0.099035
M2 | 0.376970 | 0.869556 | 0.265273 | 0.086956
M3 | 0.339653 | 0.599702 | 0.294418 | 0.059970
M4 | 0.599307 | 0.782265 | 0.166860 | 0.078227
M5 | 0.404826 | 0.199861 | 0.247019 | 0.019986
----------------------------------------------------------------
最高分: M1  分数 = 0.099035

=== 最高分记忆随 B_xg 变化 ===
B_xg | 最高分记忆 | 分数
----------------------------------------
1.0 | M1 | 0.779215
0.5 | M1 | 0.779215
0.2 | M1 | 0.396142
0.1 | M1 | 0.198071
0.05 | M1 | 0.099035

探针完成
```

---

## 3. XG_COMPLEXITY_BASE 数值与 required_xg 分布

**XG_COMPLEXITY_BASE = 0.5**（来自 config.py，未改动）

required_xg = XG_COMPLEXITY_BASE × activation：

| 记忆 | 激活值 | required_xg |
|------|--------|-----------|
| M1 | 0.786804 | 0.393402 |
| M2 | 0.376970 | 0.188485 |
| M3 | 0.339653 | 0.169827 |
| M4 | 0.599307 | 0.299653 |
| M5 | 0.404826 | 0.202413 |

**资源适配度压制阈值**：当 B_xg < required_xg 时 resource_fit < 1。
- M1 required_xg=0.393：B_xg ≤ 0.2 时被压制
- M4 required_xg=0.300：B_xg ≤ 0.2 时被压制
- M5 required_xg=0.202：B_xg ≤ 0.1 时被压制
- M2 required_xg=0.188：B_xg ≤ 0.1 时被压制
- M3 required_xg=0.170：B_xg ≤ 0.05 时被压制

---

## 4. 一致性校验结果

**全部通过。**

校验方式：对每条记忆、每个 B_xg 档位，比较"分项复算值（activation × match_val × resource_fit）"与"competition_score 真值"，误差阈值 1e-9。

5 条记忆 × 5 档位 = **25 次比对，全部误差 ≤ 1e-9**，无一触发 TODO_REVIEW。

NaN/inf 检查：25 次调用的 activation / match_val / resource_fit / score 全部通过 math.isfinite。

硬性停止条件检查：B_xg=0.05 时 5 条 resource_fit 分别为 0.127 / 0.265 / 0.294 / 0.167 / 0.247，**不全为 1**，未触发停止。

---

## 5. M1 模块 hash 对比（21 个文件）

| 文件 | 探针前 hash（前 16 位） | 探针后 hash（前 16 位） | 一致 |
|------|----------------------|----------------------|------|
| config.py | 4C08C8A615F73802 | 4C08C8A615F73802 | ✅ |
| emotion.py | 5C3B62AE2F1F2116 | 5C3B62AE2F1F2116 | ✅ |
| emotion_releaser.py | 98FC7D769E003194 | 98FC7D769E003194 | ✅ |
| event_parser.py | BD7F71EAB7F1264C | BD7F71EAB7F1264C | ✅ |
| fast_path.py | 25D837BF496060BD | 25D837BF496060BD | ✅ |
| feature_encoder.py | 7E960C11F2D54FD3 | 7E960C11F2D54FD3 | ✅ |
| hippocampal_evaluator.py | 704716715903BF93 | 704716715903BF93 | ✅ |
| intercept.py | F594D53EDADBC353 | F594D53EDADBC353 | ✅ |
| llm_client.py | 93580C7D67A5029C | 93580C7D67A5029C | ✅ |
| logger.py | 76A179D173979B1D | 76A179D173979B1D | ✅ |
| main.py | A580118515DBF54C | A580118515DBF54C | ✅ |
| memory_graph.py | FD7FB1058E22F8D9 | FD7FB1058E22F8D9 | ✅ |
| multimodal.py | 6C2978D1A872E615 | 6C2978D1A872E615 | ✅ |
| pipeline.py | FBE5B9907CE8AE17 | FBE5B9907CE8AE17 | ✅ |
| policy.py | 0E541AF602A60E9A | 0E541AF602A60E9A | ✅ |
| prompt_builder.py | 15DE0D3EEA9C8D1E | 15DE0D3EEA9C8D1E | ✅ |
| schemas.py | BAA9F2CE59196394 | BAA9F2CE59196394 | ✅ |
| slow_path.py | 34540EF43E1CE240 | 34540EF43E1CE240 | ✅ |
| state.py | AB49DFC3FBC90E46 | AB49DFC3FBC90E46 | ✅ |
| state_translator.py | B9EA5135060DA6A2 | B9EA5135060DA6A2 | ✅ |
| xinguang.py | 694FCCDA9BDB885E | 694FCCDA9BDB885E | ✅ |

**全部 21 个 M1 模块 hash 完全一致**，未修改任何 M1 模块。

本次唯一新增文件：`d:\aiqg\emotion_agent\scripts\probe_xinguang_budget.py`。

---

## 关键观察（供裁决参考，未改任何参数）

1. **资源适配度因子被验证到**：B_xg=0.2 时 M1 的 resource_fit 从 1.0 降到 0.508，分数从 0.779 降到 0.396；B_xg=0.05 时全部 5 条 resource_fit < 0.3。
2. **B_xg=1.0 与 B_xg=0.5 输出完全相同**：因 XG_COMPLEXITY_BASE=0.5，5 条 required_xg 最大 0.393 < 0.5，故 B_xg≥0.5 时 resource_fit 全为 1，资源适配度因子无差异——这是参数选择的后果，非 bug。
3. **最高分记忆始终为 M1**：5 档 B_xg 下最高分都是 M1（高激活+高匹配），但分数随 B_xg 紧张单调下降（0.779 → 0.396 → 0.198 → 0.099），证明压制有效。
4. **未触发任何硬性停止条件**。

---

停止，不进入下一步，等待裁决。