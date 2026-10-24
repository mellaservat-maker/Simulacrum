# M2 第四步探针回应 · probe_rank_divergence.py

回应日期：2026-10-06
状态：探针通过，**排序分歧构造成功**，等待裁决。

---

## 1. scripts/probe_rank_divergence.py 完整原文

```python
# ============================================================
# probe_rank_divergence.py · M2 探针 v4 · 排序口径分歧构造
# 目的：构造多条都过 MATCH_THRESHOLD 的候选记忆，让
#       "match_score 排序"与"competition_score 排序"出现分歧。
#       不调用 evaluate，只做打分对比。
# 不接入主链路，不修改任何 M1 模块。
# ============================================================

import sys
import os
import math

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import EventNode
from xinguang import competition_score, match_score


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
    )
    return mem, set(persons), set(tasks)


def main():
    NOW = 1_000_000.0
    B_xg = 0.5
    cur_persons = {"用户"}
    cur_tasks = {"骂 AI"}

    # ---- 4 条候选记忆（全部与当前事件 cosine > 0.6）----
    # 当前事件 embedding = [1, 0, 0, 0, 0, 0, 0, 0]

    # 记忆 A：用户上次骂过 AI。相似度中等偏高，importance/emotion/unfinished 全高。
    #   embedding 前两位 [0.8, 0.5] → cosine = 0.8 / sqrt(0.8²+0.5²) ≈ 0.848
    #   权重高 → competition_score 期望第一
    memA, pA, tA = make_memory(
        "A", NOW - 86400.0,
        importance=0.95, emotion_weight=0.9, unfinished=0.85, danger_score=0.7,
        embedding=[0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"骂 AI"},
        raw_text="上次骂过 AI",
    )

    # 记忆 B：用户上次提过类似技术问题。相似度最高，但权重低。
    #   embedding [0.99, 0.1] → cosine ≈ 0.995
    #   权重低 → match_score 第一，但 competition_score 被压
    memB, pB, tB = make_memory(
        "B", NOW - 86400.0 * 2,
        importance=0.3, emotion_weight=0.15, unfinished=0.1, danger_score=0.05,
        embedding=[0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"技术问题"},
        raw_text="提过技术问题",
    )

    # 记忆 C：用户上次夸过 AI。相似度中等，权重中等。
    #   embedding [0.7, 0.5, 0.5] → cosine ≈ 0.7 / sqrt(0.99) ≈ 0.704
    memC, pC, tC = make_memory(
        "C", NOW - 86400.0 * 3,
        importance=0.4, emotion_weight=0.5, unfinished=0.2, danger_score=0.1,
        embedding=[0.7, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"夸 AI"},
        raw_text="夸过 AI",
    )

    # 记忆 D：用户上次闲聊。相似度刚过线，权重低。
    #   embedding [0.65, 0.6, 0.45] → cosine ≈ 0.65 / sqrt(0.65²+0.6²+0.45²) ≈ 0.691
    memD, pD, tD = make_memory(
        "D", NOW - 86400.0 * 5,
        importance=0.2, emotion_weight=0.1, unfinished=0.0, danger_score=0.0,
        embedding=[0.65, 0.6, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0],
        persons={"用户"}, tasks={"闲聊"},
        raw_text="闲聊",
    )

    candidates = [(memA, pA, tA), (memB, pB, tB), (memC, pC, tC), (memD, pD, tD)]

    # ---- 当前事件 ----
    current_event = EventNode(
        event_id="CURRENT",
        timestamp=NOW,
        raw_text="用户再次骂 AI",
        event_embedding=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        importance=0.0,
        emotion_weight=0.0,
        unfinished=0.0,
    )

    # ---- 计算每条记忆的两个分数 ----
    rows = []
    for mem, mem_persons, mem_tasks in candidates:
        ms = match_score(current_event.event_embedding, mem.event_embedding)
        cs = competition_score(
            mem, current_event, B_xg, NOW,
            cur_persons, mem_persons, cur_tasks, mem_tasks
        )
        # NaN/inf 检查
        if not math.isfinite(ms):
            print(f"TODO_REVIEW: {mem.event_id} match_score 非有限值 = {ms}")
            sys.exit(1)
        if not math.isfinite(cs):
            print(f"TODO_REVIEW: {mem.event_id} competition_score 非有限值 = {cs}")
            sys.exit(1)
        # 阈值检查
        if ms <= 0.6:
            print(f"TODO_REVIEW: {mem.event_id} match_score={ms:.6f} 未过 MATCH_THRESHOLD(0.6)")
            sys.exit(1)
        rows.append({"mid": mem.event_id, "ms": ms, "cs": cs})

    # ---- 按 match_score 排序（降序）----
    by_ms = sorted(rows, key=lambda r: r["ms"], reverse=True)

    # ---- 按 competition_score 排序（降序）----
    by_cs = sorted(rows, key=lambda r: r["cs"], reverse=True)

    # ---- 输出两张排序表 ----
    print("=== 按 match_score 排序（现状：hippocampal_evaluator 用的口径）===")
    print("排名 | 记忆编号 | match_score | competition_score")
    print("-" * 56)
    for i, r in enumerate(by_ms, start=1):
        print(f"{i}    | {r['mid']}       | {r['ms']:.6f}   | {r['cs']:.6f}")
    print()

    print("=== 按 competition_score 排序（设计目标：手稿04 口径）===")
    print("排名 | 记忆编号 | match_score | competition_score")
    print("-" * 56)
    for i, r in enumerate(by_cs, start=1):
        print(f"{i}    | {r['mid']}       | {r['ms']:.6f}   | {r['cs']:.6f}")
    print()

    # ---- 分歧判断 ----
    ms_first = by_ms[0]["mid"]
    cs_first = by_cs[0]["mid"]
    print(f"match_score 第一名 = {ms_first}（现状 evaluate 会选它）")
    print(f"competition_score 第一名 = {cs_first}（手稿04 设计想要它）")
    diverge = "是" if ms_first != cs_first else "否"
    print(f"两者是否分歧：{diverge}")
    print()

    # ---- 硬性停止条件 ----
    if ms_first == cs_first:
        print("TODO_REVIEW: 无法构造出排序分歧，需要所有者决策："
              "是继续调整构造，还是承认两种口径在此场景下等价。")
        sys.exit(1)

    print("探针完成")


if __name__ == "__main__":
    main()
```

---

## 2. 运行输出原文

```
=== 按 match_score 排序（现状：hippocampal_evaluator 用的口径）===
排名 | 记忆编号 | match_score | competition_score
--------------------------------------------------------
1    | B       | 0.994937   | 0.316391
2    | A       | 0.847998   | 0.782771
3    | C       | 0.703526   | 0.346365
4    | D       | 0.654931   | 0.205485

=== 按 competition_score 排序（设计目标：手稿04 口径）===
排名 | 记忆编号 | match_score | competition_score
--------------------------------------------------------
1    | A       | 0.847998   | 0.782771
2    | C       | 0.703526   | 0.346365
3    | B       | 0.994937   | 0.316391
4    | D       | 0.654931   | 0.205485

match_score 第一名 = B（现状 evaluate 会选它）
competition_score 第一名 = A（手稿04 设计想要它）
两者是否分歧：是

探针完成
```

### 分歧机制说明

- **B**：相似度最高（0.995），但 importance=0.3 / emotion_weight=0.15 / unfinished=0.1 全低
  → match_score 第一，competition_score 第三（被低 activation 拖累）
- **A**：相似度次高（0.848），但 importance=0.95 / emotion_weight=0.9 / unfinished=0.85 全高
  → match_score 第二，competition_score 第一（高 activation + 资源适配度=1）
- **C**：相似度中等（0.704），权重中等 → 两口径都排第三
- **D**：相似度刚过线（0.655），权重低 → 两口径都排第四

构造满足核心约束：**match_score 第一名（B）≠ competition_score 第一名（A）**，分歧成立。

---

## 3. match_score 第一名与 competition_score 第一名

- **match_score 第一名 = B**（相似度 0.994937，competition_score 仅 0.316391）
- **competition_score 第一名 = A**（相似度 0.847998，competition_score 0.782771）

---

## 4. 是否分歧

**是。**

两口径第一名不同（B vs A），分歧构造成功，未触发硬性停止条件。

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

本次唯一新增文件：`d:\aiqg\emotion_agent\scripts\probe_rank_divergence.py`。

---

停止，不进入下一步，等待裁决。