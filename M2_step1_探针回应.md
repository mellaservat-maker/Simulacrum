# M2 第一步探针回应 · probe_xinguang.py

回应日期：2026-10-06
状态：探针通过，等待裁决。

---

## 1. scripts/probe_xinguang.py 完整原文

```python
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
```

---

## 2. 运行命令原文

```powershell
python scripts/probe_xinguang.py
```

工作目录：`d:\aiqg\emotion_agent`

---

## 3. 终端输出原文

```
记忆编号 | 激活值 | 匹配度 | 资源适配度 | 竞争分
------------------------------------------------------------
M1 | 0.786804 | 0.990354 | 1.000000 | 0.779215
M2 | 0.376970 | 0.869556 | 1.000000 | 0.327796
M3 | 0.339653 | 0.599702 | 1.000000 | 0.203691
M4 | 0.599307 | 0.782265 | 1.000000 | 0.468817
M5 | 0.404826 | 0.199861 | 1.000000 | 0.080909
------------------------------------------------------------
竞争分最高: M1  分数 = 0.779215
探针完成
```

### 验收检查

| 验收项 | 结果 |
|--------|------|
| 5 条竞争分全部为有限值（非 NaN/inf） | ✅ 通过（math.isfinite 全过） |
| 5 条竞争分有差异（不全部相同） | ✅ 通过（5 个值两两不同） |
| 分数表随脚本运行直接打印在终端 | ✅ 通过 |

### 分数差异说明（便于人工核对）

- **M1**（高重要+近时+高情绪+未完成+人物任务双匹配）：0.779，最高
- **M2**（远时间，时间衰减拉低 recency_term）：0.328
- **M3**（低重要低情绪，但语义高度相似）：0.204
- **M4**（人物匹配任务不匹配）：0.469
- **M5**（embedding 几乎正交，语义相似度低）：0.081，最低

5 条分数差异显著且排序符合公式预期（M1 > M4 > M2 > M3 > M5）。

---

## 4. 未改动任何 M1 模块的证据

### Hash 比对（探针前 vs 探针后，21 个 M1 .py 文件）

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

**全部 21 个 M1 模块 hash 完全一致**，包括 xinguang.py / memory_graph.py / hippocampal_evaluator.py / config.py。

本次唯一新增的文件：`d:\aiqg\emotion_agent\scripts\probe_xinguang.py`（探针脚本本身）。

### 硬性约束确认

- ✅ 未修改 xinguang.py
- ✅ 未修改 memory_graph.py
- ✅ 未修改 hippocampal_evaluator.py
- ✅ 未修改任何 M1 模块
- ✅ 未引入新符号、新模块、新依赖（仅用 sys / os / time / math / numpy + schemas + xinguang + config）
- ✅ 未改动 config.py

---

停止。不进入下一步，等待裁决。