# M2 第三步探针回应 · probe_recall_with_history.py

回应日期：2026-10-06
状态：探针通过，**召回正确**，等待裁决。

---

## 1. hippocampal_evaluator.py 完整原文

```python
# ============================================================
# hippocampal_evaluator.py · V4.2 · 模块3 海马体
# 规格：5.5.6
# 输出字段名严格为 M（禁止 CurrentThought_Projected）
# M1 三场景走空历史分支（L8）
# ============================================================

import numpy as np

from config import m, K_XG, MATCH_THRESHOLD
from xinguang import match_score, time_decay_weight


def evaluate(current_event, memory_graph, x, B_xg, now):
    candidates = [
        mem for mem in memory_graph.get_all()
        if match_score(current_event.event_embedding, mem.event_embedding) > MATCH_THRESHOLD
    ]

    if not candidates:
        return {
            "HistoricalResidual": 0.0,
            "HabituationSignal": 0.0,
            "ReframedContext": np.tanh(x),
            "M": np.zeros(m),
            "M_text": "(无历史记忆)",
            "B_xg": B_xg,
            "MatchedMemories": [],
        }

    HistoricalResidual = float(np.mean([mem.danger_score for mem in candidates]))
    weighted_count = sum(time_decay_weight(mem, now) for mem in candidates)
    HabituationSignal = weighted_count / (weighted_count + 1.0)

    scored = []
    for mem in candidates:
        s = match_score(current_event.event_embedding, mem.event_embedding)
        scored.append((s, mem))
    scored.sort(key=lambda t: t[0], reverse=True)

    K = min(K_XG, len(scored))
    topk_pairs = scored[:K]
    topk = [mem for _, mem in topk_pairs]

    M = np.mean([mem.event_embedding for mem in topk], axis=0)
    M = M / (np.linalg.norm(M) + 1e-8)
    M_text = "；".join([f"记得{mem.event_id}（相似度{s:.2f}）" for s, mem in topk_pairs])
    ReframedContext = np.tanh(x + HistoricalResidual + HabituationSignal)

    return {
        "HistoricalResidual": HistoricalResidual,
        "HabituationSignal": HabituationSignal,
        "ReframedContext": ReframedContext,
        "M": M,
        "M_text": M_text,
        "B_xg": B_xg,
        "MatchedMemories": [
            {"event_id": mem.event_id, "score": s, "residual": mem.danger_score}
            for s, mem in topk_pairs
        ],
    }
```

### 关键观察

- **候选分支已实现**：第 31~61 行是有候选时的完整执行路径，不是空壳。
- L8"三场景走空历史分支"的真正含义：M1 fixture 三场景的 `memory_graph` 在 reset_state 后为空，`get_all()` 返回空列表，`candidates` 自然为空，于是走空分支。**不是代码缺候选路径**。
- 候选筛选用 `match_score > MATCH_THRESHOLD(0.6)`（cosine 相似度）。
- 候选排序用 `match_score`（非 competition_score），TopK=K_XG=3。
- M = TopK 候选 event_embedding 的归一化均值（对应 L4）。

---

## 2. scripts/probe_recall_with_history.py 完整原文

```python
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
```

---

## 3. 运行输出原文

```
=== 历史记忆列表 ===
记忆 A (用户上周骂过 AI):
  importance=0.9, emotion_weight=0.85, unfinished=0.7, danger_score=0.6
  embedding=[0.950, 0.050, 0.000, ...] norm=0.9513
  与当前事件 cosine 相似度 = 0.998618  (>MATCH_THRESHOLD, 进入候选)
记忆 B (用户夸过 AI):
  importance=0.4, emotion_weight=0.2, unfinished=0.1, danger_score=0.05
  embedding=[-0.900, 0.100, 0.000, ...] norm=0.9055
  与当前事件 cosine 相似度 = -0.993884  (<=MATCH_THRESHOLD, 不进入候选)
记忆 C (用户提过技术问题):
  importance=0.5, emotion_weight=0.3, unfinished=0.4, danger_score=0.1
  embedding=[0.100, 0.900, 0.300, ...] norm=0.9539
  与当前事件 cosine 相似度 = 0.104828  (<=MATCH_THRESHOLD, 不进入候选)
记忆 D (无关闲聊):
  importance=0.2, emotion_weight=0.15, unfinished=0.0, danger_score=0.0
  embedding=[0.000, 0.000, 0.000, ...] norm=1.0000
  与当前事件 cosine 相似度 = 0.000000  (<=MATCH_THRESHOLD, 不进入候选)

识海记忆总数: 4

=== 当前事件 ===
event_id=CURRENT, raw_text="用户再次骂 AI"
embedding=[1.000, 0.000, 0.000, ...] norm=1.0000

=== evaluate 返回结果 ===
HistoricalResidual = 0.6
HabituationSignal  = 0.44192989412604666
ReframedContext     = [0.77864902 0.77864902 0.77864902 0.77864902 0.77864902 0.77864902
 0.77864902 0.77864902]
M                   = [0.99861782 0.05255883 0.         0.         0.         0.
 0.         0.        ]
M_text              = 记得A（相似度1.00）
B_xg                = 0.5
MatchedMemories     = [{'event_id': 'A', 'score': 0.9986178293325098, 'residual': 0.6}]

=== 结论 ===
被召回的记忆编号: ['A']
召回正确：首位召回 A（与当前事件最相关），M_text='记得A（相似度1.00）'

探针完成
```

### 关键观察

1. **进入了候选分支**：M_text = "记得A（相似度1.00）"，不是 "(无历史记忆)"，证明 evaluate 的候选分支被触发。
2. **召回正确**：MatchedMemories = [{'event_id': 'A', ...}]，首位且唯一被召回的是 A。
3. **相似度筛选生效**：B/C/D 三条记忆因 cosine ≤ 0.6 未进入候选，只有 A（相似度 0.9986）进入。
4. **M 字段（念头向量）**：M = [0.9986, 0.0526, 0, 0, 0, 0, 0, 0]，是 A 的 embedding 归一化结果（TopK=1 时均值=A 自身），即"念头"就是被召回的记忆 A 的 embedding。
5. **HistoricalResidual = 0.6**：来自 A 的 danger_score=0.6（候选均值）。
6. **HabituationSignal = 0.442**：来自 A 的时间衰减权重（7 天前，TAU_DAYS 衰减）。
7. **未触发任何硬性停止条件**。

---

## 4. 是否召回记忆 A

**召回正确。**

证据：
- `result['M_text'] = '记得A（相似度1.00）'`
- `result['MatchedMemories'] = [{'event_id': 'A', 'score': 0.9986178293325098, 'residual': 0.6}]`
- 首位且唯一被召回的记忆就是 A，与"当前事件再次骂 AI"最相关。
- `result['M']` = A 的 event_embedding 归一化（[0.9986, 0.0526, 0, 0, 0, 0, 0, 0]）。

三选一结论：**召回正确**（非"召回错误"、非"未进入候选分支"）。

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

本次唯一新增文件：`d:\aiqg\emotion_agent\scripts\probe_recall_with_history.py`。

---

停止，不进入下一步，等待裁决。