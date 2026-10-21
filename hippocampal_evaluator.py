# ============================================================
# hippocampal_evaluator.py · V4.3 · 模块3 海马体
# 规格：5.5.6 + competition_score 接入（T19）
# 输出字段名严格为 M（禁止 CurrentThought_Projected）
# M1 三场景走空历史分支（L8）
# ============================================================

import numpy as np

from config import m, K_XG, MATCH_THRESHOLD
from xinguang import match_score, time_decay_weight, competition_score


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
        s = competition_score(
            mem, current_event, B_xg, now,
            current_event.persons, mem.persons,
            current_event.tasks, mem.tasks,
        )
        scored.append((s, mem))
    scored.sort(key=lambda t: t[0], reverse=True)

    K = min(K_XG, len(scored))
    topk_pairs = scored[:K]
    topk = [mem for _, mem in topk_pairs]

    M = np.mean([mem.event_embedding for mem in topk], axis=0)
    M = M / (np.linalg.norm(M) + 1e-8)
    M_text = "；".join([f"记得{mem.event_id}（竞争分{s:.2f}）" for s, mem in topk_pairs])
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
