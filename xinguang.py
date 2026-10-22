# ============================================================
# xinguang.py · V4.2 · 心光预算 + 心光竞争
# 规格：5.5.3 / 5.5.4 / 5.5.5
# ============================================================

import math

import numpy as np

import config


def update_xinguang_budget(B_energy, B_arousal, base_xg=None):
    if base_xg is None:
        base_xg = config.XG_BASE
    energy_factor = B_energy
    arousal_penalty = 1 - abs(B_arousal - 0.5) * 0.5
    B_xg = base_xg * energy_factor * arousal_penalty
    return float(np.clip(B_xg, 0.1, 1.0))


def retention(memory, current_time):
    dt_days = (current_time - memory.timestamp) / 86400.0
    decay_factor = memory.importance * 10 + memory.emotion_weight * 5 + 1e-6
    return math.exp(-dt_days / decay_factor) * memory.emotion_weight


def time_decay_weight(mem, now, tau_days=None):
    if tau_days is None:
        tau_days = config.TAU_DAYS
    dt_days = (now - mem.timestamp) / 86400.0
    return math.exp(-dt_days / tau_days)


def match_score(current_event_embedding, mem_embedding):
    if np.linalg.norm(current_event_embedding) < 1e-8 or np.linalg.norm(mem_embedding) < 1e-8:
        return 0.0
    return float(np.dot(current_event_embedding, mem_embedding) /
                 (np.linalg.norm(current_event_embedding) * np.linalg.norm(mem_embedding)))


def top_k(candidates, k, key):
    scored = [(key(c), c) for c in candidates]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(c, s) for s, c in scored[:k]]


def competition_score(mem, current_event, B_xg, now, cur_persons, mem_persons,
                      cur_tasks, mem_tasks):
    importance_term = mem.importance
    recency_term = time_decay_weight(mem, now)
    emotion_weight_term = mem.emotion_weight
    relation_weight_term = 0.0
    unfinished_term = mem.unfinished

    semantic_sim_term = match_score(current_event.event_embedding, mem.event_embedding)
    person_match_term = 1.0 if (cur_persons & mem_persons) else 0.0
    time_match_term = recency_term
    task_match_term = 1.0 if (cur_tasks & mem_tasks) else 0.0

    activation = (
        config.XG_W_ACT[0] * importance_term +
        config.XG_W_ACT[1] * recency_term +
        config.XG_W_ACT[2] * emotion_weight_term +
        config.XG_W_ACT[3] * relation_weight_term +
        config.XG_W_ACT[4] * unfinished_term
    )
    match_val = (
        config.XG_W_MATCH[0] * semantic_sim_term +
        config.XG_W_MATCH[1] * person_match_term +
        config.XG_W_MATCH[2] * time_match_term +
        config.XG_W_MATCH[3] * task_match_term
    )
    required_xg = config.XG_COMPLEXITY_BASE * activation
    resource_fit = min(1.0, B_xg / max(required_xg, 1e-6))
    return activation * match_val * resource_fit