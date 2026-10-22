# ============================================================
# feature_encoder.py · V4.2 · 特征编码
# 规格：5.5.1 / 5.5.2
# 输出：x (8维) + x_fast (4维)
# ============================================================

import numpy as np

import config


def safe_cosine(a, b):
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def punctuation_heuristic(raw_text):
    count = (raw_text.count("!") + raw_text.count("！") +
             raw_text.count("?") + raw_text.count("？"))
    return min(1.0, count / 3.0)


def pos_count(raw_text):
    return sum(raw_text.count(w) for w in config.POSITIVE_WORDS)


def neg_count(raw_text):
    return sum(raw_text.count(w) for w in config.NEGATIVE_WORDS)


def emotion_polarity(x_fast_1):
    return float(x_fast_1)


def person_overlap_ratio(current_event, memory_graph):
    if memory_graph.is_empty():
        return 0.0
    cur_persons = set(current_event.nodes.get("person", []))
    if not cur_persons:
        return 0.0
    mem_persons = set()
    for mem in memory_graph.get_all():
        mem_persons |= set(mem.nodes.get("person", []))
    if not mem_persons:
        return 0.0
    return len(cur_persons & mem_persons) / (len(cur_persons) + 1e-6)


def task_relevance(current_event):
    return 1.0 if current_event.nodes.get("task") else 0.0


def encode(event, M_prev, HabituationSignal_prev, B, memory_graph):
    raw_text = event.raw_text

    x_fast_0 = 1.0 if any(kw in raw_text for kw in config.DANGER_KEYWORDS) else 0.0
    x_fast_1 = (neg_count(raw_text) - pos_count(raw_text)) / (
        pos_count(raw_text) + neg_count(raw_text) + 1e-6
    )
    x_fast_2 = punctuation_heuristic(raw_text)
    x_fast_3 = 0.0
    x_fast = np.array([x_fast_0, x_fast_1, x_fast_2, x_fast_3])

    x0 = safe_cosine(event.event_embedding, M_prev)
    x1 = person_overlap_ratio(event, memory_graph)
    x2 = task_relevance(event)
    x3 = emotion_polarity(x_fast[1])
    x4 = event.event_severity
    x5 = HabituationSignal_prev
    x6 = B[1]
    x7 = B[2]
    x = np.array([x0, x1, x2, x3, x4, x5, x6, x7], dtype=float)

    return x, x_fast