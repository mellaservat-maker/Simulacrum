# ============================================================
# event_parser.py · V4.5 · 模块1 事件拆解
# 规格：5.5.1 / 5.7
# 输出：EventNode + event_embedding
# V4.5（T3-B）：嵌入改用embedder.get_embedder("bge")，1024 维语义嵌入
# M1：极简解析，只提取 person/task
# ============================================================

import hashlib
import time

import numpy as np

from schemas import EventNode
from embedder import get_embedder


def hash_to_8d(raw_text):
    """【已弃用】8 维 MD5 hash 嵌入，无语义。

    仅保留供 scripts/probe_embedder.py 向后兼容验证使用（它比对
    HashEmbedder 与本函数输出是否逐元素一致），主链路不再调用。
    """
    h = hashlib.md5(raw_text.encode()).digest()
    vec = np.array([h[i] / 255.0 for i in range(8)])
    return vec / (np.linalg.norm(vec) + 1e-8)


# 模块级单例：避免每次 parse 都重建 embedder（BGE 为懒加载，构造本身轻量）
_embedder = get_embedder("bge")


_PERSON_HINTS = ("你", "我", "他", "她", "它", "大家", "老板", "用户", "AI", "老师")
_TASK_HINTS = ("方案", "代码", "问题", "任务", "答案", "回复", "报告", "文档", "项目")


def _extract_persons(raw_text):
    return [p for p in _PERSON_HINTS if p in raw_text]


def _extract_tasks(raw_text):
    return [t for t in _TASK_HINTS if t in raw_text]


def parse(raw_text, event_id=None, timestamp=None,
          event_severity=0.0, importance=0.5,
          emotion_weight=0.5, unfinished=0.0,
          danger_score=0.0):
    if event_id is None:
        event_id = "E" + str(int(time.time() * 1000) % 1000000)
    if timestamp is None:
        timestamp = time.time()

    return EventNode(
        event_id=event_id,
        timestamp=timestamp,
        raw_text=raw_text,
        event_embedding=_embedder.encode(raw_text),
        nodes={
            "person": _extract_persons(raw_text),
            "task": _extract_tasks(raw_text),
            "topic": [],
            "state": [],
            "relation": [],
        },
        relationships=[],
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
        danger_score=danger_score,
        event_severity=event_severity,
    )