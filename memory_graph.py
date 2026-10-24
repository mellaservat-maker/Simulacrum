# ============================================================
# memory_graph.py · V4.2 · M1 识海存储
# M1 只实现"存 / 取"，关联与遗忘留待 M2。
# [054] 新增 mean_vector()：会话内全部事件嵌入的均值基，
#   供 slow_path 做 M 的减均值基（去掉 BGE 各向异性的共模方向）。
#   用增量和维护，避免每轮 O(n) 重算。
# ============================================================

import numpy as np

from schemas import EventNode


class LayeredMemoryGraph:
    def __init__(self):
        self._memories = []
        self._emb_sum = None
        self._emb_n = 0

    def is_empty(self):
        return len(self._memories) == 0

    def get_all(self):
        return list(self._memories)

    def add(self, memory: EventNode):
        self._memories.append(memory)
        # [054] 增量维护嵌入均值：只累加与首个向量同维的嵌入
        emb = getattr(memory, "event_embedding", None)
        if emb is not None:
            try:
                e = np.asarray(emb, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                e = None
            if e is not None and e.size > 0:
                if self._emb_sum is None or self._emb_sum.shape != e.shape:
                    if self._emb_n == 0:
                        self._emb_sum = e.copy()
                        self._emb_n = 1
                else:
                    self._emb_sum += e
                    self._emb_n += 1
        return memory

    def mean_vector(self):
        """[054] 全部事件嵌入的均值基；无事件或维度不一致时返回 None。"""
        if self._emb_sum is None or self._emb_n == 0:
            return None
        return self._emb_sum / float(self._emb_n)

    def __len__(self):
        return len(self._memories)