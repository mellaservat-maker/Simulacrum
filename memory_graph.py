# ============================================================
# memory_graph.py · V4.2 · M1 识海存储
# M1 只实现"存 / 取"，关联与遗忘留待 M2。
# ============================================================

from schemas import EventNode


class LayeredMemoryGraph:
    def __init__(self):
        self._memories = []

    def is_empty(self):
        return len(self._memories) == 0

    def get_all(self):
        return list(self._memories)

    def add(self, memory: EventNode):
        self._memories.append(memory)
        return memory

    def __len__(self):
        return len(self._memories)