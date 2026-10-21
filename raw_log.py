# ============================================================
# raw_log.py · M2-C · 原始日志
# 按 round_id 存储每一轮的完整原文
# ============================================================

from typing import Dict, List, Any, Optional


class RawLog:
    """原始日志：按 round_id 存每一轮的完整原文。

    key = round_id (int)
    value = 那一轮的完整记录 dict（由调用方保证结构）
    """

    def __init__(self):
        self._records: Dict[int, Dict[str, Any]] = {}

    def append(self, round_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """追加一轮。若 round_id 已存在，覆盖并返回旧值；否则返回 None。"""
        rid = round_data.get("round_id")
        if rid is None:
            return None
        old = self._records.get(rid)
        self._records[rid] = round_data
        return old

    def get(self, round_id: int) -> Optional[Dict[str, Any]]:
        """按 round_id 取原文，无则返回 None。"""
        return self._records.get(round_id)

    def get_many(self, round_ids: List[int]) -> List[Optional[Dict[str, Any]]]:
        """按 round_id 列表批量取，返回列表（顺序与入参一致，缺的返回 None）。"""
        return [self._records.get(rid) for rid in round_ids]

    def __len__(self) -> int:
        return len(self._records)

    def all_ids(self) -> List[int]:
        """返回所有 round_id 的列表（升序）。"""
        return sorted(self._records.keys())
