# ============================================================
# short_term_buffer.py · M2-A · 短期缓冲区
# 存储每一轮原始记录，供未来归档/反思使用
# ============================================================

from typing import List, Dict, Any


class ShortTermBuffer:
    """短期缓冲区：按顺序存储每一轮的原始记录。

    每条记录字段（由调用方保证）：
      round_id     轮次号
      user_input   用户输入
      ai_output    AI 输出
      timestamp    时间戳（秒）
    """

    def __init__(self, default_keep: int = 2):
        self._records: List[Dict[str, Any]] = []
        self._default_keep = default_keep

    def append(self, round_data: Dict[str, Any]) -> None:
        """追加一轮记录。"""
        self._records.append(round_data)

    def get_all(self) -> List[Dict[str, Any]]:
        """返回全部记录（按时间顺序）。"""
        return list(self._records)

    def get_recent(self, n: int = None) -> List[Dict[str, Any]]:
        """返回最近 n 轮。n 缺省时用 default_keep。"""
        if n is None:
            n = self._default_keep
        return list(self._records[-n:]) if n > 0 else []

    def flush_keep_recent(self, n: int = None):
        """清空除最近 n 轮之外的所有内容，返回被清掉的部分。

        返回: (flushed_list, remaining_list)
        """
        if n is None:
            n = self._default_keep
        if n <= 0:
            flushed = list(self._records)
            self._records = []
            return flushed, []
        if n >= len(self._records):
            return [], list(self._records)
        flushed = self._records[:-n]
        remaining = self._records[-n:]
        self._records = remaining
        return list(flushed), list(remaining)

    def prepend_many(self, records) -> None:
        """把一批更早的记录插回队首（[053] 归档失败回灌用）。

        语义：这批记录比缓冲区现有内容更早，必须放回队首才能保持 round_id
        顺序；不做去重（round_id 唯一性由调用方保证）。
        """
        if not records:
            return
        self._records = list(records) + self._records

    def __len__(self) -> int:
        return len(self._records)
