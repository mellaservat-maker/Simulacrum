# ============================================================
# memory_gateway.py · 029 · 检索网关（第一阶段：编排，行为等价）
# 目标：把事件级（海马体找经历）与包级（读原文）收进同一个入口。
# 第一阶段只做编排，不合并两套打分逻辑、不改任何阈值：
#   - 事件级：直接调用 hippocampal_evaluator.evaluate（内部逻辑零改动）
#   - 包级：直接调用 pipeline._recall_match（内部逻辑零改动）
#   - 两路结果装进同一个 RecallResult 返回，下游各取所需。
# 依赖方向：pipeline -> memory_gateway -> hippocampal_evaluator；
#   对 pipeline._recall_match 采用函数内延迟导入，避免循环导入。
# ============================================================

from dataclasses import dataclass
from typing import Any, Dict, Optional

import config
from hippocampal_evaluator import evaluate as hippocampal_evaluate


@dataclass
class RecallResult:
    """一次检索的统一结果：事件级与包级各一份，不做分数合并。"""

    hippo: Dict[str, Any]
    recalled: Optional[Dict[str, Any]]

    # 事件级便捷字段（与旧 pipeline 局部变量一一对应）
    @property
    def historical_residual(self) -> float:
        return float(self.hippo.get("HistoricalResidual", 0.0))

    @property
    def habituation_signal(self) -> float:
        return float(self.hippo.get("HabituationSignal", 0.0))

    @property
    def reframed_context(self):
        return self.hippo.get("ReframedContext")

    @property
    def m_vec(self):
        return self.hippo.get("M")

    @property
    def m_text(self) -> str:
        return str(self.hippo.get("M_text", ""))

    @property
    def matched_events(self):
        return self.hippo.get("MatchedMemories", [])


def recall(query_text, event, memory_graph, x, B_xg, now, state,
           round_counter, now_valence=None):
    """统一检索入口。

    参数:
      query_text     本轮用户输入原文（包级匹配用）
      event          本轮 EventNode（事件级匹配用）
      memory_graph   识海（事件索引）
      x, B_xg, now   事件级评估所需上下文
      state          运行态（取 archiver/decompressor）
      round_counter  本轮轮次号（包龄计算用）
      now_valence    本轮 S 效价（包级极性过滤用）

    返回:
      RecallResult(hippo=事件级结果, recalled=包级结果或 None)

    行为约定（与 029 改造前逐位等价）：
      - 事件级每轮都算，结果结构与 hippocampal_evaluator.evaluate 完全一致。
      - 包级仅在 MEMORY_READ_ENABLED 且存在已归档包时调用；
        其余情况 recalled=None（与旧 pipeline 的 gating 相同）。
    """
    hippo = hippocampal_evaluate(event, memory_graph, x, B_xg, now)

    recalled = None
    if config.MEMORY_READ_ENABLED:
        packages = state["archiver"].get_packages()
        if packages:
            # 延迟导入：pipeline 在模块顶层导入本网关，此处调用时 pipeline
            # 已完整加载，避免循环导入。
            from pipeline import _recall_match
            recalled = _recall_match(state, packages, query_text, round_counter,
                                     now_valence=now_valence)

    return RecallResult(hippo=hippo, recalled=recalled)
