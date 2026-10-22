# ============================================================
# schemas.py · V4.2 · M1 数据契约
# ============================================================

from dataclasses import dataclass, field
from typing import Any, List, Dict, TypedDict

import numpy as np


@dataclass
class EventNode:
    event_id: str
    timestamp: float
    raw_text: str
    event_embedding: np.ndarray
    nodes: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Any] = field(default_factory=list)
    importance: float = 0.0
    emotion_weight: float = 0.0
    unfinished: float = 0.0
    persons: set = field(default_factory=set)
    tasks: set = field(default_factory=set)
    danger_score: float = 0.0
    event_severity: float = 0.0


@dataclass
class MemoryPackage:
    # 数值层（系统用）
    seq: int = 0
    emotion_value: float = 0.0
    emotion_polarity: str = "neutral"
    keywords: List[str] = field(default_factory=list)
    raw_pointer: str = ""

    # [017.5 T31] 包级双向量索引（016 方案C + 017 雷点1补丁(2)）：
    #   mean_vec = 包内各轮 user_input embedding 的均值（归一化）——保主题；
    #   peak_vec = 包内 intensity 最高轮的单轮 embedding——保强度
    #     （intensity 信息缺失时退化为 mean_vec，见 archiver._compute_pkg_vectors）。
    #   读取侧 sim = max(cos_now_mean, cos_now_peak)，全包 argmax。
    #   默认 None 兼容旧包（无嵌入包不参与匹配，读取侧退回 latest 路径）。
    mean_vec: np.ndarray = None
    peak_vec: np.ndarray = None

    # [019] 包级整体情绪极性（018 议题二·极性一致性过滤用）：
    #   mean_valence = 包内各轮 S_valence 的均值——代表「这个包整体上
    #   是 positive 还是 negative」；不用峰值轮 valence：混合包里极端
    #   强度轮（intensity 最高的骂轮）会绑架峰值判据，使 3 夸 1 骂的包
    #   被判成负极性而逃过过滤（018 沙盘实测漏洞）。
    #   旧包（无 valence 记录）记 None —— 读取侧对 None 不过滤（无信息不惩罚）。
    mean_valence: float = None

    # [029] 包级危险度聚合（写入侧原料，供后续 HR 反推与去重口径使用）：
    #   n_rounds   包内轮数（含全部记录）
    #   mean_danger 包内各轮 danger_score 均值（无信息 → None）
    #   max_danger  包内各轮 danger_score 最大值（无信息 → None）
    #   旧包缺这些字段时按默认值读取，读取侧不得因此改判。
    n_rounds: int = 0
    mean_danger: float = None
    max_danger: float = None

    # 叙事层（AI 读，分标题）
    narrative_cause: str = ""
    narrative_trigger: str = ""
    narrative_consequence: str = ""
    narrative_evaluation: str = ""
    narrative_emotion: str = ""
    narrative_thought: str = ""


class HippocampalOutput(TypedDict):
    HistoricalResidual: float
    HabituationSignal: float
    ReframedContext: Any
    M: Any
    M_text: str
    B_xg: float
    MatchedMemories: List[Any]


class InternalStateLog(TypedDict):
    x: Any
    x_fast: Any
    B: Any
    D_fast: float
    B_xg: float
    D_slow: float
    V: Any
    raw_emotion_impact: Any
    released_impact: Any
    E_raw: Any
    neg_intensity: Any
    S_E: float
    D_error: float
    I_inhibit: float
    E_final: Any
    w_PFC: float
    release_rate: float
    HabituationSignal: float
    HistoricalResidual: float
    M_text: str
    action: str
    feature_mode: str