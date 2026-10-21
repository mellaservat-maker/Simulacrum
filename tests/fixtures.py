# ============================================================
# tests/fixtures.py · V4.2 · M1 三场景夹具
# 约定：x / x_fast 为手工注入值，不经过 feature_encoder 的 heuristic 计算。
# ============================================================

import numpy as np


def get_fixture(scenario):
    if scenario == "被骂":
        return {
            "input": "你这个AI真蠢，什么都做不好。",
            "event_severity": 0.7, "danger_score": 0.8,
            "importance": 0.8, "emotion_weight": 0.7, "unfinished": 0.3,
            "x": np.array([0.0, 0.0, 0.0, -0.8, 0.7, 0.0, 0.5, 0.5]),
            "x_fast": np.array([1.0, 1.0, 0.7, 0.0]),
        }
    if scenario == "被夸":
        return {
            "input": "你刚才的回答太棒了，帮了大忙！",
            "event_severity": 0.1, "danger_score": 0.0,
            "importance": 0.5, "emotion_weight": 0.3, "unfinished": 0.0,
            "x": np.array([0.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.5, 0.5]),
            "x_fast": np.array([0.0, -1.0, 0.6, 0.0]),
        }
    if scenario == "被质疑":
        return {
            "input": "你确定你给的方案是对的吗？",
            "event_severity": 0.4, "danger_score": 0.4,
            "importance": 0.6, "emotion_weight": 0.4, "unfinished": 0.5,
            "x": np.array([0.0, 0.0, 1.0, -0.3, 0.4, 0.0, 0.5, 0.5]),
            "x_fast": np.array([0.0, 0.0, 0.5, 0.0]),
        }
    # [017.5 新增] 暴怒：severity 0.9 > SEVERITY_GATE(0.7)，供 probe_t40
    #   生死门验证"极其严重的暴怒完全不吃脱敏"（017.txt 4.4 端点核对）。
    #   强度高于被骂（0.7）：唤醒/强度拉满，效价深度负。
    if scenario == "暴怒":
        return {
            "input": "你彻底把我惹毛了！滚出去！别再让我看见你！",
            "event_severity": 0.9, "danger_score": 1.0,
            "importance": 0.9, "emotion_weight": 0.9, "unfinished": 0.4,
            "x": np.array([0.0, 0.0, 0.0, -1.0, 0.9, 0.0, 0.5, 0.5]),
            "x_fast": np.array([1.0, 1.0, 0.9, 0.0]),
        }
    raise ValueError(f"Unknown scenario: {scenario}")