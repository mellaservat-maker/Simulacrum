# ============================================================
# logger.py · V4.2 · 强制字段日志
# 格式：JSON Lines，pd.read_json(path, lines=True) 可读
# ============================================================

import json
import os

import numpy as np

REQUIRED_FIELDS = [
    "x", "x_fast", "B", "D_fast", "B_xg", "D_slow", "V",
    "raw_emotion_impact", "released_impact", "E_raw", "neg_intensity",
    "S_E", "D_error", "I_inhibit", "E_final", "w_PFC",
    "release_rate", "HabituationSignal", "HistoricalResidual",
    "M_text", "action", "feature_mode",
]


def _to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):  # 017.5：recall_detail 曾泄 np.bool_ 崩 json.dumps
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x


def write_log(path, log):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for f in REQUIRED_FIELDS:
        assert f in log, f"logger: missing required field '{f}'"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_py(log), ensure_ascii=False) + "\n")