# ============================================================
# emotion.py · V4.2 · 模块8 情绪加点
# 规格：5.5.10
# ============================================================

import numpy as np

import config


def run_emotion(E_prev, released_impact, B):
    E_raw = np.clip(
        config.LAMBDA * E_prev
        + released_impact
        + config.C_INIT @ (B - config.B_INIT),
        -1.0, 1.0,
    )
    return E_raw