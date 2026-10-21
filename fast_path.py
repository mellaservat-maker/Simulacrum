# ============================================================
# fast_path.py · V4.2 · 模块4 快通路
# 规格：5.5.7
# ============================================================

import numpy as np

from config import w_f, b_f, THETA_FAST


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_fast_path(x_fast, B):
    B = np.array(B, dtype=float)
    D_fast = float(sigmoid(w_f @ x_fast + b_f))
    if D_fast > THETA_FAST:
        B[1] = np.clip(B[1] + 0.3, 0.0, 1.0)
    return D_fast, B