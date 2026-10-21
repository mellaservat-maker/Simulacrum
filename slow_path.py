# ============================================================
# slow_path.py · V4.2 · 模块5 慢通路
# 规格：5.5.8
# ============================================================

import numpy as np

from config import w_s, u_s, W_v, U_v, v_s, v_v, b_v, b_s, ETA, W_H, e_s, f_v, S_DIM


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S=None):
    # [012] S 为语义信号向量（长度 S_DIM）；缺省 None 时按全 0 处理，
    #   保持旧调用点行为恒等。e_s@S 进 z_s（危险评估），f_v@S 进 z_v（效价评估）。
    if S is None:
        S = np.zeros(S_DIM)
    z_s = (
        w_s @ (x + ReframedContext)
        + u_s @ M
        + v_s @ B
        - ETA * w_PFC  # [012 损耗型] w_PFC 是疲劳值，降低理智判断权重
        + W_H * HabituationSignal
        + b_s
        + e_s @ S  # [012] 语义信号通路（危险侧）
    )
    D_slow = float(sigmoid(z_s))

    z_v = W_v @ (x + ReframedContext) + U_v @ M + v_v * D_slow + b_v + f_v @ S  # [012] 语义信号通路（效价侧）
    V = sigmoid(z_v)
    return D_slow, V