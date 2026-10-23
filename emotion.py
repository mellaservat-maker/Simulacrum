# ============================================================
# emotion.py · V4.2 · 模块8 情绪加点
# 规格：5.5.10
# ============================================================

import numpy as np

import config


def run_emotion(E_prev, released_impact, B):
    # [051 裁决] 向零回归项（OU）：- E_HOMEOSTASIS_GAMMA * E_prev。
    #   代数上等价于把有效保留系数从 0.9 降到 0.9-gamma，作用是压低
    #   「持续微小负反馈」的稳态偏移，并缩短停止输入后的尾巴。
    #   不改 clip 边界，不改 B 偏离耦合，不改其余三维。
    E_raw = np.clip(
        (config.LAMBDA - config.E_HOMEOSTASIS_GAMMA) * E_prev
        + released_impact
        + config.C_INIT @ (B - config.B_INIT),
        -1.0, 1.0,
    )
    return E_raw