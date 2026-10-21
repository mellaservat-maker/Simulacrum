# ============================================================
# emotion_releaser.py · V4.2 · 模块7 情绪释放器
# 规格：5.5.9
# ============================================================

import numpy as np

from config import A_INIT, b_E, BASE_RELEASE_RATE, c_E


def run_releaser(V, D_fast, HabituationSignal, S_valence=0.0):
    # [012] S_valence 为语义信号的效价分量（-1~1）；c_E * S_valence 直接进
    #   raw_emotion_impact[0]，修复效价轴结构性恒负（V 出自随机 W_v/U_v 无语义、
    #   b_E[0]*D_fast 恒负，见 L24 / 011 草案问题五）。
    #   [013] S 改由 evaluator 从用户消息真提取（pipeline 两处链），
    #   "占位期硬编码 S=[0.8,...]" 的配置已废弃（其副作用见 012.1 第 4.3 节）。
    raw_emotion_impact = A_INIT @ V + b_E * D_fast
    raw_emotion_impact = np.asarray(raw_emotion_impact, dtype=float).copy()
    raw_emotion_impact[0] += c_E * S_valence
    release_rate = float(np.clip(BASE_RELEASE_RATE * (1 + HabituationSignal), 0.0, 0.9))
    released_impact = raw_emotion_impact * (1 - release_rate)
    return raw_emotion_impact, released_impact, release_rate