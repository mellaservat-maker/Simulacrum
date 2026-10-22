# ============================================================
# emotion_releaser.py · V4.2 · 模块7 情绪释放器
# 规格：5.5.9
# ============================================================

import numpy as np

import config


def run_releaser(V, D_fast, HabituationSignal, S_valence=0.0, S=None):
    # [012] S_valence 为语义信号的效价分量（-1~1）；c_E * S_valence 直接进
    #   raw_emotion_impact[0]，修复效价轴结构性恒负（V 出自随机 W_v/U_v 无语义、
    #   b_E[0]*D_fast 恒负，见 L24 / 011 草案问题五）。
    #   [013] S 改由 evaluator 从用户消息真提取（pipeline 两处链），
    #   "占位期硬编码 S=[0.8,...]" 的配置已废弃（其副作用见 012.1 第 4.3 节）。
    #   [030 四维有值化] 新增语义冲击 G_S @ S（唤醒/掌控线性，焦虑非线性合取）；
    #   S=None 时保持旧行为恒等（供旧探针直接调用）。G_S 冲击并入
    #   released_impact 的括号内，与 V/D_fast 共享同一释放率 (1-release_rate)。
    raw_emotion_impact = config.A_INIT @ V + config.b_E * D_fast
    raw_emotion_impact = np.asarray(raw_emotion_impact, dtype=float).copy()
    if S is not None:
        S = np.asarray(S, dtype=float)
        _sem = config.G_S @ S
        # 焦虑行：负效价 × 高唤醒 × 高强度 × 对方强势（低掌控）的合取；
        # 用合取而非线性加权，避免"夸得激动也算焦虑"。
        _s0, _s1, _s2, _s3 = (float(S[0]), float(S[1]), float(S[2]), float(S[3]))
        _sem[3] += (config.K_ANX
                    * max(0.0, -_s0)
                    * (0.5 + 0.5 * _s1)
                    * (0.5 + 0.5 * _s3)
                    * (0.5 + 0.5 * _s2))
        raw_emotion_impact += _sem
    raw_emotion_impact[0] += config.c_E * S_valence
    release_rate = float(np.clip(config.BASE_RELEASE_RATE * (1 + HabituationSignal), 0.0, 0.9))
    released_impact = raw_emotion_impact * (1 - release_rate)
    return raw_emotion_impact, released_impact, release_rate