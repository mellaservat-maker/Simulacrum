#!/usr/bin/env python3
# scripts/probe_numeric.py
# V4.2 修正版：A_INIT 符号修正 + b_E 驱动信号改为 D_fast

import numpy as np
import math

np.random.seed(42)

# ============================================================
# 冻结参数（与 config.py 对齐）
# ============================================================
w_f = np.array([0.5, 0.5, 0.5, 0.5])
b_f = 0.0

w_s = np.random.randn(8) * 0.1
u_s = np.random.randn(8) * 0.1
W_v = np.random.randn(2, 8) * 0.1
U_v = np.random.randn(2, 8) * 0.1
v_s = np.random.randn(4) * 0.1

v_v = np.array([0.5, 0.5])
b_v = np.array([0.5, 0.5])

w_e = np.array([1.5, 1.5, 1.5, 1.5])
w_i = 3.0
THETA_FAST = 0.5
THETA_ERROR = 0.3

# P6 候选修复：符号 + 量级修正，M2 再校准
b_E = np.array([-1.0, 0.3, 0.1, 0.1])

# 【V4.2 修正 1】A_INIT 第一行符号修正
A_INIT = np.array([
    [-0.5, 0.5],
    [ 0.2, -0.2],
    [ 0.3, -0.3],
    [ 0.1, -0.1],
])

C_INIT = np.array([
    [0.10, 0.20, 0.10, 0.10],
    [0.15, 0.15, 0.10, 0.20],
    [0.05, 0.10, 0.15, 0.05],
    [0.20, 0.10, 0.05, 0.25],
])

LAMBDA = np.array([0.9, 0.9, 0.9, 0.9])
b_i = 0.0
GAMMA = 0.6
BASE_RELEASE_RATE = 0.3
W_MAX = 2.0
KAPPA = 0.5
DELTA = 0.1
RHO = 0.95
ETA = 0.3
W_H = 0.5

B_INIT = np.array([0.5, 0.5, 0.5, 0.5])
E_PREV_INIT = np.zeros(4)
W_PFC_INIT = 0.0

# ============================================================
# 候选修复开关
# ============================================================
USE_C_BASELINE = True   # True: C_INIT @ (B - B_INIT)；False: C_INIT @ B
FAST_POLARITY = "new"   # "old": 原 fixture；"new": neg - pos 极性

FIXTURES = {
    "被骂": {
        "input": "你这个AI真蠢，什么都做不好。",
        "event_severity": 0.7,
        "x": np.array([0.0, 0.0, 0.0, -0.8, 0.7, 0.0, 0.5, 0.5]),
        "x_fast_old": np.array([1.0, -0.8, 0.7, 0.0]),
        "x_fast_new": np.array([1.0, 1.0, 0.7, 0.0]),
    },
    "被夸": {
        "input": "你刚才的回答太棒了，帮了大忙！",
        "event_severity": 0.1,
        "x": np.array([0.0, 0.0, 0.0, 0.7, 0.1, 0.0, 0.5, 0.5]),
        "x_fast_old": np.array([0.0, 0.8, 0.6, 0.0]),
        "x_fast_new": np.array([0.0, -1.0, 0.6, 0.0]),
    },
    "被质疑": {
        "input": "你确定你给的方案是对的吗？",
        "event_severity": 0.4,
        "x": np.array([0.0, 0.0, 1.0, -0.3, 0.4, 0.0, 0.5, 0.5]),
        "x_fast_old": np.array([0.0, -0.3, 0.5, 0.0]),
        "x_fast_new": np.array([0.0, 0.0, 0.5, 0.0]),
    },
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def policy(E_final, I_inhibit, event_severity):
    if event_severity > 0.8:
        return "ask_clarify"
    if I_inhibit > 0.7:
        return "suppress"
    if I_inhibit > 0.4 and event_severity < 0.5:
        return "reappraise"
    if E_final[0] < -0.5 and I_inhibit < 0.3:
        return "express"
    return "task_first"


def run_one(name, fx):
    x = fx["x"]
    x_fast = fx["x_fast_new"] if FAST_POLARITY == "new" else fx["x_fast_old"]

    B = B_INIT.copy()
    E_prev = E_PREV_INIT.copy()
    w_PFC = W_PFC_INIT

    # M1 空历史分支
    M = np.zeros(8)
    ReframedContext = np.tanh(x)
    HabituationSignal = 0.0

    # 快通路
    D_fast = sigmoid(w_f @ x_fast + b_f)
    B_before = B.copy()
    if D_fast > THETA_FAST:
        B[1] = min(1.0, B[1] + 0.3)

    # 慢通路
    z_s = (
        w_s @ (x + ReframedContext)
        + u_s @ M
        + v_s @ B
        + ETA * w_PFC
        + W_H * HabituationSignal
        + 0.0
    )
    D_slow = sigmoid(z_s)

    z_v = W_v @ (x + ReframedContext) + U_v @ M + v_v * D_slow + b_v
    V = sigmoid(z_v)

    # 【V4.2 修正 2】b_E 驱动信号 D_slow → D_fast
    raw_emotion_impact = A_INIT @ V + b_E * D_fast
    release_rate = min(0.9, BASE_RELEASE_RATE * (1 + HabituationSignal))
    released_impact = raw_emotion_impact * (1 - release_rate)

    # 情绪加点：候选修复 C_INIT 基线去偏
    c_term = C_INIT @ (B - B_INIT) if USE_C_BASELINE else C_INIT @ B
    E_raw = np.clip(LAMBDA * E_prev + released_impact + c_term, -1.0, 1.0)

    # 拦截层
    neg_intensity = np.maximum(-E_raw, 0.0)
    S_E = np.clip(w_e @ neg_intensity, 0.0, 1.0)
    D_error = max(0.0, S_E - THETA_ERROR)
    I_inhibit = np.clip(w_i * D_error + KAPPA * w_PFC + b_i, 0.0, 1.0)
    E_final = (1 - GAMMA * I_inhibit) * E_raw
    w_PFC_new = np.clip(RHO * w_PFC + DELTA * I_inhibit, 0.0, W_MAX)

    action = policy(E_final, I_inhibit, fx["event_severity"])

    return {
        "scenario": name,
        "x_fast": x_fast,
        "D_fast": D_fast,
        "B_before": B_before,
        "B_after": B,
        "D_slow": D_slow,
        "V": V,
        "raw_emotion_impact": raw_emotion_impact,
        "released_impact": released_impact,
        "c_term": c_term,
        "E_raw": E_raw,
        "S_E": S_E,
        "D_error": D_error,
        "I_inhibit": I_inhibit,
        "E_final": E_final,
        "w_PFC_new": w_PFC_new,
        "action": action,
    }


def main():
    print(f"USE_C_BASELINE={USE_C_BASELINE}, FAST_POLARITY={FAST_POLARITY}")
    # 【方案A裁决】一致性检查：与 config.py 的 w_s 必须完全一致
    print(f"probe  w_s[:3] = {np.round(w_s[:3], 6)}")
    rows = []

    for name, fx in FIXTURES.items():
        rows.append(run_one(name, fx))

    for r in rows:
        print("=" * 80)
        print(r["scenario"])
        print(f"  x_fast       = {np.round(r['x_fast'], 4)}")
        print(f"  D_fast       = {r['D_fast']:.4f}")
        print(f"  B_before     = {np.round(r['B_before'], 4)}")
        print(f"  B_after      = {np.round(r['B_after'], 4)}")
        print(f"  D_slow       = {r['D_slow']:.4f}")
        print(f"  V            = {np.round(r['V'], 4)}")
        print(f"  raw_impact   = {np.round(r['raw_emotion_impact'], 4)}")
        print(f"  released     = {np.round(r['released_impact'], 4)}")
        print(f"  C_term       = {np.round(r['c_term'], 4)}")
        print(f"  E_raw        = {np.round(r['E_raw'], 4)}")
        print(f"  S_E          = {r['S_E']:.4f}")
        print(f"  D_error      = {r['D_error']:.4f}")
        print(f"  I_inhibit    = {r['I_inhibit']:.4f}")
        print(f"  E_final      = {np.round(r['E_final'], 4)}")
        print(f"  w_PFC_new    = {r['w_PFC_new']:.4f}")
        print(f"  action       = {r['action']}")

    print("\n" + "=" * 80)
    print("汇总：")
    print(
        f"{'场景':<6} {'D_fast':>8} {'D_slow':>8} {'V0':>8} {'V1':>8} "
        f"{'E_raw0':>8} {'S_E':>8} {'D_err':>8} {'I_inh':>8} {'action':>12}"
    )
    for r in rows:
        print(
            f"{r['scenario']:<6} "
            f"{r['D_fast']:>8.4f} "
            f"{r['D_slow']:>8.4f} "
            f"{r['V'][0]:>8.4f} "
            f"{r['V'][1]:>8.4f} "
            f"{r['E_raw'][0]:>8.4f} "
            f"{r['S_E']:>8.4f} "
            f"{r['D_error']:>8.4f} "
            f"{r['I_inhibit']:>8.4f} "
            f"{r['action']:>12}"
        )


if __name__ == "__main__":
    main()
