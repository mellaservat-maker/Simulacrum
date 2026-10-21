import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pipeline import run_scenario
from config import (A_INIT, b_E, LAMBDA, C_INIT, B_INIT, w_e, w_i,
                    THETA_ERROR, KAPPA, b_i, GAMMA)

def plan_c_e_final_0(V, D_fast, release_rate, B, E_prev, w_PFC):
    """方案 C：用 max(0, V[0]-V[1]) 替换 D_fast 作为危险门控"""
    danger_gate = max(0.0, V[0] - V[1])
    raw_impact_0 = A_INIT[0] @ V + b_E[0] * danger_gate
    released_0 = raw_impact_0 * (1 - release_rate)
    E_raw_0 = float(np.clip(LAMBDA[0]*E_prev[0] + released_0 + C_INIT[0] @ (B - B_INIT), -1.0, 1.0))
    E_raw = np.array([E_raw_0, 0.0, 0.0, 0.0])  # 其他维度不影响 I_inhibit 的 S_E（因为 w_e 均匀）
    # 实际上其他维度也变了，但只看 E_final[0]，且 I_inhibit 只依赖 S_E = w_e @ neg_intensity
    # 为了精确，需要完整 4 维。但方案 C 只改第 0 维，其他维不变
    # 从 log 拿完整 E_raw，只替换第 0 维
    neg = np.maximum(-E_raw, 0.0)
    S_E = float(np.clip(w_e @ neg, 0.0, 1.0))
    D_error = max(0.0, S_E - THETA_ERROR)
    I_inhibit = float(np.clip(w_i * D_error + KAPPA * w_PFC + b_i, 0.0, 1.0))
    E_final_0 = (1 - GAMMA * I_inhibit) * E_raw_0
    return E_final_0

for sc in ["被骂", "被夸", "被质疑"]:
    log = run_scenario(sc, feature_mode="fixture")
    V = np.array(log["V"])
    D_fast = log["D_fast"]
    release_rate = log["release_rate"]
    B = np.array(log["B"])
    E_prev = np.zeros(4)  # 首轮
    w_PFC = 0.0  # 首轮
    ef0 = plan_c_e_final_0(V, D_fast, release_rate, B, E_prev, w_PFC)
    print(f"{sc}   方案C E_final[0] = {ef0:.6f}")

# 判断
results = {}
for sc in ["被骂", "被夸", "被质疑"]:
    log = run_scenario(sc, feature_mode="fixture")
    V = np.array(log["V"])
    D_fast = log["D_fast"]
    release_rate = log["release_rate"]
    B = np.array(log["B"])
    ef0 = plan_c_e_final_0(V, D_fast, release_rate, B, np.zeros(4), 0.0)
    results[sc] = ef0

ma = results["被骂"]
kua = results["被夸"]
zhi = results["被质疑"]
print()
print(f"被夸 > 0? {kua > 0}")
print(f"被骂 < 0? {ma < 0}")
print(f"被质疑 < 0? {zhi < 0}")
print(f"三者两两不同? {len(set([round(ma,6), round(kua,6), round(zhi,6)])) == 3}")
