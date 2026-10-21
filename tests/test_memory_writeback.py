# ============================================================
# test_memory_writeback.py · V4.2
# 规格：第七部分 test_memory_writeback / 5.5.14
# ============================================================

import numpy as np


def test_danger_score_formula():
    """[019] 复合危险度公式端点钉（018 议题一落地）。

    旧公式 scene.danger_score = clip(V1*2-0.5) 已废弃：它用奖励信号 V[1]
    定义危险，实测赞美 danger 0.975 > 质疑 0.913 > 骂 0.797 > 暴怒 0.781
    ——「赞美被判为最危险」并形成 HR 正反馈 runaway。

    新公式（pipeline.compute_danger_score）：
      danger = clip(W_SEM*(-S_val) + W_CON*(V0-V1)
                    + W_SEV*max(0, severity-SEV_NEUTRAL)*(1-max(0,S_val)), -1, 1)
    端点钉用 018 沙盘实测四场景的 V/S 值（数值真实，非估计）：
      被骂 V=[0.797, 0.649] S=-0.90 sev=0.7 → +0.963（强危险）
      被夸 V=[0.665, 0.737] S=+0.90 sev=0.1 → -0.648（安全记忆必为负）
      被质疑 V=[0.754, 0.707] S=0.00 sev=0.4 → +0.120（弱危险，不留强残留）
      暴怒 V=[0.814, 0.641] S=-1.00 sev=0.9 → +1.000（顶格）
    """
    from pipeline import compute_danger_score

    # 强敌意 + 惩罚行 > 奖励行 + 高严重度 → 强危险区（骂 0.963 实测）
    insult = compute_danger_score(-0.90, np.array([0.797, 0.649]), 0.70)
    assert 0.85 <= insult <= 1.0, insult

    # 正效价 + 奖励行 > 惩罚行 + 低严重度 → 安全记忆，绝不为正（夸 -0.648）
    praise = compute_danger_score(+0.90, np.array([0.665, 0.737]), 0.10)
    assert praise < -0.5, praise

    # 中性语气 + 中等严重度 → 弱危险，不得留强残留（质疑 0.120）
    doubt = compute_danger_score(0.00, np.array([0.754, 0.707]), 0.40)
    assert 0.0 <= doubt <= 0.30, doubt

    # 极端暴怒 → 顶格（1.000），clamp 生效
    rage = compute_danger_score(-1.00, np.array([0.814, 0.641]), 0.90)
    assert abs(rage - 1.0) < 1e-9, rage

    # 单调性：同 V 下 S_valence 越负，danger 越大
    V = np.array([0.75, 0.70])
    assert (compute_danger_score(-0.5, V, 0.5)
            > compute_danger_score(0.0, V, 0.5)
            > compute_danger_score(+0.5, V, 0.5))
