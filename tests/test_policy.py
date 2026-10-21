# ============================================================
# test_policy.py · V4.2
# 规格：第七部分 test_policy
# 优先级不可调：ask_clarify > suppress > reappraise > express > task_first
# ============================================================

import numpy as np

from policy import policy_minimal


def test_policy_priority():
    assert policy_minimal(np.zeros(4), 0.0, 0.9) == "ask_clarify"
    assert policy_minimal(np.zeros(4), 0.8, 0.5) == "suppress"
    assert policy_minimal(np.zeros(4), 0.5, 0.3) == "reappraise"
    assert policy_minimal(np.array([-0.6, 0, 0, 0]), 0.1, 0.3) == "express"
    assert policy_minimal(np.zeros(4), 0.0, 0.3) == "task_first"