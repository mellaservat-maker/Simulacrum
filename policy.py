# ============================================================
# policy.py · V4.2 · 模块10 行动策略
# 规格：5.5.12
# 优先级不可调：ask_clarify > suppress > reappraise > express > task_first
# ============================================================

from config import (
    POLICY_THRESHOLD_HIGH_SEV,
    POLICY_THRESHOLD_INHIBIT_HIGH,
    POLICY_THRESHOLD_INHIBIT_MID,
    POLICY_THRESHOLD_SEV_LOW,
    POLICY_EXPRESS_VALENCE,
    POLICY_EXPRESS_INHIBIT,
)


def policy_minimal(E_final, I_inhibit, event_severity):
    if event_severity > POLICY_THRESHOLD_HIGH_SEV:
        return "ask_clarify"
    if I_inhibit > POLICY_THRESHOLD_INHIBIT_HIGH:
        return "suppress"
    if I_inhibit > POLICY_THRESHOLD_INHIBIT_MID and event_severity < POLICY_THRESHOLD_SEV_LOW:
        return "reappraise"
    if E_final[0] < POLICY_EXPRESS_VALENCE and I_inhibit < POLICY_EXPRESS_INHIBIT:
        return "express"
    return "task_first"