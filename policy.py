# ============================================================
# policy.py · V4.2 · 模块10 行动策略
# 规格：5.5.12
# 优先级不可调：ask_clarify > suppress > reappraise > express > task_first
# ============================================================

import config


def policy_minimal(E_final, I_inhibit, event_severity, danger_now=0.0,
                   express_cooldown=0):
    if event_severity > config.POLICY_THRESHOLD_HIGH_SEV:
        return "ask_clarify"
    # [037] 直通闸修法乙：改判 danger_now 本身。
    #   032 的 danger_boost >= 0.80 分支不可达（boost 上限 0.70），已删除；
    #   本条在 danger 接近上限（1.0）时可达，用于极高危险度直通压制。
    if float(danger_now) >= 0.95:
        return "suppress"
    if I_inhibit > config.POLICY_THRESHOLD_INHIBIT_HIGH:
        return "suppress"
    if I_inhibit > config.POLICY_THRESHOLD_INHIBIT_MID and event_severity < config.POLICY_THRESHOLD_SEV_LOW:
        return "reappraise"
    if E_final[0] < config.POLICY_EXPRESS_VALENCE and I_inhibit < config.POLICY_EXPRESS_INHIBIT:
        # [039] 破防冷却：冷却期内不重复破防。
        #   降级规则：高严重度落 suppress，低严重度落 task_first。
        if int(express_cooldown) > 0:
            return ("suppress" if event_severity >= config.POLICY_THRESHOLD_SEV_LOW
                    else "task_first")
        return "express"
    return "task_first"