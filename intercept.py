# ============================================================
# intercept.py · V4.2 · 模块9 拦截层
# 规格：5.5.11
# ============================================================

import numpy as np

import config


def run_intercept(E_raw, w_PFC, danger_boost=0.0):
    neg_intensity = np.maximum(-E_raw, 0.0)
    S_E = float(np.clip(config.w_e @ neg_intensity, 0.0, 1.0))
    # [032] 危险度接入自控：D_error = max(0, S_E-THETA_ERROR) + danger_boost×W_DANGER
    #   danger_boost 为本轮 danger_score（性格五项 + danger_gain 已生效）。
    #   说明：danger_boost 为负（被夸/安全事件）时该项可为负，使 D_error 下探；
    #   下游 I_inhibit 仍 clip 到 [0,1]，不会产生负自控。
    D_error = (max(0.0, S_E - config.THETA_ERROR)
               + float(danger_boost) * config.W_DANGER)
    # [032 损耗型] w_PFC 是被消耗的意志力疲劳值：累积越高抑制防线越脆弱。
    # 009.5 及以前为 + KAPPA * w_PFC（越压抑越能压的"修为"叙事，与 004/005
    #   文档记述的"自我损耗"相反）；012 号工作按所有者裁决 B 改为减号。
    # KAPPA 已由所有者逐轮上调：0.2（012 敏感度）→ 0.5（014 相变临界）→
    #   0.6（014.5，渐近侵蚀更深；不动点分析见 config.py 注释）。
    # [033→037] 自控衰竭系数已在 037 撤回，W_PFC_FACTOR 固定 1.0，
    #   公式恢复为原始疲劳侵蚀 -KAPPA*w_PFC（保留乘子只为 t49 还原断言）。
    I_inhibit = float(np.clip(config.w_i * D_error
                              - config.KAPPA * w_PFC * config.W_PFC_FACTOR
                              + config.b_i, 0.0, 1.0))
    E_final = (1 - config.GAMMA * I_inhibit) * E_raw
    # [014.5 疲劳代谢] 区分场景的疲劳代谢（所有者裁决：不能无条件降疲劳，
    #   否则"越骂越清醒"变受气包）：
    #   安全轮（D_error < 阈值 或 I_inhibit == 0）：疲劳自然消退，每轮
    #       max(0, w_PFC - W_PFC_RECOVERY_RATE)；
    #   受压轮（I_inhibit > 0 且 D_error >= 阈值）：按原损耗累积逻辑走，
    #       绝不降低——挨骂期间疲劳只涨不跌。
    if D_error < config.W_PFC_RECOVERY_THRESHOLD or I_inhibit == 0.0:
        w_PFC_new = max(0.0, w_PFC - config.W_PFC_RECOVERY_RATE)
    else:
        w_PFC_new = float(np.clip(config.RHO * w_PFC + config.DELTA * I_inhibit,
                                  0.0, config.W_MAX))
    return neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new


# ============================================================
# [017.5 T31] 召回拦截层：威胁汇流 + 迟滞模式机 + 动态方向公式
# 蓝图：017.txt 4.2-4.3（所有者拍板的修订版最终形态）
#   threat_now = max(D_error, HistoricalResidual, -S_valence)
#   H_eff = min(sqrt(H), H_CAP)；severity > SEVERITY_GATE 时脱敏项清零
#   迟滞：threat_now > THETA_UP → alpha_raw=1（PTSD）；< THETA_DOWN → alpha_raw=0
#         （脱敏）；缓冲带内保持前值；alpha_t = KEEP*alpha_{t-1}+NEW*alpha_raw_t
#   脱敏生效需 H_eff >= H_EVIDENCE_GATE 且命中包在最近 RECALL_TIME_WINDOW 轮内
#   delta = clip(ev*sign*(alpha + (1-alpha)*(-H_eff)), -1.0, CAP_POS)
#   一致性护栏：HR > HR_GUARD_THRESHOLD 且注入与 HR 语义反号且幅值超过
#     HR_GUARD_MAGNITUDE → 注入幅值减半（先减半后截断）。
#     语义说明：HR 正=历史示警，注入正=情绪示好——数学同号即语义反号；
#     护栏允许小幅"嘴硬心软"（<=0.3），禁止情绪通道反向大喊（>0.3 减半）。
#   严重度门第二闸（017.5 生死门补丁）：severity > SEVERITY_GATE 且输出为
#     正 → 归零。第一闸清零脱敏项只管 factor；sign 来自召回包，包极性与
#     严重事件背离时（暴怒输入匹配到夸包）第一闸拦不住正值，故输出端否决。
# 端点自检（017.txt 4.4）：
#   纯 PTSD：alpha=1 → ev×sign（与 012 旧公式向后兼容）
#   纯脱敏：alpha=0、H_eff=0.4、ev=0.8 负向 → +0.32 → cap 到 +0.3（淡定有顶）
#   PUA 早期：H=0.1 → H_eff=0.32 < 0.4 → 脱敏不生效 → factor=alpha，低威胁 → 0
#   严重威胁：severity>0.7 → 脱敏清零 → 只许放大，不许麻木
# ============================================================
def update_threat_mode(threat_now, alpha_prev):
    """迟滞模式机 + 一阶低通（017 雷点4补丁）。每轮演进，与是否命中无关。

    返回: (alpha_new, alpha_raw)
      alpha_raw 本轮原始模式（1=PTSD / 0=脱敏 / 缓冲带=保持前值）
      alpha_new 低通平滑后的模式状态（供注入公式消费并回写跨轮状态）
    """
    threat_now = float(threat_now)
    alpha_prev = float(alpha_prev)
    if threat_now > config.THETA_UP:
        alpha_raw = 1.0
    elif threat_now < config.THETA_DOWN:
        alpha_raw = 0.0
    else:
        alpha_raw = alpha_prev  # 缓冲带：记仇但不动
    alpha_new = (config.THREAT_LOWPASS_KEEP * alpha_prev
                 + config.THREAT_LOWPASS_NEW * alpha_raw)
    return alpha_new, alpha_raw


def run_recall_intercept(ev, sign, H, alpha, event_severity,
                         pkg_age_rounds=None, historical_residual=0.0):
    """计算本轮召回注入量（017.txt 4.3 方向公式，全部补丁在身）。

    参数:
      ev                   命中轮级新鲜情绪强度（0~1，evaluator 重提的 intensity）
      sign                 方向（+1/-1/0，0 = neutral 不注入）
      H                    海马体习惯化信号 HabituationSignal（0~1）
      alpha                本轮迟滞模式值（pipeline 经 update_threat_mode 算好）
      event_severity       当前事件严重度（fixture 直传字段）
      pkg_age_rounds       命中包轮龄（当前轮 - 包内最大 round_id）；
                           None = 年龄未知，按"最近"处理（不因未知而冤枉脱敏）
      historical_residual  HR（一致性护栏用）

    返回:
      (delta, detail)
      delta   截断后的实际注入量（[-1.0, CAP_POS]）
      detail  公式中间量（观测/日志用，不参与计算）
    """
    ev = float(ev)
    sign = float(sign)
    H = float(H)
    alpha = float(alpha)
    event_severity = float(event_severity)

    # H_eff 软上限（017 雷点5补丁(1)：sqrt 凹变换 + H_CAP 硬顶）
    # 显式 float()：np.sqrt 返回 np.float64，若不转原生类型会随 detail
    # 泄漏进 logger 的 json.dumps（np.bool_ 不可序列化，017.5 生死门实测崩过）
    H_eff = float(min(np.sqrt(max(H, 0.0)), config.H_CAP))

    # 脱敏生效条件（017 雷点5补丁(2) 严重度门 + 雷点6补丁(2)(3) 证据门与时间窗）
    recent = (pkg_age_rounds is None) or (float(pkg_age_rounds) <= config.RECALL_TIME_WINDOW)
    desens_active = bool((H_eff >= config.H_EVIDENCE_GATE) and bool(recent)
                         and (event_severity <= config.SEVERITY_GATE))
    if desens_active:
        factor = alpha + (1.0 - alpha) * (-H_eff)
    else:
        factor = alpha  # 脱敏项不生效：只保留 PTSD 项（017 4.4 端点核对）

    # 原始注入 + 一致性护栏（先减半后截断）
    delta_raw = float(ev * sign * factor)
    hr = float(historical_residual)
    guard_applied = False
    if (abs(hr) > config.HR_GUARD_THRESHOLD
            and abs(delta_raw) > config.HR_GUARD_MAGNITUDE
            and ((hr > 0 and delta_raw > 0) or (hr < 0 and delta_raw < 0))):
        delta_raw *= 0.5
        guard_applied = True

    # 非对称 clip（017 雷点6补丁(1)：伤痛不限，淡定封顶）
    delta = float(np.clip(delta_raw, -1.0, config.CAP_POS))

    # 严重度门第二闸（017.5 生死门实测补丁）：severity > SEVERITY_GATE 时
    #   严禁正值注入——017.txt 4.4「只许放大，不许麻木」的输出端兑现。
    #   第一闸（脱敏项清零）只管得住 factor，管不住 sign 来源：命中的包
    #   极性与当前严重事件背离时（生死门实测第 29 轮：暴怒输入匹配到夸包，
    #   sim=0.418 > TAU_FLOOR=0.15，极性 positive），raw=ev×sign 仍是正的，
    #   = 最惨的时刻心情变好，与严重度门的存在意义矛盾。故在输出端加最终
    #   否决：正值一律归零；负向放大（ev×sign 本身为负）不受影响。
    severity_positive_banned = bool(event_severity > config.SEVERITY_GATE and delta > 0.0)
    if severity_positive_banned:
        delta = 0.0

    detail = {
        "H_eff": float(H_eff),
        "alpha": float(alpha),
        "factor": float(factor),
        "desens_active": bool(desens_active),
        "recent": bool(recent),
        "delta_after_guard": float(delta_raw),
        "guard_applied": bool(guard_applied),
        "delta": float(delta),
        "severity_positive_banned": severity_positive_banned,
    }
    return delta, detail