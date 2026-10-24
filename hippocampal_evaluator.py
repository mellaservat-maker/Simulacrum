# ============================================================
# hippocampal_evaluator.py · V4.3 · 模块3 海马体
# 规格：5.5.6 + competition_score 接入（T19）
# 输出字段名严格为 M（禁止 CurrentThought_Projected）
# M1 三场景走空历史分支（L8）
# [055 任务2] HabituationSignal 从"计数器"改为"同类事件重复度"：
#   旧式 H = Σw/(Σw+1)，w 只有时间衰减，与会话内候选数成正比，
#   聊到第 3 轮就顶到 H_eff 封顶，导致"聊什么都一样麻木"。
#   新式只让与当前事件高度相似（sim > H_SIM_THRESHOLD）的历史计入。
# [055 任务3] 跨天：gap_days 让习惯化在空档期自然消退。
# [059 任务2] 事件级软阈值：MATCH_THRESHOLD 退居锚点（0.5 处 w=1/3），
#   准入线下探 MATCH_SOFT_MARGIN_LO（默认 0.45）、全权线上延
#   MATCH_SOFT_MARGIN_HI（默认 0.60）；带内候选按 w=clip((s-lo)/span)
#   温和参与 HR（mean(w×danger)）与 M（加权均值），topk 竞争不变。
#   带外（w=1）逐位退化为原硬阈值路径；习惯化层与空历史分支不动。
# ============================================================

import math

import numpy as np

import config
from config import m, K_XG, MATCH_THRESHOLD
from xinguang import match_score, time_decay_weight, competition_score


def _habituation(pairs, now, gap_days=0.0):
    """[055] 同类事件重复度 → HabituationSignal。

    参数:
      pairs     [(mem, sim), ...]，sim 为当前事件与该记忆的余弦
      now       当前时间（时间衰减用）
      gap_days  距上次会话的真实天数（跨天空档，0 = 连续会话）

    公式:
      w_sim = max(0, (sim - H_SIM_THRESHOLD) / (1 - H_SIM_THRESHOLD))
      w     = time_decay_weight(mem, now) × w_sim
      H     = Σw / (Σw + 1) × exp(-gap_days / TAU_GAP_DAYS_HABIT)
    语义:
      sim <= 阈值 → 贡献 0，不同话题不累积习惯化（这是修 bug 的核心）；
      sim = 1     → 贡献 1，完全同一件事，像旧的计数器一样累积；
      空档期      → 整体衰减，三天没聊麻木感归零。
    """
    thr = float(getattr(config, "H_SIM_THRESHOLD", 0.7))
    span = max(1e-6, 1.0 - thr)
    weighted = 0.0
    for mem, sim in pairs:
        if sim <= thr:
            continue
        w_sim = (float(sim) - thr) / span
        weighted += float(time_decay_weight(mem, now)) * w_sim

    gap = float(gap_days or 0.0)
    tau_gap = float(getattr(config, "TAU_GAP_DAYS_HABIT", 0.0) or 0.0)
    if gap > 0.0 and tau_gap > 0.0:
        weighted *= math.exp(-gap / tau_gap)

    return float(weighted / (weighted + 1.0))


def evaluate(current_event, memory_graph, x, B_xg, now, gap_days=0.0):
    """[059 任务2] 软阈值候选准入与加权。

    硬阈值时代：s > MATCH_THRESHOLD 二元闸门——0.50 附近的边缘记忆
    全有或全无，阈值微移即翻转（058 探针实测黄金轨迹 max 余弦
    0.547~0.677 密集落在 [0.5,0.6)，第 8 轮 max=0.495 更是擦边）。

    软阈值（059）：MATCH_THRESHOLD 退居锚点，软化带相对它定义——
      soft_lo = MATCH_THRESHOLD - MATCH_SOFT_MARGIN_LO（默认 0.45）
      soft_hi = MATCH_THRESHOLD + MATCH_SOFT_MARGIN_HI（默认 0.60）
      w = clip((s - soft_lo) / (soft_hi - soft_lo), 0, 1)
    准入 s > soft_lo；带内候选以 w 温和参与：
      HistoricalResidual = mean(w_i × danger_i)（所有者指定口径）
      M = normalize(Σ w_i·emb_i / Σ w_i)（topk 仍按 competition_score 竞争）
    带外语义保持旧观：s >= soft_hi 时 w=1，公式逐一退化为原硬阈值结果。
    不动：习惯化层（H_SIM_THRESHOLD=0.7，[soft_lo, thr) 候选贡献恒 0）、
    空历史分支、M_text 文案、topk 竞争分排序。
    MatchedMemories 增加 soft_weight 观测字段（additive，消费方安全）。
    """
    # [059 任务2] 软化带相对锚点计算（模块级 MATCH_THRESHOLD 可被探针
    #   覆盖，带随锚点平移——probe_t44 阈值扫描语义不变）
    thr = float(MATCH_THRESHOLD)
    soft_lo = thr - float(config.MATCH_SOFT_MARGIN_LO)
    soft_hi = thr + float(config.MATCH_SOFT_MARGIN_HI)
    span = max(1e-6, soft_hi - soft_lo)

    scored_candidates = []
    for mem in memory_graph.get_all():
        s = match_score(current_event.event_embedding, mem.event_embedding)
        if s > soft_lo:
            w = float(np.clip((float(s) - soft_lo) / span, 0.0, 1.0))
            scored_candidates.append((mem, s, w))

    if not scored_candidates:
        return {
            "HistoricalResidual": 0.0,
            "HabituationSignal": 0.0,
            "ReframedContext": np.tanh(x),
            "M": np.zeros(m),
            "M_text": "(无历史记忆)",
            "B_xg": B_xg,
            "MatchedMemories": [],
        }

    candidates = [mem for mem, _, _ in scored_candidates]
    # [059 任务2] HR 软化口径：每个候选的危险度先乘软化系数再取平均
    #   （带内 w<1 → 边缘记忆对历史示警的贡献被温和衰减；带外 w=1 → 原值）
    HistoricalResidual = float(np.mean(
        [w * float(mem.danger_score) for mem, _, w in scored_candidates]))
    # [055] 习惯化只认"同类事件"：相似度不够的历史不计入重复度
    #   （_habituation 自带 H_SIM_THRESHOLD=0.7 硬闸，软阈值放宽的
    #    [soft_lo, thr) 候选对其贡献恒 0——两层阈值互不干扰）
    HabituationSignal = _habituation(
        [(mem, s) for mem, s, _ in scored_candidates], now, gap_days=gap_days)

    scored = []
    for mem in candidates:
        s = competition_score(
            mem, current_event, B_xg, now,
            current_event.persons, mem.persons,
            current_event.tasks, mem.tasks,
        )
        scored.append((s, mem))
    scored.sort(key=lambda t: t[0], reverse=True)

    K = min(K_XG, len(scored))
    topk_pairs = scored[:K]
    topk = [mem for _, mem in topk_pairs]

    # [059 任务2] M 软化口径：topk 内按 w 加权平均再归一化
    #   （w 全 1 时走原 np.mean 路径，逐位退化为硬阈值结果）
    w_of = {mem.event_id: w for mem, _, w in scored_candidates}
    w_top = np.array([w_of[mem.event_id] for mem in topk], dtype=float)
    w_sum = float(np.sum(w_top))
    if w_sum <= 0.0 or bool(np.all(w_top == 1.0)):
        M = np.mean([mem.event_embedding for mem in topk], axis=0)
    else:
        M = (np.sum([float(w) * mem.event_embedding
                     for w, (_, mem) in zip(w_top, topk_pairs)], axis=0)
             / w_sum)
    M = M / (np.linalg.norm(M) + 1e-8)
    M_text = "；".join([f"记得{mem.event_id}（竞争分{s:.2f}）" for s, mem in topk_pairs])
    ReframedContext = np.tanh(x + HistoricalResidual + HabituationSignal)

    return {
        "HistoricalResidual": HistoricalResidual,
        "HabituationSignal": HabituationSignal,
        "ReframedContext": ReframedContext,
        "M": M,
        "M_text": M_text,
        "B_xg": B_xg,
        "MatchedMemories": [
            {"event_id": mem.event_id, "score": s, "residual": mem.danger_score,
             # [059 任务2] 软化系数观测字段（additive，不参与既有计算）
             "soft_weight": float(w_of[mem.event_id])}
            for s, mem in topk_pairs
        ],
    }
