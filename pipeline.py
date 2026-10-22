# ============================================================
# pipeline.py · V4.2 · 完整链路
# run_scenario(scenario, feature_mode)
# 模式：
#   fixture - 从 tests/fixtures.py 直接读 x/x_fast（Step 1/6 验收用，断言 action）
#   encoder - 完整链路冒烟（只检查无异常、字段完整，不断言 action）
# ============================================================

import os
import sys
import time

import numpy as np

from config import (m, MEMORY_WRITE_ENABLED, MEMORY_READ_ENABLED,
                    RECALL_MATCH_ENABLED, TAU_FLOOR, PACKAGE_EMBEDDER,
                    W_SEM, W_CON, W_SEV, SEV_NEUTRAL,
                    POLARITY_FILTER_ENABLED, THETA_RAGE, THETA_POS,
                    THETA_MILD, LAMBDA_POL)
from state import reset_state
from fast_path import run_fast_path
from slow_path import run_slow_path
from emotion_releaser import run_releaser
from emotion import run_emotion
from intercept import run_intercept, run_recall_intercept, update_threat_mode
from policy import policy_minimal
from xinguang import update_xinguang_budget
from hippocampal_evaluator import evaluate as hippocampal_evaluate
from event_parser import parse as event_parse
from feature_encoder import encode as feature_encode
from state_translator import state_to_prompt
from prompt_builder import build_prompt
from llm_client import get_llm
from evaluator import extract_semantic_signals
from embedder import get_embedder
from logger import write_log
from tests.fixtures import get_fixture

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILES = {
    "被骂": "run_insult.jsonl",
    "被夸": "run_praise.jsonl",
    "被质疑": "run_doubt.jsonl",
    "暴怒": "run_rage.jsonl",
}

# [017.5 T31] 包级匹配嵌入器：与 event_parser/archiver/decompressor 同款
#   （embedder 实例缓存保证全系统共用一个 BGE 模型）
_pkg_embedder = get_embedder(PACKAGE_EMBEDDER)


def _cos(a, b) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_danger_score(S_valence, V, event_severity):
    """[019] 复合危险度公式（018 议题一沙盘实证后落地，替代 V[1]*2-0.5）。

    病根：旧公式 scene.danger_score = clip(V[1]*2-0.5) 用奖励信号 V[1]
      定义危险，实测赞美 danger 0.975 > 质疑 0.913 > 骂 0.797 > 暴怒
      0.781——「赞美被判为最危险」，并形成 HR 正反馈 runaway。

    公式（参数见 config [019 施工] 块，全部 [未校准]）：
      danger = clip( W_SEM * (-S_valence)                语义厌恶锚
                   + W_CON * (V[0] - V[1])               奖惩对比
                   + W_SEV * max(0, severity - SEV_NEUTRAL)
                            * (1 - max(0, S_valence)),   严重度门
                   -1, 1 )
    三项语义：
      语义厌恶锚——主观「当时多难受」，透明可解释，负价推高危险；
      奖惩对比——慢通路内部惩罚/奖励两行相减（共模抑制），代表身体层
        受伤程度，对「中性语气的重话」敏感；
      严重度门——只罚中高严重度事件，且正效价事件几乎不吃
        （乘 1-max(0,S_valence)），防「笑着说的严重事」被洗白。

    沙盘实测（018.txt 第三节）：暴怒 +1.000 / 被骂 +0.963 /
      被质疑 +0.120 / 被夸 -0.648——单调性与语义全部正确。
    自稳定性：夸段 HR 由 +0.65 翻负（沙盘 [-0.650,-0.644]），
      threat_now 汇流口干净，正反馈 runaway 变负反馈自稳。

    降级：入参一律 float() 强转；无异常路径（纯算术，不抛）。
    """
    sv = float(S_valence)
    v0 = float(V[0])
    v1 = float(V[1])
    sev = float(event_severity)
    sem = -sv
    con = v0 - v1
    sev_gate = max(0.0, sev - SEV_NEUTRAL) * (1.0 - max(0.0, sv))
    return float(np.clip(W_SEM * sem + W_CON * con + W_SEV * sev_gate,
                         -1.0, 1.0))


def _recall_match(st, packages, input_text, current_round, now_valence=None):
    """[017.5 T31] 包级相关性匹配（017.txt 4.1 检索层落地）。
    [019] + 情绪极性一致性过滤（018 议题二，不对称设计，沙盘实证后落地）。

    sim = max(cos_now_mean, cos_now_peak)，全包 argmax；sim < TAU_FLOOR
    不召回（返回 None）；命中 → decompressor 轮级回捞 + evaluator 现场重提 S。
    降级（不崩、不静默）：嵌入失败或无带嵌入的包（旧存档）→ 退回
    packages[-1] 的旧路径；开关 RECALL_MATCH_ENABLED=False 同样退回旧路径。

    [019] 极性一致性过滤（POLARITY_FILTER_ENABLED=True 时生效）：
      不对称设计（仿生学依据 mood-congruent retrieval——情绪状态本身是
      检索的偏置信号，威胁态下大脑倾向召回负性材料），硬过滤下沉轮级：
        包级（本函数）：威胁区 now_val < -THETA_MILD 对 positive 包
          （mean_valence > 0）软惩罚 sim_eff = sim x (1 - LAMBDA_POL*pv*(-now_val))；
          不硬滤——混合包（3 夸 1 骂）里还有黑历史，存活可能必须保留；
        轮级（decompressor.for_emotion_evaluator_matched，同参同构）：
          暴怒区 now_val < -THETA_RAGE：valence > THETA_POS 的轮硬过滤；
          轻度威胁区：positive 轮软惩罚（同公式）；包内轮全被滤 →
          recall_source="polarity_exhausted" → 本函数返回 None（暴怒时
          这个包里没有黑历史，静默好过召回夸包）。
        安全区/中性区 now_val >= -THETA_MILD：两级都不惩罚——正当前态绝不过滤
          负极性包（平安时召回旧伤 → H_eff 翻正，是脱敏通道命根子）；
          中性区不过滤（保 014 质疑校准，valence≈0）。
      包级判据用 mean_valence（各轮 valence 均值）而非峰值轮：混合包里
         intensity 最高的骂轮会绑架峰值判据（3 夸 1 骂包被判负极性逃逸，
        018 沙盘实测漏洞）。
      旧包 mean_valence=None（无极性信息）：不过滤。
      无带嵌入的包 → 退回 latest 旧路径（不静默）。
    """
    if not RECALL_MATCH_ENABLED:
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    try:
        now_vec = _pkg_embedder.encode(input_text)
    except Exception as e:
        print(f"[pipeline] 告警：包级匹配嵌入失败（{type(e).__name__}: {e}），"
              f"退回 latest 旧路径", file=sys.stderr)
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    best, best_sim = None, -1.0
    best_raw_sim, best_pen = -1.0, 0.0
    had_candidates = False
    for pkg in packages:
        mv = getattr(pkg, "mean_vec", None)
        if mv is None or len(mv) == 0 or len(mv) != len(now_vec):
            continue  # 旧包（无嵌入）或维度不符：不参与 argmax
        had_candidates = True
        s_m = _cos(now_vec, mv)
        pv = getattr(pkg, "peak_vec", None)
        s_p = _cos(now_vec, pv) if (pv is not None and len(pv) == len(now_vec)) else -1.0
        sim = max(s_m, s_p)
        # [019] 极性一致性过滤（不对称）——包级只软惩罚，硬过滤下沉轮级
        #   （018 实证修正：包级硬过滤会把 3 夸 1 骂的混合包也滤掉，可它
        #   明明含骂轮——被骂早期语料里唯一的包正是这种，硬滤导致无召回、
        #   B2「暴怒时被放大」失败。混合包的存活可能必须保留到轮级。）
        sim_eff, pen = sim, 0.0
        if POLARITY_FILTER_ENABLED and now_valence is not None:
            pkg_pv = getattr(pkg, "mean_valence", None)
            if pkg_pv is not None:
                nv = float(now_valence)
                pvv = float(pkg_pv)
                if nv < -THETA_MILD and pvv > 0.0:
                    pen = LAMBDA_POL * pvv * (-nv)   # 威胁区（含暴怒区）软惩罚
                    sim_eff = sim * (1.0 - min(1.0, pen))
        if sim_eff > best_sim:
            best, best_sim = pkg, sim_eff
            best_raw_sim, best_pen = sim, pen
    if not had_candidates:
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    if best is None:
        return None
    if best_sim < TAU_FLOOR:
        return None
    recalled = st["decompressor"].for_emotion_evaluator_matched(
        best, now_vec, now_valence=now_valence)
    # [019] 轮级硬过滤兜底：包内轮全被极性过滤（纯 positive 包进了检索
    #   但暴怒区下无黑历史可捞）→ 无召回（静默好过召回夸包）
    if recalled is not None and recalled.get("recall_source") == "polarity_exhausted":
        return None
    if recalled is not None:
        max_rid = st["decompressor"].package_max_round(best)
        recalled["pkg_age_rounds"] = (float(current_round - max_rid)
                                      if max_rid is not None else None)
        recalled["matched_sim"] = float(best_sim)
        recalled["matched_sim_raw"] = float(best_raw_sim)
        recalled["polarity_penalty"] = float(best_pen)
        recalled["matched_pkg_seq"] = best.seq
    return recalled


def run_scenario(scenario, feature_mode="fixture"):
    fx = get_fixture(scenario)
    st = reset_state(seed=42)

    now = time.time()

    if feature_mode == "fixture":
        x = fx["x"]
        x_fast = fx["x_fast"]
        event = event_parse(
            fx["input"],
            event_severity=fx["event_severity"],
            importance=fx["importance"],
            emotion_weight=fx["emotion_weight"],
            unfinished=fx["unfinished"],
            danger_score=fx["danger_score"],
        )
    elif feature_mode == "encoder":
        event = event_parse(
            fx["input"],
            event_severity=fx["event_severity"],
            importance=fx["importance"],
            emotion_weight=fx["emotion_weight"],
            unfinished=fx["unfinished"],
            danger_score=fx["danger_score"],
        )
        x, x_fast = feature_encode(
            event, st["M_prev"], st["HabituationSignal_prev"], st["B"], st["memory_graph"]
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    # 心光预算（用回合开始的 B）
    B = st["B"]
    B_xg = update_xinguang_budget(B[0], B[1])

    # 海马体评估
    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, now)
    M = hippo["M"]
    ReframedContext = hippo["ReframedContext"]
    HabituationSignal = hippo["HabituationSignal"]
    HistoricalResidual = hippo["HistoricalResidual"]
    M_text = hippo["M_text"]

    # [013 语义信号 S] 真提取：evaluator 从用户消息文本提取 4 维信号
    #   （效价/唤醒/支配/强度；valence ∈ [-1,1]，敌意消息给负价——修复 012
    #   占位 S 把"攻击"当正效价注入导致抑制链路熄火的问题）。
    #   降级链：LLM → 关键词词典 → 全零，source 标记随 log 记录，不静默。
    #   [017.5] 上提到读取侧之前：threat_now 汇流口需要本轮 S_valence，
    #   而召回注入点位于本轮 D_error 产生之前（先有 ReframedContext 才有
    #   慢通路→E_raw→D_error）。上提不改变 S 的取值（同输入同缓存）。
    S, S_source = extract_semantic_signals(fx["input"])

    # 读取侧（T31 重构 · 017.5 落地：包级相关性匹配 + 轮级回捞）
    #   匹配：sim = max(cos_now_mean, cos_now_peak)，全包 argmax，
    #         sim < TAU_FLOOR 不召回；
    #   命中：raw_pointer 回捞 top2 最相关轮，evaluator 现场重提 S
    #         （轮级新鲜语义，告别整包 emotion_value 稀释）。
    #   旧包（无嵌入）/ 嵌入失败 / 开关关闭：退回 packages[-1] 旧路径。
    if MEMORY_READ_ENABLED:
        packages = st["archiver"].get_packages()
        if packages:
            st["recalled_memory"] = _recall_match(st, packages, fx["input"],
                                                  st["round_counter"] + 1,
                                                  now_valence=float(S[0]))
        else:
            st["recalled_memory"] = None
    else:
        st["recalled_memory"] = None

    # T31 方向公式（017.txt 4.3 落地，取代 T24 的 ev×sign 直注）：
    #   threat_now = max(D_error_prev, HR, -S_valence)——海马体与读取侧
    #     在此握手（017 雷点3补丁）；D_error_prev 为本轮注入点之后才产生
    #     的量，取上一轮写回值（一轮延迟 = 威胁感惯性；run_scenario 单轮
    #     模式无跨轮状态，记 0）。
    #   迟滞模式机每轮演进（与是否命中无关）；delta 仅在命中时注入。
    #   delta = clip(ev×sign×(alpha+(1-alpha)×(-H_eff)), -1.0, CAP_POS)。
    # 双记录（009.5 批次丁起沿用）：
    #   recalled_delta_intended = 意图注入量（经护栏与截断后的实际注入值）
    #   recalled_delta_applied  = 逐维有效增量均值（RC 触顶时两者出现差异）
    # recalled_emotion_delta 字段保留（值 = intended），旧探针 probe_t24.py 依赖。
    recalled_emotion_delta = 0.0
    recalled_delta_intended = 0.0
    recalled_delta_applied = 0.0
    rc_pre = ReframedContext
    threat_now = max(float(st.get("D_error_prev", 0.0)),
                     float(HistoricalResidual), -float(S[0]))
    alpha_new, _alpha_raw = update_threat_mode(threat_now, float(st.get("alpha_prev", 0.0)))
    st["alpha_prev"] = alpha_new
    if st["recalled_memory"] is not None:
        rm = st["recalled_memory"]
        _ev = float(rm.get("emotion_value", 0.0))
        _pol = rm.get("emotion_polarity", "neutral")
        _sign = 1.0 if _pol == "positive" else (-1.0 if _pol == "negative" else 0.0)
        _delta, _rd = run_recall_intercept(
            _ev, _sign, HabituationSignal, alpha_new, fx["event_severity"],
            rm.get("pkg_age_rounds"), float(HistoricalResidual))
        rm["recall_detail"] = _rd  # 公式中间量（H_eff/factor/护栏/截断），观测用
        recalled_delta_intended = _delta
    recalled_emotion_delta = recalled_delta_intended
    ReframedContext = np.clip(ReframedContext + recalled_delta_intended, -1.0, 1.0)
    recalled_delta_applied = float(np.mean(ReframedContext - rc_pre))

    # 快通路（可能更新 B[1]）
    D_fast, B = run_fast_path(x_fast, B)

    # 慢通路（用更新后的 B）
    D_slow, V = run_slow_path(x, ReframedContext, M, B, st["w_PFC"], HabituationSignal, S)

    # 情绪释放器（S_valence = S[0]）
    raw_emotion_impact, released_impact, release_rate = run_releaser(V, D_fast, HabituationSignal, S[0])

    # 情绪加点
    E_raw = run_emotion(st["E_prev"], released_impact, B)

    # 拦截层
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(E_raw, st["w_PFC"])

    # 策略
    action = policy_minimal(E_final, I_inhibit, fx["event_severity"])

    # 状态翻译 + prompt + LLM
    # [023] Action 闭环（清单问题1）：action 进 Prompt——state_translator
    #   输出规则第 6 条 + prompt_builder 硬指令位，LLM 不再对 action 无感。
    monologue = state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal, action)
    prompt = build_prompt(monologue, fx["input"], action)
    llm_output = get_llm().generate(prompt)

    # [023] Action 闭环后处理：suppress 暴走护栏——LLM 无视"20 字"指令时，
    #   超过 30 字强行截断并加省略号（宁短勿长：suppress 的语义就是收着）。
    if action == "suppress" and len(llm_output) > 30:
        llm_output = llm_output[:30] + "……"
        print("[pipeline] suppress 护栏触发：输出超过 30 字，已截断加……",
              file=sys.stderr)

    # 回写识海（5 字段，5.5.14）
    event.event_severity = fx["event_severity"]
    # [019] 复合危险度（018 议题一）：语义厌恶锚 + 奖惩对比 + 严重度门，
    #   替代 V[1]*2-0.5（奖励信号驱动，赞美被判最危险，HR 正反馈 runaway）
    event.danger_score = compute_danger_score(S[0], V, fx["event_severity"])
    event.importance = fx["importance"]
    event.emotion_weight = fx["emotion_weight"]
    event.unfinished = fx["unfinished"]
    st["memory_graph"].add(event)

    log = {
        "x": x,
        "x_fast": x_fast,
        "B": B,
        "D_fast": D_fast,
        "B_xg": B_xg,
        "D_slow": D_slow,
        "V": V,
        "S": np.asarray(S, dtype=float),
        "S_source": S_source,
        "raw_emotion_impact": raw_emotion_impact,
        "released_impact": released_impact,
        "E_raw": E_raw,
        "neg_intensity": neg_intensity,
        "S_E": S_E,
        "D_error": D_error,
        "I_inhibit": I_inhibit,
        "E_final": E_final,
        "w_PFC": w_PFC_new,
        "release_rate": release_rate,
        "HabituationSignal": HabituationSignal,
        "HistoricalResidual": HistoricalResidual,
        "M_text": M_text,
        "action": action,
        "feature_mode": feature_mode,
        "llm_output": llm_output,
        "recalled_memory": st["recalled_memory"],
        "ReframedContext": ReframedContext,
        "recalled_emotion_delta": recalled_emotion_delta,
        "recalled_delta_intended": recalled_delta_intended,
        "recalled_delta_applied": recalled_delta_applied,
        # [017.5 T31] 召回拦截层观测字段（threat_now 汇流口 / 迟滞模式状态）
        "threat_now": threat_now,
        "alpha_mode": alpha_new,
        "recall_detail": (st["recalled_memory"] or {}).get("recall_detail")
        if isinstance(st["recalled_memory"], dict) else None,
    }

    log_path = os.path.join(LOGS_DIR, LOG_FILES[scenario])
    write_log(log_path, log)

    # 写入侧（M2-D）：每轮结束写缓冲区，达阈值归档
    #   [017.5] 记录补 intensity/valence（= 本轮 S[3]/S[0]）——归档器据此
    #   选包内"情绪最强轮"作 peak_vec（见 archiver._compute_pkg_vectors）。
    if MEMORY_WRITE_ENABLED:
        st["round_counter"] += 1
        st["memory_buffer"].append({
            "round_id": st["round_counter"],
            "user_input": fx["input"],
            "ai_output": llm_output,
            "timestamp": now,
            "intensity": float(S[3]),
            "valence": float(S[0]),
        })
        if st["archiver"].should_archive():
            st["archiver"].archive()

    return log


def run_turn(state, scenario, feature_mode="fixture"):
    """单轮执行（不 reset），使用传入的 state。

    与 run_scenario 的唯一区别：不调用 reset_state，直接用外部传入的 state，
    支持跨多轮积累 memory_buffer / raw_log / archiver 等。
    """
    fx = get_fixture(scenario)
    st = state

    now = time.time()

    if feature_mode == "fixture":
        x = fx["x"]
        x_fast = fx["x_fast"]
        event = event_parse(
            fx["input"],
            event_severity=fx["event_severity"],
            importance=fx["importance"],
            emotion_weight=fx["emotion_weight"],
            unfinished=fx["unfinished"],
            danger_score=fx["danger_score"],
        )
    elif feature_mode == "encoder":
        event = event_parse(
            fx["input"],
            event_severity=fx["event_severity"],
            importance=fx["importance"],
            emotion_weight=fx["emotion_weight"],
            unfinished=fx["unfinished"],
            danger_score=fx["danger_score"],
        )
        x, x_fast = feature_encode(
            event, st["M_prev"], st["HabituationSignal_prev"], st["B"], st["memory_graph"]
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    # 心光预算（用回合开始的 B）
    B = st["B"]
    B_xg = update_xinguang_budget(B[0], B[1])

    # 海马体评估
    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, now)
    M = hippo["M"]
    ReframedContext = hippo["ReframedContext"]
    HabituationSignal = hippo["HabituationSignal"]
    HistoricalResidual = hippo["HistoricalResidual"]
    M_text = hippo["M_text"]

    # [013 语义信号 S] 真提取：evaluator 从用户消息文本提取 4 维信号
    #   （效价/唤醒/支配/强度；valence ∈ [-1,1]，敌意消息给负价——修复 012
    #   占位 S 把"攻击"当正效价注入导致抑制链路熄火的问题）。
    #   降级链：LLM → 关键词词典 → 全零，source 标记随 log 记录，不静默。
    #   [017.5] 上提到读取侧之前：threat_now 汇流口需要本轮 S_valence，
    #   而召回注入点位于本轮 D_error 产生之前（先有 ReframedContext 才有
    #   慢通路→E_raw→D_error）。上提不改变 S 的取值（同输入同缓存）。
    S, S_source = extract_semantic_signals(fx["input"])

    # 读取侧（T31 重构 · 017.5 落地：包级相关性匹配 + 轮级回捞）
    #   匹配：sim = max(cos_now_mean, cos_now_peak)，全包 argmax，
    #         sim < TAU_FLOOR 不召回；
    #   命中：raw_pointer 回捞 top2 最相关轮，evaluator 现场重提 S
    #         （轮级新鲜语义，告别整包 emotion_value 稀释）。
    #   旧包（无嵌入）/ 嵌入失败 / 开关关闭：退回 packages[-1] 旧路径。
    if MEMORY_READ_ENABLED:
        packages = st["archiver"].get_packages()
        if packages:
            st["recalled_memory"] = _recall_match(st, packages, fx["input"],
                                                  st["round_counter"] + 1,
                                                  now_valence=float(S[0]))
        else:
            st["recalled_memory"] = None
    else:
        st["recalled_memory"] = None

    # T31 方向公式（017.txt 4.3 落地，取代 T24 的 ev×sign 直注）：
    #   threat_now = max(D_error_prev, HR, -S_valence)——海马体与读取侧
    #     在此握手（017 雷点3补丁）；D_error_prev 为本轮注入点之后才产生
    #     的量，取上一轮写回值（一轮延迟 = 威胁感惯性；run_scenario 单轮
    #     模式无跨轮状态，记 0）。
    #   迟滞模式机每轮演进（与是否命中无关）；delta 仅在命中时注入。
    #   delta = clip(ev×sign×(alpha+(1-alpha)×(-H_eff)), -1.0, CAP_POS)。
    # 双记录（009.5 批次丁起沿用）：
    #   recalled_delta_intended = 意图注入量（经护栏与截断后的实际注入值）
    #   recalled_delta_applied  = 逐维有效增量均值（RC 触顶时两者出现差异）
    # recalled_emotion_delta 字段保留（值 = intended），旧探针 probe_t24.py 依赖。
    recalled_emotion_delta = 0.0
    recalled_delta_intended = 0.0
    recalled_delta_applied = 0.0
    rc_pre = ReframedContext
    threat_now = max(float(st.get("D_error_prev", 0.0)),
                     float(HistoricalResidual), -float(S[0]))
    alpha_new, _alpha_raw = update_threat_mode(threat_now, float(st.get("alpha_prev", 0.0)))
    st["alpha_prev"] = alpha_new
    if st["recalled_memory"] is not None:
        rm = st["recalled_memory"]
        _ev = float(rm.get("emotion_value", 0.0))
        _pol = rm.get("emotion_polarity", "neutral")
        _sign = 1.0 if _pol == "positive" else (-1.0 if _pol == "negative" else 0.0)
        _delta, _rd = run_recall_intercept(
            _ev, _sign, HabituationSignal, alpha_new, fx["event_severity"],
            rm.get("pkg_age_rounds"), float(HistoricalResidual))
        rm["recall_detail"] = _rd  # 公式中间量（H_eff/factor/护栏/截断），观测用
        recalled_delta_intended = _delta
    recalled_emotion_delta = recalled_delta_intended
    ReframedContext = np.clip(ReframedContext + recalled_delta_intended, -1.0, 1.0)
    recalled_delta_applied = float(np.mean(ReframedContext - rc_pre))

    # 快通路（可能更新 B[1]）
    D_fast, B = run_fast_path(x_fast, B)

    # 慢通路（用更新后的 B）
    D_slow, V = run_slow_path(x, ReframedContext, M, B, st["w_PFC"], HabituationSignal, S)

    # 情绪释放器（S_valence = S[0]）
    raw_emotion_impact, released_impact, release_rate = run_releaser(V, D_fast, HabituationSignal, S[0])

    # 情绪加点
    E_raw = run_emotion(st["E_prev"], released_impact, B)

    # 拦截层
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(E_raw, st["w_PFC"])

    # 策略
    action = policy_minimal(E_final, I_inhibit, fx["event_severity"])

    # 状态翻译 + prompt + LLM
    # [023] Action 闭环（清单问题1）：action 进 Prompt——state_translator
    #   输出规则第 6 条 + prompt_builder 硬指令位，LLM 不再对 action 无感。
    monologue = state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal, action)
    prompt = build_prompt(monologue, fx["input"], action)
    llm_output = get_llm().generate(prompt)

    # [023] Action 闭环后处理：suppress 暴走护栏——LLM 无视"20 字"指令时，
    #   超过 30 字强行截断并加省略号（宁短勿长：suppress 的语义就是收着）。
    if action == "suppress" and len(llm_output) > 30:
        llm_output = llm_output[:30] + "……"
        print("[pipeline] suppress 护栏触发：输出超过 30 字，已截断加……",
              file=sys.stderr)

    # 回写识海（5 字段，5.5.14）
    event.event_severity = fx["event_severity"]
    # [019] 复合危险度（018 议题一）：语义厌恶锚 + 奖惩对比 + 严重度门，
    #   替代 V[1]*2-0.5（奖励信号驱动，赞美被判最危险，HR 正反馈 runaway）
    event.danger_score = compute_danger_score(S[0], V, fx["event_severity"])
    event.importance = fx["importance"]
    event.emotion_weight = fx["emotion_weight"]
    event.unfinished = fx["unfinished"]
    st["memory_graph"].add(event)

    log = {
        "x": x,
        "x_fast": x_fast,
        "B": B,
        "D_fast": D_fast,
        "B_xg": B_xg,
        "D_slow": D_slow,
        "V": V,
        "S": np.asarray(S, dtype=float),
        "S_source": S_source,
        "raw_emotion_impact": raw_emotion_impact,
        "released_impact": released_impact,
        "E_raw": E_raw,
        "neg_intensity": neg_intensity,
        "S_E": S_E,
        "D_error": D_error,
        "I_inhibit": I_inhibit,
        "E_final": E_final,
        "w_PFC": w_PFC_new,
        "release_rate": release_rate,
        "HabituationSignal": HabituationSignal,
        "HistoricalResidual": HistoricalResidual,
        "M_text": M_text,
        "action": action,
        "feature_mode": feature_mode,
        "llm_output": llm_output,
        "recalled_memory": st["recalled_memory"],
        "ReframedContext": ReframedContext,
        "recalled_emotion_delta": recalled_emotion_delta,
        "recalled_delta_intended": recalled_delta_intended,
        "recalled_delta_applied": recalled_delta_applied,
        # [017.5 T31] 召回拦截层观测字段（threat_now 汇流口 / 迟滞模式状态）
        "threat_now": threat_now,
        "alpha_mode": alpha_new,
        "recall_detail": (st["recalled_memory"] or {}).get("recall_detail")
        if isinstance(st["recalled_memory"], dict) else None,
    }

    log_path = os.path.join(LOGS_DIR, LOG_FILES[scenario])
    write_log(log_path, log)

    # 写入侧（M2-D）：每轮结束写缓冲区，达阈值归档
    #   [017.5] 记录补 intensity/valence（= 本轮 S[3]/S[0]）——归档器据此
    #   选包内"情绪最强轮"作 peak_vec（见 archiver._compute_pkg_vectors）。
    if MEMORY_WRITE_ENABLED:
        st["round_counter"] += 1
        st["memory_buffer"].append({
            "round_id": st["round_counter"],
            "user_input": fx["input"],
            "ai_output": llm_output,
            "timestamp": now,
            "intensity": float(S[3]),
            "valence": float(S[0]),
        })
        if st["archiver"].should_archive():
            st["archiver"].archive()

    # T33：跨轮状态回写（总阀门）。
    # 状态量在本轮结束时写回共享 state，下一轮 run_turn 读取；
    # run_scenario 每轮 reset 是设计行为，不在此列；
    # run_session 内部循环调 run_turn，自动获得跨轮累积。
    # [017.5] alpha_prev 在注入块内已写（迟滞机每轮演进）；此处补
    #   D_error_prev——threat_now 汇流口的上一轮威胁输入。
    st["E_prev"] = E_final
    st["w_PFC"] = w_PFC_new
    st["B"] = B
    st["M_prev"] = M
    st["HabituationSignal_prev"] = HabituationSignal
    st["D_error_prev"] = D_error

    return log


def run_session(scenarios, feature_mode="fixture", seed=42):
    """多轮会话：跨轮共享 state，写入侧可以积累。

    参数:
      scenarios: 场景名列表，每一项是一个场景（如 "被骂"）
      feature_mode: fixture / encoder
      seed: 随机种子

    返回: (logs, st)
      logs: 每轮 log 的列表
      st: 会话结束后的最终 state
    """
    st = reset_state(seed=seed)
    logs = []
    for sc in scenarios:
        log = run_turn(st, sc, feature_mode=feature_mode)
        logs.append(log)
    return logs, st