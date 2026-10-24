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

import config
from state import reset_state
from fast_path import run_fast_path
from slow_path import run_slow_path
from emotion_releaser import run_releaser
from emotion import run_emotion
from intercept import run_intercept, run_recall_intercept, update_threat_mode
from policy import policy_minimal
from xinguang import update_xinguang_budget
from memory_gateway import recall as memory_recall
from event_parser import parse as event_parse
from feature_encoder import encode as feature_encode
from state_translator import state_to_prompt
from prompt_builder import build_prompt, build_messages, collect_history
from llm_client import get_llm
from evaluator import extract_semantic_signals
from embedder import get_embedder
from semantic_projection import project_M
from logger import write_log
from tests.fixtures import get_fixture
import relationship as rel_mod
import session_store

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILES = {
    "被骂": "run_insult.jsonl",
    "被夸": "run_praise.jsonl",
    "被质疑": "run_doubt.jsonl",
    "暴怒": "run_rage.jsonl",
}

# [017.5 T31] 包级匹配嵌入器：与 event_parser/archiver/decompressor 同款
#   （embedder 实例缓存保证全系统共用一个 BGE 模型）
_pkg_embedder = get_embedder(config.PACKAGE_EMBEDDER)


def _cos(a, b) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_danger_score(S_valence, V, event_severity, D_slow=0.0):
    """[019] 复合危险度公式（018 议题一沙盘实证后落地，替代 V[1]*2-0.5）。

    病根：旧公式 scene.danger_score = clip(V[1]*2-0.5) 用奖励信号 V[1]
      定义危险，实测赞美 danger 0.975 > 质疑 0.913 > 骂 0.797 > 暴怒
      0.781——「赞美被判为最危险」，并形成 HR 正反馈 runaway。

    公式（参数见 config [019 施工] 块，全部 [未校准]）：
      danger = clip( W_SEM * (-S_valence)                语义厌恶锚
                   + W_CON * (V[0] - V[1])               奖惩对比
                   + W_SEV * max(0, severity - SEV_NEUTRAL)
                            * (1 - max(0, S_valence)),   严重度门
                   + W_DANGER_SLOW * D_slow,             熟人慢通路介入
                   -1, 1 )
    前三项语义：
      语义厌恶锚——主观「当时多难受」，透明可解释，负价推高危险；
      奖惩对比——慢通路内部惩罚/奖励两行相减（共模抑制），代表身体层
        受伤程度，对「中性语气的重话」敏感；
      严重度门——只罚中高严重度事件，且正效价事件几乎不吃
        （乘 1-max(0,S_valence)），防「笑着说的严重事」被洗白。

    [059 任务1] 第四项 W_DANGER_SLOW * D_slow（B19 修复）：056 起熟人
      效应（_d_slow_gain=1.5）只放大 log 里的 D_slow，下游无人消费。
      本项把放大后的 D_slow 接进危险度：熟人互动时慢通路介入更深 →
      danger 抬升 → 经 033 高门（danger_boost = max(0, danger -
      DANGER_NEUTRAL)）驱动 I_inhibit，进而同时影响 E_final 与 w_PFC
      疲劳积分的恢复/累积分叉。门控在调用方（pipeline._run_chain）：
      仅 _d_slow_gain != 1.0 时传入放大后 D_slow，其余路径传默认
      0.0（本参数默认值），逐位不变。

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
    d_slow = float(D_slow)
    sem = -sv
    con = v0 - v1
    sev_gate = max(0.0, sev - config.SEV_NEUTRAL) * (1.0 - max(0.0, sv))
    # [031] 性格整体刻度：先乘三项和再 clip，保证危险度与阈值的关系可控
    # [059 任务1] 第四项熟人慢通路介入（B19）：门控同调用方，默认 0.0
    return float(np.clip(config.DANGER_GAIN
                         * (config.W_SEM * sem + config.W_CON * con
                            + config.W_SEV * sev_gate
                            + config.W_DANGER_SLOW * d_slow),
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

    [053] 时间衰减（修 B3）：打分从 sim 改为
      sim_eff = sim × exp(-age / PACKAGE_RECENCY_TAU_ROUNDS)
      age = current_round - 包内最大 round_id（单位：轮）。
      三项因子全部落日志，可反推：matched_sim = matched_sim_raw
      × recency_decay × (1 - polarity_penalty)。
      age 未知（raw_pointer 解析失败）→ decay=1.0，不因未知而惩罚。

    [054] 地板与衰减解耦（修 053 的死锁）：
      相关性闸 TAU_FLOOR 判 sim_raw（纯语义），
      argmax 与 matched_sim 用 sim_eff（语义 × 新鲜度）。
      先闸后排序，避免「无关老包赢了 argmax 又被自己的 raw 分挡掉、
      把相关包一起带死」。效果：老记忆不会被判为不相关，只会被排到后面。
    """
    if not config.RECALL_MATCH_ENABLED:
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    try:
        now_vec = _pkg_embedder.encode(input_text)
    except Exception as e:
        print(f"[pipeline] 告警：包级匹配嵌入失败（{type(e).__name__}: {e}），"
              f"退回 latest 旧路径", file=sys.stderr)
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    # [054] 地板与衰减解耦（修 053 遗留的死锁）：
    #   053 把衰减乘进 sim_eff 之后又拿 sim_eff 去比 TAU_FLOOR，两者相乘
    #   导致「老包永久不可达」——age 60 轮时 decay≈0.05，要过 0.45 的地板
    #   需要原始相似度 9.0，数学上不可能。
    #   现在分两段：
    #     相关性闸（TAU_FLOOR）判 sim_raw —— 「像不像同一件事」只由语义决定；
    #     排序（argmax）与 matched_sim 用 sim_eff —— 「先想起哪一件」由语义×新鲜度决定。
    #   顺序上必须先闸后排序：若先 argmax 再对赢家查闸，会出现「一个衰减到
    #   极低的无关包赢了 argmax、又被自己的 raw 分挡掉，把真正相关的包一起
    #   带死」的情况。
    #   结果：老包不会因为时间久被判定为不相关，只会被排到后面。
    cands = []          # (sim_eff, pkg, sim_raw, decay, pen)
    best_sim = -1.0
    best = None
    best_raw_sim, best_pen, best_decay = -1.0, 0.0, 1.0
    had_candidates = False
    gated_out = 0
    # [055 任务3] 跨天折扣：对同一会话内的所有包是同一乘数，
    #   因此不改变 argmax 排序，只压低绝对置信度（matched_sim）。
    #   真正让"隔 3 天"与"隔 30 轮"不同的是 session_store 在加载时对
    #   E_prev / w_PFC / H_prev 做的空档回落，以及 hippocampal 的 H 消退。
    _gap = float(st.get("session_gap_days", 0.0) or 0.0)
    _tau_gap = float(getattr(config, "TAU_GAP_DAYS_RECALL", 0.0) or 0.0)
    gap_factor = 1.0
    if _gap > 0.0 and _tau_gap > 0.0:
        gap_factor = float(np.exp(-_gap / _tau_gap))
    for pkg in packages:
        mv = getattr(pkg, "mean_vec", None)
        if mv is None or len(mv) == 0 or len(mv) != len(now_vec):
            continue  # 旧包（无嵌入）或维度不符：不参与 argmax
        had_candidates = True
        s_m = _cos(now_vec, mv)
        pv = getattr(pkg, "peak_vec", None)
        s_p = _cos(now_vec, pv) if (pv is not None and len(pv) == len(now_vec)) else -1.0
        sim = max(s_m, s_p)
        # [054] 相关性闸：只看原始语义相似度，不受时间影响
        if sim < config.TAU_FLOOR:
            gated_out += 1
            continue
        # [053] 包级时间衰减：sim_eff = sim * exp(-age/tau)，age 用轮龄。
        #   age 未知（raw_pointer 解析失败）时 decay=1.0，不因未知而惩罚。
        _mx = st["decompressor"].package_max_round(pkg)
        age = (float(current_round - _mx) if _mx is not None else None)
        decay = 1.0
        if age is not None:
            _tau = float(getattr(config, "PACKAGE_RECENCY_TAU_ROUNDS", 0.0) or 0.0)
            if _tau > 0.0:
                decay = float(np.exp(-max(0.0, age) / _tau))
        # [055] 再乘跨天空档折扣（全局同乘数，不改排序）
        decay *= gap_factor
        sim_eff = sim * decay
        # [019] 极性一致性过滤（不对称）——包级只软惩罚，硬过滤下沉轮级
        #   （018 实证修正：包级硬过滤会把 3 夸 1 骂的混合包也滤掉，可它
        #   明明含骂轮——被骂早期语料里唯一的包正是这种，硬滤导致无召回、
        #   B2「暴怒时被放大」失败。混合包的存活可能必须保留到轮级。）
        pen = 0.0
        if config.POLARITY_FILTER_ENABLED and now_valence is not None:
            pkg_pv = getattr(pkg, "mean_valence", None)
            if pkg_pv is not None:
                nv = float(now_valence)
                pvv = float(pkg_pv)
                if nv < -config.THETA_MILD and pvv > 0.0:
                    pen = config.LAMBDA_POL * pvv * (-nv)   # 威胁区（含暴怒区）软惩罚
                    sim_eff = sim_eff * (1.0 - min(1.0, pen))
        cands.append((sim_eff, pkg, sim, decay, pen))
        if sim_eff > best_sim:
            best, best_sim = pkg, sim_eff
            best_raw_sim, best_pen, best_decay = sim, pen, decay
    if not had_candidates:
        return st["decompressor"].for_emotion_evaluator(packages[-1])
    if best is None:
        # [054] 有候选包但全部被相关性闸挡掉：静默无召回（原语义不变）
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
        # [053] 时间衰减因子：matched_sim = raw * recency_decay * (1 - penalty)
        recalled["recency_decay"] = float(best_decay)
        # [054] 闸门观测：通过相关性闸的包数 / 被挡掉的包数
        recalled["gate_candidates"] = int(len(cands))
        recalled["gate_blocked"] = int(gated_out)
        recalled["matched_pkg_seq"] = best.seq
    return recalled


def _apply_resident_gap(st, now):
    """[058 任务1] 常驻进程跨天衰减（修 B16 假死）。

    病灶：055 的跨天衰减只在 session_store.load_state 里按 saved_at
    结算。server_api 的会话表是 LRU 缓存——进程不重启、会话不淘汰，
    就永远不走 load：用户隔三天回来，"气三天"与"没气过"同值。

    本函数在轮初被调：now - st["last_active"] 超过
    config.RESIDENT_GAP_THRESHOLD_SEC 时，按 load 的同四条通道
    （session_store.apply_gap_decay）结算一次，返回明细 dict；
    未超阈值返回 None 且绝不触碰任何数值（阈值内机制透明——
    黄金轨迹与既有测试逐位不受影响）。

    时序边界：last_active 由每轮结束写回（见 _run_chain 尾部），
    load 时对齐到加载时刻（session_store.load_state）——
    两处共同保证同一段空档不会被结算两次。
    """
    try:
        last = float(st.get("last_active", now) or now)
    except (TypeError, ValueError):
        last = float(now)
    gap_sec = float(now) - last
    if gap_sec <= float(config.RESIDENT_GAP_THRESHOLD_SEC):
        return None
    detail = session_store.apply_gap_decay(st, gap_sec / 86400.0)
    detail["gap_sec"] = float(gap_sec)
    return detail


def _run_chain(st, scenario, feature_mode="fixture"):
    """053：18 步主链路唯一实现（合并原 run_scenario / run_turn 两份拷贝）。

    合并前两个入口的函数体逐字相同，仅差三处：
      1) 状态来源：run_scenario 内部 reset_state，run_turn 用传入的 state；
      2) 性格加载：只有 run_turn 每轮按会话 load_personality；
      3) 跨轮回写：只有 run_turn 写回 E_prev / w_PFC / B / M_prev /
         HabituationSignal_prev / D_error_prev。
    合并策略：第 3 项移入本函数（单轮入口的 st 用完即弃，写回不可观测），
    第 1/2 项留在各自的薄包装里：
      run_scenario -> reset_state + 本函数（单轮验收 / 探针）
      run_turn     -> load_personality + 本函数（CLI / 网页 / 多轮）
    行为等价性由 tests/test_golden_trajectory.py 逐位验证（容差 1e-6）：
    合并前后 20 轮轨迹完全一致，见 053.txt 第一节。
    """
    fx = get_fixture(scenario)

    # [039] 破防冷却：轮初先读旧值、再递减；旧值用于本轮判定，
    #   使 express 触发后置 10 恰好封住接下来 10 轮（第 11 轮可再次破防）。
    express_cooldown_before = int(st.get("express_cooldown", 0))
    st["express_cooldown"] = max(0, express_cooldown_before - 1)

    now = time.time()

    # [058 任务1] 常驻进程跨天衰减（修 B16）：LRU 命中不走 load 的会话，
    #   轮初按 last_active 结算空档。阈值内返回 None，数值链逐位不动。
    #   必须在任何跨轮状态被读取之前（feature_encode / 召回 / 慢通路
    #   都吃 E_prev / w_PFC / HabituationSignal_prev）。
    _resident_gap = _apply_resident_gap(st, now)

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

    # [013 语义信号 S] 真提取：evaluator 从用户消息文本提取 4 维信号
    #   （效价/唤醒/支配/强度；valence ∈ [-1,1]，敌意消息给负价——修复 012
    #   占位 S 把"攻击"当正效价注入导致抑制链路熄火的问题）。
    #   降级链：LLM → 关键词词典 → 全零，source 标记随 log 记录，不静默。
    #   [017.5] 上提到读取侧之前：threat_now 汇流口需要本轮 S_valence，
    #   而召回注入点位于本轮 D_error 产生之前（先有 ReframedContext 才有
    #   慢通路→E_raw→D_error）。上提不改变 S 的取值（同输入同缓存）。
    S, S_source = extract_semantic_signals(fx["input"])

    # 检索网关（029 第一阶段）：事件级（海马体）与包级（读原文）一次调用。
    #   网关内部仍分别调用 hippocampal_evaluator.evaluate 与 _recall_match，
    #   不合并打分、不改阈值；本处只负责取回结果（行为与改造前逐位等价）。
    _recall = memory_recall(fx["input"], event, st["memory_graph"], x, B_xg,
                            now, st, st["round_counter"] + 1,
                            now_valence=float(S[0]),
                            gap_days=float(st.get("session_gap_days", 0.0) or 0.0))
    M = _recall.m_vec
    ReframedContext = _recall.reframed_context
    HabituationSignal = _recall.habituation_signal
    HistoricalResidual = _recall.historical_residual
    M_text = _recall.m_text
    st["recalled_memory"] = _recall.recalled

    # [056 任务2] 熟人效应（读上一轮结转 trust；本轮更新在 danger_now 之后）：
    #   trust>0.7 → D_slow ×1.5（熟人面前想得多）；trust<0.3 → H ×1.5（麻了）。
    #   H 放大只作用于当轮三个使用点（召回拦截/慢通路/释放器），写回
    #   HabituationSignal_prev 用未放大值——H 本身是饱和函数，逐轮自叠加
    #   会指数爆炸（055 实测 3 轮即触顶，不能再喂增益）。
    #   深拷贝快照进 log：账本是跨轮共享 dict，log 要的是当轮截面。
    _rel_trust = rel_mod.user_trust(st)
    _d_slow_gain, _h_gain = rel_mod.acquaintance_effects(_rel_trust)
    _habituation_raw = HabituationSignal
    if _h_gain != 1.0:
        HabituationSignal = float(np.clip(_habituation_raw * _h_gain, 0.0, 1.0))

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
    # [054] M 先减去会话均值基再进慢通路：BGE 各向异性使原始 M 的
    #   全 1 方向分量对任何话题都相同，u_s@M 因此退化成常数偏置。
    #   居中只影响慢通路的语义读出；M 原样写回 M_prev（x0 的余弦口径不变）。
    M_in = project_M(M, st["memory_graph"])
    D_slow, V = run_slow_path(x, ReframedContext, M_in, B, st["w_PFC"], HabituationSignal, S)
    # [056 任务2] 熟人效应第二条通道：熟人（trust>0.7）慢通路介入放大。
    #   D_slow 每轮重算、不写回，直接乘无累积风险。
    if _d_slow_gain != 1.0:
        D_slow = float(D_slow) * _d_slow_gain

    # 情绪释放器（S_valence = S[0]）
    raw_emotion_impact, released_impact, release_rate = run_releaser(
        V, D_fast, HabituationSignal, S[0], S=S)

    # 情绪加点
    E_raw = run_emotion(st["E_prev"], released_impact, B)

    # [032] 危险度接入自控：先按本轮 V/S 算 danger_score。
    # [033] 高门：danger_boost = max(0, danger - DANGER_NEUTRAL)，
    #   中性线以下的轻度事件（如被质疑）不放大自控误差。
    #   danger_now 同时用于回写 event.danger_score，避免重复计算与口径漂移。
    # [059 任务1] B19 修复：熟人（_d_slow_gain != 1.0）把放大后的 D_slow
    #   接进第四项（W_DANGER_SLOW），经高门驱动 E_final / w_PFC；
    #   非熟人恒传 0.0，黄金轨迹逐位不变。
    danger_now = compute_danger_score(
        S[0], V, fx["event_severity"],
        D_slow=(D_slow if _d_slow_gain != 1.0 else 0.0))
    danger_boost = max(0.0, danger_now - config.DANGER_NEUTRAL)

    # [056 任务2] 关系账本轮末更新：按本轮实担危险度结算信任
    #   （攻击 -=0.02×danger / 安抚 +0.01 / intimacy +0.01 / 破裂判定）。
    #   时序说明：熟人效应本轮生效用的是更新前的 trust（_rel_trust，
    #   上一轮结转），攻击对"麻木/深思"的影响从下一轮开始——人对关系
    #   的感知滞后于事件，这是设计不是缺陷。
    #   随后做当轮截面快照进 log（账本是跨轮共享 dict，不能直接引用）。
    rel_mod.update_relationship(st, event.entity_ids, danger_now, float(S[0]))
    _relationships_snapshot = {k: dict(v) for k, v
                               in (st.get("relationships") or {}).items()}

    # 拦截层
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(
        E_raw, st["w_PFC"], danger_boost=danger_boost)

    # 策略（[037] 直通闸改判 danger_now 本身，不再传 danger_boost）
    action = policy_minimal(E_final, I_inhibit, fx["event_severity"],
                            danger_now=danger_now,
                            express_cooldown=express_cooldown_before)
    # [039] 破防后进入冷却：接下来 10 轮不再触发 express
    if action == "express":
        st["express_cooldown"] = 10

    # 状态翻译 + prompt + LLM
    # [023] Action 闭环（清单问题1）：action 进 Prompt——state_translator
    #   输出规则第 6 条 + prompt_builder 硬指令位，LLM 不再对 action 无感。
    monologue = state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal, action,
                                relationship=(st.get("relationships") or {}).get("user_001"))
    prompt = build_prompt(monologue, fx["input"], action)
    # [055 任务1] 对话历史：模型现在能看到前几轮聊了什么。
    #   独白与行动指令仍放在最后一条 user 消息里（状态注入方式不变），
    #   历史只作为前序 user/assistant 对。
    history = collect_history(st)
    messages = build_messages(monologue, fx["input"], action, history)
    llm_output = get_llm().generate(prompt, messages=messages)

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
    event.danger_score = danger_now
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
        # [041] 观测字段：当轮复合危险度，供前端展示（不参与任何计算）
        "danger_now": float(danger_now),
        "feature_mode": feature_mode,
        # [056 任务2] 关系状态观测：当轮结算后的账本截面、熟人效应读入的
        #   trust（更新前）、两条通道的增益倍数、H 放大前的原值。
        #   全部只观测不参与后续计算——除增益本身已在上面应用。
        "relationships": _relationships_snapshot,
        "rel_trust_used": float(_rel_trust),
        "rel_d_slow_gain": float(_d_slow_gain),
        "rel_h_gain": float(_h_gain),
        "HabituationSignal_raw": float(_habituation_raw),
        # [055] 观测：本轮实际带给模型的历史消息条数（0 = 首轮/无历史）
        "history_msgs": len(history),
        # [058 任务1] 观测：常驻空档结算明细（None = 未超阈值未结算）。
        #   四条通道的 before/after 全量落盘，探针据此验证衰减真的发生。
        "resident_gap": _resident_gap,
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
    if config.MEMORY_WRITE_ENABLED:
        st["round_counter"] += 1
        st["memory_buffer"].append({
            "round_id": st["round_counter"],
            "user_input": fx["input"],
            "ai_output": llm_output,
            "timestamp": now,
            "intensity": float(S[3]),
            "valence": float(S[0]),
            # [029 任务3] 写入侧补危险度原料：供包级聚合与后续 HR 反推
            #   （round 级 danger，避免整包均值稀释威胁；旧记录无此字段）
            "danger_score": float(event.danger_score),
        })
        if st["archiver"].should_archive():
            st["archiver"].archive()

    # T33：跨轮状态回写（总阀门）。
    # 状态量在本轮结束时写回共享 state，下一轮读取；
    # run_scenario 每轮 reset 是设计行为，不在此列；
    # run_session 内部循环调 run_turn，自动获得跨轮累积。
    # [017.5] alpha_prev 在注入块内已写（迟滞机每轮演进）；此处补
    #   D_error_prev——threat_now 汇流口的上一轮威胁输入。
    st["E_prev"] = E_final
    st["w_PFC"] = w_PFC_new
    st["B"] = B
    st["M_prev"] = M
    # [056] 写回未放大的 H：熟人效应的 H 增益只作用于当轮三个使用点，
    #   逐轮自叠加会让这个饱和函数指数爆炸（055 实测 3 轮触顶）。
    st["HabituationSignal_prev"] = _habituation_raw
    st["D_error_prev"] = D_error
    # [058 任务1] 常驻空档基准推进到本轮开始时刻（轮初为界：
    #   一轮之内墙钟漂移不计入空档，长轮次不会自我触发衰减）。
    st["last_active"] = now

    return log

def run_scenario(scenario, feature_mode="fixture"):
    """[兼容入口] 单轮场景验收：每次新建状态，返回该轮 log。

    053 起不再自带一份链路实现，内部转调 _run_chain；
    返回值结构与合并前完全一致（旧探针与 main.py 不受影响）。
    """
    return _run_chain(reset_state(seed=42), scenario, feature_mode=feature_mode)


def run_turn(state, scenario, feature_mode="fixture"):
    """单轮执行（不 reset），使用传入的 state。

    与 run_scenario 的唯一区别：不调用 reset_state，直接用外部传入的 state，
    支持跨多轮积累 memory_buffer / raw_log / archiver 等。
    """
    # [031] 性格冷切换：会话创建时固定 profile_id，本入口每轮按会话加载一次。
    #   作用：即使同一进程里别的会话改过全局 config，本会话也不会漂移。
    config.load_personality(state.get("profile_id", config.PERSONALITY_DEFAULT))
    return _run_chain(state, scenario, feature_mode=feature_mode)



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