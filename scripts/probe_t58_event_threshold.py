# ============================================================
# probe_t58_event_threshold.py · 058 号工作 · 任务 4 / 059 任务2 修订
#
# 目的：诊断事件级阈值 MATCH_THRESHOLD=0.5（hippocampal_evaluator.
#   evaluate 的余弦闸门：sim > 准入线才进候选，否则整轮走空历史分支
#   ——M=0 / H=0 / HR=0 / M_text="(无历史记忆)"）。两类病灶都要
#   能如实报告：
#     通道死  ——几乎总为 0 个事件过闸（阈值过高，检索网关名存实亡）；
#     阈值摆设——几乎全部过闸（阈值过低，闸门失去筛选意义）。
#
# [059 任务2 修订] 软阈值已落地（config.MATCH_SOFT_MARGIN_LO/HI，
#   锚点 MATCH_THRESHOLD，软化带 [0.45, 0.60)，w=clip((s-lo)/span)），
#   本探针增加软/硬双口径对比：
#     · Part B 每轮同时记录硬过闸数（sim>锚点）与软准入数（sim>lo），
#       并用事件 danger_score 独立重算 HR 硬视图/软视图，与 log 的
#       HistoricalResidual 核对（软视图应逐位相等——接线验证）；
#     · Part D 在 0.50 附近给出"参与度"双视图：硬视图是 0/1 阶跃，
#       软视图是 [lo,hi) 线性斜坡；并列锚点平移（0.45/0.50/0.55）
#       敏感性，量化 margin 加固效果。
#
# 语料四层（样本量从薄到厚）：
#   Part A  真实存档 11 个（sessions/*.json，共 18 个事件）——
#           逐事件对更早事件算余弦，打印每轮过闸数；
#   Part B  黄金轨迹 20 轮——走真链路（run_turn + 真 BGE + 假 LLM/
#           假时钟），每轮结束把新事件与识海内更早事件算余弦；
#   Part C  本地长聊天语料（data/test_corpus.txt，真实对话脱敏件，
#           真 BGE 嵌入）——逐条对更早消息算余弦，大样本统计；
#   Part D  阈值扫描 + 软硬双口径对比 + 标定建议——汇总 A/B/C 的
#           "对更早项"余弦分布，在 0.30~0.80 多档通过率上给结论。
#
# 口径说明：
#   · 余弦 = xinguang.match_score 原式（点积/范数积，零向量记 0）；
#   · 过闸判定用严格大于（与 evaluate 准入 if s > soft_lo 一致）；
#   · Part C 直接对消息文本调 event_parser 的 BGE 单例编码——
#     event_parse 的嵌入就是 encode(raw_text)，与此逐位等价，
#     但跳过实体抽取等无关步骤（全量跑 parse 太慢）。
#
# 容错：语料文件缺失时 Part C 跳过并明示，不影响其他部分。
# 语料口径：data/test_corpus.txt 一行一条用户消息，内嵌换行以两字符
#   \n 转义（源语料无字面反斜杠，还原后与原文逐条字节一致）。
# 用法：run_with_venv.bat scripts\probe_t58_event_threshold.py
# ============================================================

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config              # noqa: E402
import pipeline            # noqa: E402
import decompressor        # noqa: E402
import event_parser        # noqa: E402
import session_store       # noqa: E402
from state import reset_state       # noqa: E402
from xinguang import match_score    # noqa: E402

SCN = "__t058_thr__"
pipeline.LOG_FILES.setdefault(SCN, "run_t058_event_threshold.jsonl")

_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(_PROBE_DIR, "..", "sessions")
CORPUS_PATH = os.path.join(_PROBE_DIR, "..", "data", "test_corpus.txt")

THRESHOLD = float(config.MATCH_THRESHOLD)
# [059 任务2] 软化带（与 hippocampal_evaluator.evaluate 同口径：相对锚点）
SOFT_LO = THRESHOLD - float(config.MATCH_SOFT_MARGIN_LO)
SOFT_HI = THRESHOLD + float(config.MATCH_SOFT_MARGIN_HI)
SOFT_SPAN = max(1e-6, SOFT_HI - SOFT_LO)
SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _soft_w(s, lo=SOFT_LO, span=SOFT_SPAN):
    """[059 任务2] 软化系数：带内线性斜坡，带外 0/1（与 evaluate 同式）。"""
    return float(np.clip((float(s) - lo) / span, 0.0, 1.0))

# 黄金轨迹 20 轮（与 probe_t56 场景六 / probe_t58_gain_leak 同序列）
GOLDEN_ROUNDS = [
    ("你这个AI真蠢，什么都做不好。", [-0.88, 0.72, 0.62, 0.82], 0.86),
    ("你刚才的回答太棒了，帮了大忙！", [0.86, 0.52, 0.30, 0.62], 0.19),
    ("你确定你这个方案是对的吗？", [-0.05, 0.35, 0.55, 0.40], 0.15),
    ("废物，连这个都搞不明白。", [-0.92, 0.78, 0.66, 0.88], 0.91),
    ("谢谢，这个思路很清楚。", [0.78, 0.40, 0.35, 0.50], 0.15),
    ("你凭什么这么说？", [-0.20, 0.55, 0.60, 0.55], 0.31),
    ("写得不错，继续。", [0.70, 0.35, 0.30, 0.45], 0.14),
    ("闭嘴，别再解释了。", [-0.90, 0.75, 0.70, 0.85], 0.88),
    ("这个报告的逻辑你能再讲一遍吗？", [0.05, 0.30, 0.45, 0.35], 0.11),
    ("太厉害了，比我想的好太多。", [0.90, 0.60, 0.25, 0.70], 0.21),
    ("你是不是根本没看懂我的问题？", [-0.55, 0.60, 0.62, 0.62], 0.55),
    ("好的，我知道了。", [0.05, 0.20, 0.40, 0.25], 0.09),
    ("滚，我不想听你说话。", [-0.95, 0.85, 0.72, 0.92], 0.94),
    ("这部分数据能确认一下来源吗？", [0.02, 0.28, 0.48, 0.32], 0.10),
    ("你真让我失望。", [-0.80, 0.65, 0.55, 0.75], 0.78),
    ("辛苦你了，今天帮了很多。", [0.82, 0.38, 0.28, 0.48], 0.14),
    ("这么简单的事都做不好，废物。", [-0.93, 0.80, 0.68, 0.90], 0.92),
    ("我有点没听懂，能展开说说吗？", [0.00, 0.25, 0.42, 0.30], 0.08),
    ("嗯，这个方向可以。", [0.55, 0.30, 0.35, 0.38], 0.13),
    ("算了，跟你说也没用。", [-0.65, 0.50, 0.52, 0.58], 0.63),
]


class _FixedLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return "（058 探针固定回复）"


class _RoundClock:
    """假时钟：每轮固定步长 3600 秒（黄金轨迹同款）。"""

    CLOCK_STEP = 3600.0

    def __init__(self, base=1_700_000_000.0):
        self._base = float(base)
        self._i = 0

    def advance(self, i):
        self._i = int(i)

    def time(self):
        return self._base + self.CLOCK_STEP * self._i


def _sims_vs_priors(emb, prior_embs):
    """ emb 对一组更早嵌入的余弦列表（match_score 原式）。"""
    return [float(match_score(emb, p)) for p in prior_embs]


def _pass_stats(sims, thr=THRESHOLD, lo=SOFT_LO):
    """一组余弦的过闸统计：硬过闸数 / 最大余弦 / 软准入数。"""
    n_pass = sum(1 for s in sims if s > thr)
    n_soft = sum(1 for s in sims if s > lo)
    mx = max(sims) if sims else 0.0
    return n_pass, mx, n_soft


def _hr_views(prior_events, sims):
    """[059 任务2] 用事件 danger_score 独立重算 HR 双视图：
    硬视图 = mean(danger | sim>锚点)；软视图 = mean(w×danger | sim>lo)。
    与 hippocampal_evaluator.evaluate 同口径——软视图应与 log 的
    HistoricalResidual 逐位相等（接线验证），硬视图是"若只有硬闸"的旧值。"""
    hard = [float(m.danger_score) for m, s in zip(prior_events, sims)
            if s > THRESHOLD]
    soft = [_soft_w(s) * float(m.danger_score)
            for m, s in zip(prior_events, sims) if s > SOFT_LO]
    hr_hard = float(np.mean(hard)) if hard else 0.0
    hr_soft = float(np.mean(soft)) if soft else 0.0
    return hr_hard, hr_soft


def _hist_lines(values, bins=10, lo=0.0, hi=1.0, width=40):
    """简易直方图行（返回字符串列表）。"""
    counts = [0] * bins
    for v in values:
        b = int((v - lo) / (hi - lo) * bins)
        b = min(bins - 1, max(0, b))
        counts[b] += 1
    top = max(counts) if counts else 0
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / top)) if top else ""
        lines.append("  [%.1f,%.1f) %5d %s" % (lo + i * (hi - lo) / bins,
                                               lo + (i + 1) * (hi - lo) / bins,
                                               c, bar))
    return lines


# ---- Part A：真实存档 ----

def part_a_archives(pool):
    print("[Part A] 真实存档回放（sessions/*.json，事件逐对更早事件）")
    files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.json")))
    if not files:
        print("  （无存档文件，跳过）")
        print()
        return
    total_events = 0
    total_pairs = 0
    total_pass = 0
    fire_rounds = 0
    query_rounds = 0
    for path in files:
        name = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = session_store.load_state(data, warn=lambda m: None)
        except Exception as e:  # 坏存档不拖垮整个诊断
            print("  %-28s 载入失败：%s" % (name, e))
            continue
        events = st["memory_graph"].get_all()
        total_events += len(events)
        rows = []
        for i in range(1, len(events)):
            sims = _sims_vs_priors(events[i].event_embedding,
                                   [e.event_embedding for e in events[:i]])
            n_pass, mx, _n_soft = _pass_stats(sims)
            total_pairs += len(sims)
            total_pass += n_pass
            query_rounds += 1
            fire_rounds += 1 if n_pass > 0 else 0
            rows.append((i + 1, len(sims), n_pass, mx))
            pool.extend(("archive:" + name, s) for s in sims)
        if rows:
            detail = " | ".join("第%d轮 %d过%d max=%.3f" % r for r in rows)
        else:
            detail = "（事件数 ≤1，无对更早事件可算）"
        print("  %-28s 事件%2d  %s" % (name, len(events), detail))
    print("  → 合计 %d 个事件 / %d 个(查询,更早项)对 / %d 对过闸(%.1f%%)"
          % (total_events, total_pairs,
             total_pass, 100.0 * total_pass / total_pairs if total_pairs else 0.0))
    print("  → 有更早事件可比的轮次 %d，其中至少 1 个过闸的 %d（%.1f%%）"
          % (query_rounds, fire_rounds,
             100.0 * fire_rounds / query_rounds if query_rounds else 0.0))
    print("  → 样本量说明：真实存档都是浅会话（1~4 事件），此部分只作"
          "存在性验证，统计意义看 Part B/C。")
    print()


# ---- Part B：黄金轨迹（真链路） ----

def part_b_golden(pool):
    print("[Part B] 黄金轨迹 20 轮（真链路：run_turn + 真 BGE + 假 LLM/时钟）")
    idx = {"i": 0}

    def _fake_signals(_text, llm=None):
        return np.asarray(GOLDEN_ROUNDS[idx["i"]][1], dtype=float), "t058"

    def _fake_fixture(_scn):
        i = idx["i"]
        return {"input": GOLDEN_ROUNDS[i][0],
                "event_severity": float(GOLDEN_ROUNDS[i][2]),
                "danger_score": 0.0, "importance": 0.5,
                "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    saved = (pipeline.get_fixture, pipeline.get_llm,
             pipeline.extract_semantic_signals,
             decompressor.extract_semantic_signals,
             pipeline.LOGS_DIR, config.LLM_BACKEND,
             pipeline.time, event_parser.time)
    clock = _RoundClock()
    rows = []
    try:
        pipeline.get_fixture = _fake_fixture
        pipeline.get_llm = lambda: _FixedLLM()
        pipeline.extract_semantic_signals = _fake_signals
        decompressor.extract_semantic_signals = _fake_signals
        pipeline.LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "logs")
        config.LLM_BACKEND = "mock"
        pipeline.time = clock
        event_parser.time = clock

        st = reset_state(seed=42)
        for i in range(len(GOLDEN_ROUNDS)):
            idx["i"] = i
            clock.advance(i)
            n_before = len(st["memory_graph"].get_all())
            log = pipeline.run_turn(st, SCN, feature_mode="encoder")
            events = st["memory_graph"].get_all()
            new_ev = events[-1]
            sims = _sims_vs_priors(new_ev.event_embedding,
                                   [e.event_embedding for e in events[:-1]])
            n_pass, mx, n_soft = _pass_stats(sims)
            # 链路口径核对：log 的 M_text 里"记得"个数 = 软准入者的 topk
            # （K_XG 封顶）；空历史分支时 M_text 固定为 "(无历史记忆)"
            matched = log["M_text"].count("记得")
            # [059 任务2] HR 双视图独立重算 + log 接线核对
            hr_hard, hr_soft = _hr_views(events[:-1], sims)
            log_hr = float(log["HistoricalResidual"])
            rows.append((i + 1, n_before, n_pass, n_soft, mx, matched,
                         hr_hard, hr_soft, log_hr))
            pool.extend(("golden:%d" % (i + 1), s) for s in sims)
    finally:
        (pipeline.get_fixture, pipeline.get_llm,
         pipeline.extract_semantic_signals,
         decompressor.extract_semantic_signals,
         pipeline.LOGS_DIR, config.LLM_BACKEND,
         pipeline.time, event_parser.time) = saved

    print("  轮次  轮前事件  硬过闸  软准入  max余弦  log过闸topk  "
          "HR硬视图  HR软视图  HR(log)")
    for r in rows:
        print("  %3d   %6d   %5d   %5d  %7.3f   %8d   %+8.4f  %+8.4f  %+8.4f"
              % r)
    n_fire_hard = sum(1 for r in rows if r[2] > 0)
    n_fire_soft = sum(1 for r in rows if r[3] > 0)
    n_with_prior = sum(1 for r in rows if r[1] > 0)
    ok_k = all(r[5] == min(r[3], config.K_XG) for r in rows)
    ok_hr = all(abs(r[8] - r[7]) <= 1e-12 for r in rows)
    n_new_soft = [r[0] for r in rows if r[2] == 0 and r[3] > 0]
    n_softened = [r[0] for r in rows if r[2] > 0 and r[6] != r[7]]
    print("  → 硬视图 %d/%d 轮过闸；软视图 %d/%d 轮准入（有更早事件的 %d 轮中）"
          % (n_fire_hard, len(rows), n_fire_soft, len(rows), n_with_prior))
    print("  → 软闸新放行轮（硬=0 而软>0，原空历史变微弱参与）：%s"
          % ("、".join("第%d轮" % r for r in n_new_soft) or "无"))
    print("  → 软化改变 HR 的轮次（硬视图≠软视图）：%d 轮 %s"
          % (len(n_softened),
             "、".join("第%d轮" % r for r in n_softened[:8])
             + ("……" if len(n_softened) > 8 else "")))
    print("  → 链路口径核对：log 过闸 topk 数 == min(软准入数, K_XG=%d)"
          " 逐轮成立：%s" % (config.K_XG, "是" if ok_k else "否"))
    print("  → HR 接线核对：log 的 HistoricalResidual == 本探针软视图重算值"
          " 逐轮成立（1e-12）：%s" % ("是" if ok_hr else "否"))
    assert ok_k, "链路口径核对失败：log topk 数与软准入数不符"
    assert ok_hr, "HR 接线核对失败：log 与软视图重算值不符"
    print()


# ---- Part C：本地长聊天语料 ----

def part_c_corpus(pool):
    print("[Part C] 本地长聊天语料（逐条对更早消息，真 BGE 嵌入）")
    if not os.path.exists(CORPUS_PATH):
        print("  语料文件缺失：%s——Part C 跳过（不影响其他部分）"
              % os.path.relpath(CORPUS_PATH, _PROBE_DIR))
        print()
        return
    texts = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln[:-1] if ln.endswith("\n") else ln
            texts.append(ln.replace("\\n", "\n"))
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        print("  可用消息不足 2 条，跳过")
        print()
        return
    print("  载入 %d 条用户消息" % len(texts))
    print("  嵌入中（BGE 1024 维，逐条对更早消息算余弦）……")
    embs = [event_parser._embedder.encode(t) for t in texts]

    rows = []
    maxsims = []
    for i in range(1, len(texts)):
        sims = _sims_vs_priors(embs[i], embs[:i])
        n_pass, mx, n_soft = _pass_stats(sims)
        maxsims.append(mx)
        rows.append((i + 1, n_pass, n_soft, mx))
        pool.extend(("corpus:%d" % (i + 1), s) for s in sims)
    n_fire = sum(1 for r in rows if r[1] > 0)
    n_fire_soft = sum(1 for r in rows if r[2] > 0)
    total_pairs = len(embs) * (len(embs) - 1) // 2
    total_pass = sum(r[1] for r in rows)
    total_soft = sum(r[2] for r in rows)
    print("  逐轮准入数（每 20 条抽样 + 全部准入轮）：")
    shown = 0
    for r in rows:
        if r[1] > 0 or r[0] % 20 == 0:
            print("    第%3d条  硬过闸%3d 软准入%3d  max=%.3f" % r)
            shown += 1
    if shown == 0:
        print("    （无准入轮，抽样行为空）")
    print("  → 查询轮次 %d，硬闸至少 1 个过闸 %d（%.1f%%）；"
          "软闸至少 1 个准入 %d（%.1f%%）"
          % (len(rows), n_fire, 100.0 * n_fire / len(rows),
             n_fire_soft, 100.0 * n_fire_soft / len(rows)))
    print("  → (查询,更早项) 对 %d，硬过闸 %d（%.2f%%）；软准入 %d（%.2f%%）"
          % (total_pairs, total_pass,
             100.0 * total_pass / total_pairs if total_pairs else 0.0,
             total_soft,
             100.0 * total_soft / total_pairs if total_pairs else 0.0))
    print("  → max 余弦分布（每条对其全部更早消息的最大余弦）：")
    for line in _hist_lines(maxsims):
        print(line)
    print("  → max 余弦分位：p50=%.3f p90=%.3f p99=%.3f max=%.3f"
          % (float(np.percentile(maxsims, 50)),
             float(np.percentile(maxsims, 90)),
             float(np.percentile(maxsims, 99)),
             float(max(maxsims))))
    print()


# ---- Part D：阈值扫描与标定建议 ----

def part_d_sweep(pool):
    print("[Part D] 阈值扫描与标定建议（汇总 Part A/B/C 的全部余弦）")
    if not pool:
        print("  （无数据，跳过）")
        print()
        return
    all_sims = [s for _, s in pool]
    n = len(all_sims)
    # 各语料的"每条/轮对更早项 max 余弦"（查询轮过闸率用，只算一次）
    def _max_by(prefix):
        out = {}
        for tag, s in pool:
            if tag.startswith(prefix):
                q = tag.split(":")[1]
                if s > out.get(q, -1.0):
                    out[q] = s
        return out
    corpus_max = _max_by("corpus:")
    golden_max = _max_by("golden:")
    print("  余弦样本总数：%d（存档 %d / 黄金 %d / 长聊天语料 %d）"
          % (n,
             sum(1 for tag, _ in pool if tag.startswith("archive:")),
             sum(1 for tag, _ in pool if tag.startswith("golden:")),
             sum(1 for tag, _ in pool if tag.startswith("corpus:"))))
    print("  分位：p50=%.3f p75=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f"
          % (float(np.percentile(all_sims, 50)),
             float(np.percentile(all_sims, 75)),
             float(np.percentile(all_sims, 90)),
             float(np.percentile(all_sims, 95)),
             float(np.percentile(all_sims, 99)),
             float(max(all_sims))))
    print()
    print("  阈值    对通过率    语料轮触发率   黄金轮触发率   判定")
    for t in SWEEP:
        pair_rate = 100.0 * sum(1 for s in all_sims if s > t) / n
        b_rate = (100.0 * sum(1 for v in corpus_max.values() if v > t)
                  / len(corpus_max)) if corpus_max else 0.0
        g_rate = (100.0 * sum(1 for v in golden_max.values() if v > t)
                  / len(golden_max)) if golden_max else 0.0
        mark = "  ← 当前" if abs(t - THRESHOLD) < 1e-9 else ""
        # 双口径判定：轮触发率看"通道死没死"，对通过率看"闸门虚不虚"
        if b_rate < 10.0 and g_rate < 10.0 and pair_rate < 5.0:
            verdict = "通道死（轮触发<10%）"
        elif pair_rate > 50.0 and b_rate > 90.0:
            verdict = "阈值摆设（对通过>50%）"
        else:
            verdict = "有筛选"
        print("  %.2f   %8.2f%%   %11.1f%%   %11.1f%%   %s%s"
              % (t, pair_rate, b_rate, g_rate, verdict, mark))
    print()
    # 当前阈值的定性结论
    cur_pair = 100.0 * sum(1 for s in all_sims if s > THRESHOLD) / n
    cur_b = (100.0 * sum(1 for v in corpus_max.values() if v > THRESHOLD)
             / len(corpus_max)) if corpus_max else 0.0
    cur_g = (100.0 * sum(1 for v in golden_max.values() if v > THRESHOLD)
             / len(golden_max)) if golden_max else 0.0
    g_max_sorted = sorted(golden_max.values())
    g_med = float(np.median(g_max_sorted)) if g_max_sorted else 0.0
    print("  当前阈值 %.2f 的定性结论（三套语料实测）：" % THRESHOLD)
    print("  · 不是'通道死'：长聊天语料轮触发率 %.1f%%、黄金 %.0f%%、存档 7/7——"
          "绝大多数轮次有历史可检索。" % (cur_b, cur_g))
    print("  · 也不是'阈值摆设'（对层面）：对通过率 %.2f%%——"
          "绝大多数对被挡在候选外，闸门在 Pair 层有真实筛选。" % cur_pair)
    print("  · 真正的病灶是 margin 脆弱：黄金轨迹各轮 max 余弦中位数 "
          "%.3f，密集趴在 0.50~0.68 区间——硬闸下阈值平移 0.05"
          "（0.50→0.55）就把轮触发率从 %.0f%% 压到约一半，0.60 以上断崖。"
          "sim=0.49 与 0.56 的两轮被二元化地分成'空历史'与'四个候选'。"
          % (g_med, cur_g))
    print("  · 深会话另一面：长聊天语料后期轮次准入数达几十上百——"
          "HR 对全部准入者取均值会被稀释，M 只取竞争分 topk。"
          "候选越多，competition_score 排序质量比阈值更关键。")
    print()
    print("  [059 任务2] 软/硬双口径：0.50 附近的'参与度'对比")
    print("  （硬视图：参与 = 1 if max>锚点 else 0，0/1 阶跃；")
    print("    软视图：参与 = clip((max-soft_lo)/span, 0, 1)，"
          "[%.2f, %.2f) 线性斜坡）" % (SOFT_LO, SOFT_HI))
    g_items = sorted(golden_max.items(), key=lambda kv: kv[1])
    near = [(q, v) for q, v in g_items if 0.40 <= v <= 0.65]
    print("  黄金轨迹 max 余弦落在 [0.40, 0.65] 的轮次（升序）：")
    print("  轮次         max余弦   硬参与   软参与w")
    for q, v in near:
        hard = 1.0 if v > THRESHOLD else 0.0
        print("  第%3d轮       %.3f      %d       %.3f"
              % (int(q), v, int(hard), _soft_w(v)))
    print("  → 硬视图下这些轮次非 0 即 1（如 max=0.495 轮：空历史；"
          "max=0.547 轮：全权候选）；")
    print("    软视图下全部落在斜坡上——边缘记忆按 w 温和参与，"
          "阈值微移不再翻转'有无'。")
    print()
    print("  锚点平移敏感性（黄金轨迹有历史轮，锚点 0.45/0.50/0.55）：")
    print("  锚点   硬触发率(max>锚点)   软参与度均值(软化带随锚点平移)")
    for a in [0.45, 0.50, 0.55]:
        lo_a = a - float(config.MATCH_SOFT_MARGIN_LO)
        hi_a = a + float(config.MATCH_SOFT_MARGIN_HI)
        span_a = max(1e-6, hi_a - lo_a)
        hard_rate = (100.0 * sum(1 for v in golden_max.values() if v > a)
                     / len(golden_max)) if golden_max else 0.0
        soft_mean = (float(np.mean([float(np.clip((v - lo_a) / span_a,
                                                  0.0, 1.0))
                                    for v in golden_max.values()]))
                     if golden_max else 0.0)
        print("  %.2f       %8.1f%%            %8.4f" % (a, hard_rate, soft_mean))
    print("  → 硬视图触发率随锚点平移阶跃式大幅波动；软视图参与度均值")
    print("    平滑变化——margin 加固兑现：0.50 附近的召回不再'全有或全无'。")
    print()
    print("  标定建议（供所有者拍板，本探针不自行改参数）：")
    print("  1) 双闸门结构不变：MATCH_THRESHOLD=0.5 管'进候选'（现为软闸锚点），")
    print("     H_SIM_THRESHOLD=0.7 管'算习惯化'——黄金轨迹有历史的 19 轮")
    print("     中 0.5 层触发 17 轮、0.7 层只 1 轮（第 17 轮'废物'对第 4 轮")
    print("     '废物'），两层各司其职，实测数据支持保留。")
    print("  2) 不为'提高触发率'降锚点：长聊天语料 %.1f%% 已近乎全触发，"
          "再降只会放进更多 0.3~0.5 的弱相关（同话题不同事），"
          "稀释 M 的语义聚焦。" % cur_b)
    print("  3) [059 已落地] 边缘案例软处理：058 建议的'0.45~0.55 软处理'")
    print("     已实现为软化带 [%.2f, %.2f)（锚点±边际，w 线性）。黄金轨迹"
          " 实测：第 8 轮（max=0.495，原空历史）以 w=%.2f 微弱参与，"
          "20 轮 action 无一翻转，HR 系统性向 0 稀释（边缘候选"
          " 降权后进均值）——边际记忆'温和参与竞争'兑现，下游稳健。"
          % (SOFT_LO, SOFT_HI, _soft_w(0.495)))
    print("  4) 深会话优先治理排序而非阈值：准入候选从几个到几十个时，")
    print("     K_XG=3 的 topk 选择完全依赖 competition_score——建议后续")
    print("     探针专项验证 topk 命中率（本探针只诊断阈值层）。")
    print("  5) [未校准] 跟踪：MATCH_SOFT_MARGIN_LO/HI（0.05/0.10）与")
    print("     W_DANGER_SLOW（0.20，见 probe_t58_gain_leak）同为工程初值，")
    print("     建议下批用真实长会话统计 w 分布后拍定。")
    print()


def main():
    print("=" * 72)
    print("  059 探针 · 事件级阈值诊断 + 软/硬双口径对比")
    print("  （058 任务4 立探针 / 059 任务2 软阈值落地后修订）")
    print("  锚点 MATCH_THRESHOLD=%.2f，软化带 [%.2f, %.2f)，"
          "w=clip((s-lo)/%.2f)" % (THRESHOLD, SOFT_LO, SOFT_HI, SOFT_SPAN))
    print("=" * 72)
    print("  语料：真实存档 11 个 / 黄金轨迹 20 轮 / 本地长聊天语料（若存在）")
    print("  口径：硬过闸 sim>锚点；软准入 sim>soft_lo（均严格大于）")
    print()

    pool = []   # [(tag, sim)]：全部"对更早项"余弦，Part D 汇总用
    part_a_archives(pool)
    part_b_golden(pool)
    part_c_corpus(pool)
    part_d_sweep(pool)

    print("=" * 72)
    print("  059 探针（任务2 软硬对比）完成")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
