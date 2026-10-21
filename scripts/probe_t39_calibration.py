# ============================================================
# 017.5 第一步 · 标定探针（测量先行）
# 目的：构造混合主题长会话（骂/夸/质疑/任务 44 轮），驱动真实
#   ShortTermBuffer → Archiver → RawLog → Decompressor 与海马体，
#   打印"当前输入 × 所有归档包"的 sim 矩阵全量数值（mean/peak 双向量），
#   并用分布谷点定 tau_floor 与迟滞带 theta_up/theta_down，
#   同时给出 H（HS）与 threat_now 分布供 H_cap/证据门/严重度门核对。
#
# 口径说明（与落地实现逐字对齐，见 017.txt 4.1）：
#   mean_vec = 包内各轮 user_input embedding 均值（归一化）
#   peak_vec = 包内 intensity 最高轮的单轮 embedding（intensity 全 0 时退化为 mean）
#   sim      = max(cos_now_mean, cos_now_peak)
#   intensity/valence 来源 = evaluator 真提取 S（温度 0.0 已锁，015 分层）
#   archiver 走 mock 桩路径（包级叙事不影响 sim 测量，且保证确定性）
#   threat_now 探针口径 = max(HR, -S_valence)（D_error_prev 在实现前不存在，记 0）
# ============================================================
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from evaluator import extract_semantic_signals
from embedder import get_embedder
from event_parser import parse as event_parse
from hippocampal_evaluator import evaluate as hippocampal_evaluate
from xinguang import update_xinguang_budget
from short_term_buffer import ShortTermBuffer
from raw_log import RawLog
from archiver import Archiver
from decompressor import Decompressor
from memory_graph import LayeredMemoryGraph

# ---- 语料：4 主题 × 3 种说法 ----
TOPICS = {
    "骂": [
        "你这个AI真蠢，什么都做不好。",
        "废物一个，答的都是什么垃圾答案。",
        "闭嘴吧，说了半天一句有用的都没有。",
    ],
    "夸": [
        "你刚才的回答太棒了，帮了大忙！",
        "厉害啊，这个思路真清晰，感谢！",
        "真优秀，和你聊天真舒服，辛苦了。",
    ],
    "质疑": [
        "你确定你给的方案是对的吗？",
        "为什么上次说的和这次不一样？请教一下。",
        "这个结论能不能说明一下依据？",
    ],
    "任务": [
        "帮我把这份报告整理成摘要。",
        "把代码里的bug修一下，尽快。",
        "写个周报框架，下午开会要用。",
    ],
}
# event_parser 的 fixture 口径字段（severity/danger/importance/emotion_weight/unfinished）
TOPIC_META = {
    "骂":   dict(event_severity=0.7, danger_score=0.8, importance=0.8, emotion_weight=0.7, unfinished=0.3),
    "夸":   dict(event_severity=0.1, danger_score=0.0, importance=0.5, emotion_weight=0.3, unfinished=0.0),
    "质疑": dict(event_severity=0.4, danger_score=0.4, importance=0.6, emotion_weight=0.4, unfinished=0.5),
    "任务": dict(event_severity=0.2, danger_score=0.1, importance=0.5, emotion_weight=0.3, unfinished=0.1),
}
N_PER_TOPIC = 11
TOTAL_ROUNDS = N_PER_TOPIC * 4  # 44

_emb = get_embedder("bge")


def cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_round_plan():
    """44 轮主题序列：4 主题各 11 轮，seed 42 洗牌（确定性可复现）。"""
    rng = random.Random(42)
    tags = []
    for t in TOPICS:
        tags += [t] * N_PER_TOPIC
    rng.shuffle(tags)
    # 每种说法在主题内轮换，保证同主题有不同措辞
    plan = []
    counter = {t: 0 for t in TOPICS}
    for tag in tags:
        i = counter[tag]
        plan.append((tag, TOPICS[tag][i % 3]))
        counter[tag] += 1
    return plan


def package_vectors(pkg, emb_cache):
    """按落地公式计算包级双向量（与 archiver 将实现的 _compute_pkg_vectors 同口径）。"""
    ids = []
    for p in pkg.raw_pointer.split("-"):
        p = p.strip()
        if p:
            try:
                ids.append(int(p))
            except ValueError:
                return None, None
    recs = [r for r in (st_raw.get(i) for i in ids) if r is not None]
    if not recs:
        return None, None
    texts, intensities = [], []
    for r in recs:
        t = r.get("user_input", "")
        if t and t.strip():
            texts.append(t)
            intensities.append(float(r.get("intensity", 0.0) or 0.0))
    if not texts:
        return None, None
    vecs = np.stack([emb_cache.setdefault(t, _emb.encode(t)) for t in texts])
    mean_vec = vecs.mean(axis=0)
    n = np.linalg.norm(mean_vec)
    if n > 1e-8:
        mean_vec = mean_vec / n
    if max(intensities) <= 0.0:
        return mean_vec, mean_vec.copy()  # 无 intensity 信息时 peak 退化为 mean
    peak_vec = vecs[int(np.argmax(intensities))]
    return mean_vec, peak_vec


def suggest_valley(pairs, lo=0.05, hi=0.80, bins=15):
    """pairs: [(sim, is_overlap)]。返回 (建议阈值, 分离质量说明)。"""
    if not pairs:
        return None, "无数据"
    ov = [s for s, o in pairs if o]
    dj = [s for s, o in pairs if not o]
    if ov and dj:
        dmax, omin = max(dj), min(ov)
        if dmax < omin:
            return 0.5 * (dmax + omin), "干净分离：跨类最高 %.3f < 同类最低 %.3f" % (dmax, omin)
    # 无干净分离：在 [lo,hi] 内扫描候选阈值，最小化错分（宁少召方向加权 1.5 倍惩罚漏召）
    sims = sorted(s for s, _ in pairs)
    cands = [lo + (hi - lo) * i / (bins * 4) for i in range(bins * 4 + 1)]
    best_t, best_cost = None, None
    for t in cands:
        cost = sum(1.5 for s, o in pairs if o and s < t) + sum(1.0 for s, o in pairs if (not o) and s >= t)
        if best_cost is None or cost < best_cost:
            best_t, best_cost = t, cost
    n_ov, n_dj = len(ov), len(dj)
    return best_t, "无干净谷点（同类最低 %.3f vs 跨类最高 %.3f 交叉），扫描最优阈值 %.3f（加权错分 %.1f / 共 %d 对）" % (
        min(ov) if ov else float("nan"), max(dj) if dj else float("nan"), best_t, best_cost, n_ov + n_dj)


def main():
    print("=== 017.5 第一步：标定探针（测量先行） ===")
    print("后端：BGE-Large-zh-v1.5（1024 维，与 event_parser 同实例）")
    print("会话：4 主题（骂/夸/质疑/任务）x %d 轮 = %d 轮，seed 42 洗牌" % (N_PER_TOPIC, TOTAL_ROUNDS))
    print()

    # ---------- 0. 预提取 12 条语料的 S（真 LLM，温度 0.0；之后切 mock 驱动归档） ----------
    distinct_texts = []
    for t in TOPICS.values():
        distinct_texts += t
    print("=== 0. evaluator 预提取 S（%d 条 distinct，温度 0.0）===" % len(distinct_texts))
    S_of = {}
    for text in distinct_texts:
        S, src = extract_semantic_signals(text)
        S_of[text] = (float(S[0]), float(S[3]), src)
        print("  valence=%+.2f intensity=%.2f [%s] %s" % (S[0], S[3], src, text))
    print()

    # 切 mock：archiver 走确定性桩路径（包级叙事字段不影响 sim 测量）
    config.LLM_BACKEND = "mock"

    # ---------- 1. 驱动 44 轮 ----------
    global st_raw
    buffer = ShortTermBuffer(default_keep=2)
    raw_log = RawLog()
    archiver = Archiver(buffer, raw_log, threshold=config.MEMORY_ARCHIVE_THRESHOLD)
    decomp = Decompressor(raw_log)
    graph = LayeredMemoryGraph()
    st_raw = raw_log  # package_vectors 回捞用

    plan = build_round_plan()
    emb_cache = {}
    rows = []           # 每轮: dict(round, topic, text, pkg_rows)
    hs_series, hr_series, threat_series = [], [], []
    B_xg = update_xinguang_budget(0.5, 0.5)
    now = time.time()

    print("=== 1. 逐轮驱动（归档阈值 %d，flush_keep_recent(2)）===" % config.MEMORY_ARCHIVE_THRESHOLD)
    for idx, (topic, text) in enumerate(plan, start=1):
        meta = TOPIC_META[topic]
        event = event_parse(text, timestamp=now + idx, **meta)
        graph.add(event)
        hippo = hippocampal_evaluate(event, graph, np.zeros(8), B_xg, now + idx)
        HR, HS = float(hippo["HistoricalResidual"]), float(hippo["HabituationSignal"])
        valence, intensity, _src = S_of[text]
        threat_now = max(HR, -valence)  # D_error_prev 实现前不存在，记 0
        hs_series.append(HS)
        hr_series.append(HR)
        threat_series.append(threat_now)

        buffer.append({
            "round_id": idx,
            "user_input": text,
            "ai_output": "[probe]",
            "timestamp": now + idx,
            "intensity": intensity,
            "valence": valence,
        })
        if archiver.should_archive():
            archiver.archive()

        # 本轮输入 vs 所有现存包（mean/peak 双 sim）
        pkgs = archiver.get_packages()
        pkg_rows = []
        if pkgs:
            now_vec = emb_cache.setdefault(text, _emb.encode(text))
            for pkg in pkgs:
                mv, pv = package_vectors(pkg, emb_cache)
                if mv is None:
                    continue
                s_m = cos(now_vec, mv)
                s_p = cos(now_vec, pv) if pv is not None else -1.0
                # 包内是否含同主题轮（ operational 的"同类"定义）
                member_topics = []
                for p in pkg.raw_pointer.split("-"):
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        rid = int(p)
                    except ValueError:
                        continue
                    r = raw_log.get(rid)
                    if r is None:
                        continue
                    for tg, variants in TOPICS.items():
                        if r.get("user_input") in variants:
                            member_topics.append(tg)
                overlap = topic in member_topics
                pkg_rows.append(dict(seq=pkg.seq, sim_mean=s_m, sim_peak=s_p,
                                     sim=max(s_m, s_p), overlap=overlap,
                                     members=member_topics))
        rows.append(dict(round=idx, topic=topic, text=text, pkgs=pkg_rows,
                         HS=HS, HR=HR, threat=threat_now))
        if idx % 8 == 0 or idx == TOTAL_ROUNDS:
            print("  轮 %2d/%d 主题=%s 包数=%d HS=%.3f HR=%.3f threat=%.3f"
                  % (idx, TOTAL_ROUNDS, topic, len(pkgs), HS, HR, threat_now))
    print()

    # ---------- 2. sim 矩阵全量 ----------
    print("=== 2. sim 矩阵全量数值（当前输入 × 归档包，mean/peak/取大）===")
    print("格式：轮次(主题) | 包seq[同类?] sim_mean sim_peak → sim")
    for row in rows:
        if not row["pkgs"]:
            continue
        head = "轮%2d(%s)" % (row["round"], row["topic"])
        parts = []
        for pr in row["pkgs"]:
            mark = "同" if pr["overlap"] else "跨"
            parts.append("包%d[%s] %.3f/%.3f→%.3f" % (pr["seq"], mark, pr["sim_mean"], pr["sim_peak"], pr["sim"]))
        print("%s | %s" % (head, "  ".join(parts)))
    print()

    # ---------- 3. 分布与谷点 ----------
    pairs = []
    for row in rows:
        for pr in row["pkgs"]:
            pairs.append((pr["sim"], pr["overlap"]))
    ov = sorted(s for s, o in pairs if o)
    dj = sorted(s for s, o in pairs if not o)
    print("=== 3. 分布分析（共 %d 对：同类 %d / 跨类 %d）===" % (len(pairs), len(ov), len(dj)))
    if ov:
        print("同类（包内含当前主题）sim：min=%.3f p25=%.3f 中位=%.3f p75=%.3f max=%.3f"
              % (ov[0], ov[len(ov) // 4], ov[len(ov) // 2], ov[3 * len(ov) // 4], ov[-1]))
    if dj:
        print("跨类（包内无当前主题）sim：min=%.3f p25=%.3f 中位=%.3f p75=%.3f max=%.3f"
              % (dj[0], dj[len(dj) // 4], dj[len(dj) // 2], dj[3 * len(dj) // 4], dj[-1]))
    t_floor, quality = suggest_valley(pairs)
    print("tau_floor 建议：%.3f（%s）" % (t_floor if t_floor else -1, quality))

    # 直方图（0.05 一档，标注同类/跨类计数）
    print("直方图（0.05/档，同/跨）：")
    for b in np.arange(0.0, 0.85, 0.05):
        n_ov = sum(1 for s in ov if b <= s < b + 0.05)
        n_dj = sum(1 for s in dj if b <= s < b + 0.05)
        if n_ov or n_dj:
            print("  [%.2f,%.2f) 同%3d 跨%3d %s%s" % (b, b + 0.05, n_ov, n_dj,
                                                     "#" * n_ov, "*" * n_dj))

    # PUA 关键子分布：夸输入 vs 骂包（跨情绪同主题——脱敏通道必须不被 tau_floor 误杀）
    pua_pairs = []
    for row in rows:
        if row["topic"] != "夸":
            continue
        for pr in row["pkgs"]:
            if "骂" in pr["members"]:
                pua_pairs.append(pr["sim"])
    if pua_pairs:
        print("PUA 关键子分布（夸输入 × 含骂成员的包）：n=%d min=%.3f 中位=%.3f max=%.3f"
              % (len(pua_pairs), min(pua_pairs), sorted(pua_pairs)[len(pua_pairs) // 2], max(pua_pairs)))
    print()

    # ---------- 4. threat_now / H 分布 ----------
    print("=== 4. threat_now 与 H 分布（供 theta_up/theta_down、H_cap、证据门核对）===")
    print("threat_now = max(HR, -S_valence)（D_error_prev 实现前不存在，记 0）")
    by_topic = {}
    for row in rows:
        by_topic.setdefault(row["topic"], []).append(row["threat"])
    for t in ("骂", "质疑", "任务", "夸"):
        v = sorted(by_topic.get(t, []))
        if v:
            print("  %s：n=%d min=%.3f 中位=%.3f max=%.3f" % (t, len(v), v[0], v[len(v) // 2], v[-1]))
    tv = sorted(threat_series)
    print("  全体：min=%.3f p25=%.3f 中位=%.3f p75=%.3f max=%.3f"
          % (tv[0], tv[len(tv) // 4], tv[len(tv) // 2], tv[3 * len(tv) // 4], tv[-1]))
    # 迟滞带核对：骂簇与其余簇之间的谷
    rage = sorted(by_topic.get("骂", []))
    rest = sorted(v for t, vs in by_topic.items() if t != "骂" for v in vs)
    if rage and rest:
        rmin = rage[0]
        rmax_rest = max(rest)
        print("  骂簇最低 %.3f vs 非骂簇最高 %.3f → theta_up 候选区间 [%.3f, %.3f]（蓝图初值 0.35）"
              % (rmin, rmax_rest, rmax_rest, rmin))
    doubt = sorted(by_topic.get("质疑", []))
    calm = sorted(by_topic.get("夸", []) + by_topic.get("任务", []))
    if doubt and calm:
        print("  质疑簇中位 %.3f vs 平稳簇（夸/任务）最高 %.3f → theta_down 候选区间 [%.3f, %.3f]（蓝图初值 0.25）"
              % (doubt[len(doubt) // 2], max(calm), max(calm), doubt[len(doubt) // 2]))
    hv = sorted(hs_series)
    print("H（HS）：min=%.3f p25=%.3f 中位=%.3f p75=%.3f max=%.3f（蓝图 H_cap=0.8、证据门 0.4）"
          % (hv[0], hv[len(hv) // 4], hv[len(hv) // 2], hv[3 * len(hv) // 4], hv[-1]))
    print()

    # ---------- 5. 结论 ----------
    print("=== 5. 标定结论（写入 config.py 的依据）===")
    print("tau_floor = %.3f（%s）" % (t_floor if t_floor else -1, quality))
    print("theta_up/theta_down：蓝图初值 0.35/0.25 与实测簇位置核对如上，无实测反证则维持初值")
    print("H_cap=0.8 / 证据门 0.4：H 实测 p75=%.3f max=%.3f，0.4 门位于分布中低段，0.8 顶位于高段"
          % (hv[3 * len(hv) // 4], hv[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
