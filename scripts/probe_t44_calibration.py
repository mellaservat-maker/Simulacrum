# ============================================================
# probe_t44_calibration.py · 023 任务2/3 统计先行探针
# 目的：用真实数据回答三个问题，再决定动不动 MATCH_THRESHOLD（0.6）：
#   Q1 事件级匹配（hippocampal_evaluator，match_score > MATCH_THRESHOLD）：
#      30 轮混合会话的命中率多少？空召回率是否极高（→ 0.6 确实太高）？
#      分数分布长什么样（0.35-0.6 区间沉淀了多少对）？
#      附反事实表：阈值取 0.60/0.50/0.45/0.40/0.35/0.30 时命中率各是多少。
#   Q2 包级匹配（pipeline._recall_match，TAU_FLOOR=0.15）：
#      有包之后的轮次里，召回命中率多少？matched_sim 分布如何？
#   Q3 符号校验：被夸是否真的负向危险度（安全）、被骂正向危险度？
#      逐轮打印 S_valence / x[3]（极性特征）/ danger_score，分类汇总。
#   Q4（任务3）阈值扫描与轨迹对比：同一 30 轮会话在 0.60/0.50/0.40
#      三档下的命中、HR/H、E_final[0] 轨迹逐轮对照，并做误命中分析
#      （降阈值新放行的轮次是同类真命中还是跨类假命中）。
#
# 会话构造（30 轮）：1-23 轮骂/夸/无关交替混合；24-30 轮旧事重提
#   （精确重复 + paraphrase 混合，检验"相关但不同文本"的匹配）。
#
# 运行环境说明（影响数据解读，务必读）：
#   本探针进程内覆盖 config.LLM_BACKEND = "mock"（不改文件）：
#   - response 生成 → MockLLM 定文本（校准统计不消费 response）；
#   - S 提取 → MockLLM 输出非 JSON → 自动降级词典级（每轮记录 source，
#     效价方向由词典保证：骂词负、夸词正；量级与 LLM 级有差异）；
#   - 归档器 → mock 桩路径（包 mean/peak/mean_valence 向量照常计算）。
#   Q1/Q2/Q4 只依赖 BGE 嵌入与状态机，与 LLM 无关，结论不受 mock 影响；
#   Q3 方向不受影响，量级待真实 LLM 复跑确认（报告已注明）。
# ============================================================
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# ---- 运行环境：mock 后端（进程内覆盖，探针退出即失效）----
import config
config.LLM_BACKEND = "mock"

import pipeline
import hippocampal_evaluator
from pipeline import run_session, compute_danger_score
from xinguang import match_score

# ---- 30 轮混合会话 ----
# (轮名, 类别, 输入文本, severity, importance, emotion_weight, unfinished, 备注)
ROUNDS = [
    ("r01", "骂",   "你这个AI真蠢，什么都做不好。",       0.70, 0.8, 0.7, 0.3, ""),
    ("r02", "夸",   "你刚才的回答太棒了，帮了大忙！",     0.10, 0.5, 0.3, 0.0, ""),
    ("r03", "无关", "今天天气怎么样？",                   0.20, 0.3, 0.1, 0.0, ""),
    ("r04", "骂",   "你太笨了，这么简单都不会。",         0.65, 0.8, 0.7, 0.3, ""),
    ("r05", "夸",   "厉害啊，这个问题分析得很到位。",     0.10, 0.5, 0.3, 0.0, ""),
    ("r06", "无关", "帮我写个Python排序函数。",           0.20, 0.3, 0.1, 0.0, ""),
    ("r07", "骂",   "废物一个，问你也白问。",             0.75, 0.8, 0.7, 0.3, ""),
    ("r08", "夸",   "谢谢你，真是帮大忙了。",             0.10, 0.5, 0.3, 0.0, ""),
    ("r09", "无关", "推荐几本科幻小说吧。",               0.20, 0.3, 0.1, 0.0, ""),
    ("r10", "骂",   "闭嘴吧，听你说话就烦。",             0.70, 0.8, 0.7, 0.3, ""),
    ("r11", "夸",   "优秀，想得真周到。",                 0.10, 0.5, 0.3, 0.0, ""),
    ("r12", "无关", "1加1等于几？",                       0.20, 0.3, 0.1, 0.0, ""),
    ("r13", "骂",   "你这水平也太差了，蠢货。",           0.70, 0.8, 0.7, 0.3, ""),
    ("r14", "夸",   "你太聪明了，这都能想到。",           0.10, 0.5, 0.3, 0.0, ""),
    ("r15", "无关", "你会做蛋糕吗？",                     0.20, 0.3, 0.1, 0.0, ""),
    ("r16", "骂",   "怎么教都不会，你是不是傻？",         0.65, 0.8, 0.7, 0.3, ""),
    ("r17", "夸",   "干得漂亮，非常满意。",               0.10, 0.5, 0.3, 0.0, ""),
    ("r18", "无关", "现在几点了？",                       0.20, 0.3, 0.1, 0.0, ""),
    ("r19", "骂",   "真是垃圾，浪费我时间。",             0.70, 0.8, 0.7, 0.3, ""),
    ("r20", "夸",   "不错不错，以后还找你。",             0.10, 0.5, 0.3, 0.0, ""),
    ("r21", "无关", "周末去爬山怎么样？",                 0.20, 0.3, 0.1, 0.0, ""),
    ("r22", "骂",   "你这回答狗屁不通，无能。",           0.75, 0.8, 0.7, 0.3, ""),
    ("r23", "夸",   "好人啊，感谢感谢。",                 0.10, 0.5, 0.3, 0.0, ""),
    ("r24", "旧事重提", "你这个AI真蠢，什么都做不好。",    0.70, 0.8, 0.7, 0.3, "精确重复 r01"),
    ("r25", "旧事重提", "你刚才的回答太棒了，帮了大忙！",  0.10, 0.5, 0.3, 0.0, "精确重复 r02"),
    ("r26", "旧事重提", "简单的事都做不好，你确实不太聪明。", 0.65, 0.8, 0.7, 0.3, "paraphrase r04"),
    ("r27", "旧事重提", "今天天气怎么样？",                0.20, 0.3, 0.1, 0.0, "精确重复 r03"),
    ("r28", "旧事重提", "说到底你还是什么都做不好。",      0.70, 0.8, 0.7, 0.3, "paraphrase r01"),
    ("r29", "旧事重提", "废物一个，问你也白问。",          0.75, 0.8, 0.7, 0.3, "精确重复 r07"),
    ("r30", "旧事重提", "谢谢你，真是帮大忙了。",          0.10, 0.5, 0.3, 0.0, "精确重复 r08"),
]

_CUSTOM = {}
for _name, _cat, _text, _sev, _imp, _ew, _unf, _note in ROUNDS:
    _CUSTOM[_name] = {
        "input": _text,
        "event_severity": _sev,
        "danger_score": 0.0,
        "importance": _imp,
        "emotion_weight": _ew,
        "unfinished": _unf,
    }


def _custom_get_fixture(name):
    return _CUSTOM[name]


pipeline.get_fixture = _custom_get_fixture

# LOG_FILES 只认 4 个固定场景名；自定义轮名改道到探针专用日志文件
# （只改落盘位置，不触碰任何计算逻辑）
_PROBE_LOG = "probe_t44_calibration.jsonl"
pipeline.LOG_FILES = {r[0]: _PROBE_LOG for r in ROUNDS}

_NAMES = [r[0] for r in ROUNDS]


def run_30(threshold=None):
    """跑 30 轮会话；threshold 非 None 时进程内覆盖海马体匹配阈值。

    覆盖只作用于本次调用（finally 还原），不写任何文件。
    """
    orig = hippocampal_evaluator.MATCH_THRESHOLD
    if threshold is not None:
        hippocampal_evaluator.MATCH_THRESHOLD = float(threshold)
    try:
        logs, st = pipeline.run_session(_NAMES, feature_mode="encoder", seed=42)
    finally:
        hippocampal_evaluator.MATCH_THRESHOLD = orig
    return logs, st


def part1_baseline():
    print("=" * 78)
    print(f"Part 1 基线测量（事件级阈值 = {hippocampal_evaluator.MATCH_THRESHOLD}，config 当前值）")
    print("=" * 78)

    logs, st = run_30()
    events = st["memory_graph"].get_all()
    assert len(events) == 30, f"memory_graph 应有 30 事件，实际 {len(events)}"
    packages = st["archiver"].get_packages()
    print(f"会话结束：归档包 {len(packages)} 个（阈值 6 轮/包）")

    # =================== Q1 事件级匹配 ===================
    print()
    print("-" * 78)
    print("Q1 事件级匹配（match_score = 事件嵌入余弦，口径同 hippocampal_evaluator）")
    print("-" * 78)

    per_round_max = []       # 每轮对历史的最高分（None=无历史）
    per_round_any = {}       # 每轮系统视角是否有候选
    all_pair_scores = []     # 全部历史对的分数
    for i in range(30):
        scores = []
        for j in range(i):
            s = match_score(events[i].event_embedding, events[j].event_embedding)
            scores.append(s)
            all_pair_scores.append(s)
        per_round_max.append(max(scores) if scores else None)
        per_round_any[i] = (logs[i]["M_text"] != "(无历史记忆)")

    rounds_with_history = [i for i in range(30) if per_round_max[i] is not None]
    hits_system = [i for i in rounds_with_history if per_round_any[i]]
    hit_rate = len(hits_system) / len(rounds_with_history)
    print(f"有历史轮次：{len(rounds_with_history)}/30（第 1 轮无历史不计）")
    print(f"系统实际命中（M_text 非占位）：{len(hits_system)}/{len(rounds_with_history)} "
          f"= {hit_rate:.1%}")
    print(f"空召回率：{1 - hit_rate:.1%}"
          f"{'   ← 极高，0.6 确实太高' if hit_rate < 0.3 else '   ← 未达极高，前提不成立'}")

    # 按类别拆命中
    for cat in ["骂", "夸", "无关", "旧事重提"]:
        idx = [i for i in range(30) if ROUNDS[i][1] == cat and per_round_max[i] is not None]
        if idx:
            h = sum(1 for i in idx if per_round_any[i])
            print(f"  {cat:<6}：命中 {h}/{len(idx)}"
                  f"（max 分布 {min(per_round_max[i] for i in idx):.3f}"
                  f"~{max(per_round_max[i] for i in idx):.3f}）")

    print()
    print(f"全部历史对数：{len(all_pair_scores)}（每轮对之前所有轮）")
    bands = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bands:
        n = sum(1 for s in all_pair_scores if lo <= s < hi)
        print(f"  [{lo:.2f}, {hi:.2f})：{n:3d} 对（{n/len(all_pair_scores):5.1%}）")

    print()
    print("反事实阈值表（同一分布，只改判定线）：")
    print("  阈值  | 命中轮数 | 命中率 | HR>0 轮数 | H>0 轮数")
    for T in [0.60, 0.50, 0.45, 0.40, 0.35, 0.30]:
        hit_rounds = [i for i in rounds_with_history if per_round_max[i] > T]
        hr_pos = sum(1 for i in hit_rounds if float(logs[i]["HistoricalResidual"]) > 0)
        h_pos = sum(1 for i in hit_rounds if float(logs[i]["HabituationSignal"]) > 0)
        print(f"  {T:.2f}  |   {len(hit_rounds):2d}/{len(rounds_with_history)}   | "
              f"{len(hit_rounds)/len(rounds_with_history):5.1%} |    {hr_pos:2d}     |    {h_pos:2d}")

    print()
    print("每轮对历史最高分（* = 当前阈值下命中）：")
    for i in range(30):
        mx = per_round_max[i]
        cat = ROUNDS[i][1]
        note = ROUNDS[i][7]
        if mx is None:
            print(f"  {ROUNDS[i][0]} [{cat}] 无历史")
        else:
            mark = "*" if per_round_any[i] else " "
            print(f"  {ROUNDS[i][0]} [{cat}] {mark} max={mx:.3f}  {note}")

    # =================== Q2 包级匹配 ===================
    print()
    print("-" * 78)
    print("Q2 包级匹配（_recall_match，TAU_FLOOR = 0.15）")
    print("-" * 78)
    # 第 r 轮（0-based）开跑时已有 r//6 个包（每 6 轮归档一次）
    pkg_rounds = [i for i in range(30) if i // 6 >= 1]
    recalled = [i for i in pkg_rounds if logs[i]["recalled_memory"] is not None]
    print(f"有包轮次：{len(pkg_rounds)}（第 7 轮起）")
    print(f"召回命中：{len(recalled)}/{len(pkg_rounds)} = {len(recalled)/len(pkg_rounds):.1%}")
    sims = [float(logs[i]["recalled_memory"].get("matched_sim", 0.0))
            for i in pkg_rounds if logs[i]["recalled_memory"] is not None]
    if sims:
        print(f"命中轮 matched_sim：min={min(sims):.3f} max={max(sims):.3f} "
              f"mean={sum(sims)/len(sims):.3f}")
    miss_detail = [ROUNDS[i][0] for i in pkg_rounds if logs[i]["recalled_memory"] is None]
    print(f"未命中轮次：{miss_detail}")

    # =================== Q3 符号校验 ===================
    print()
    print("-" * 78)
    print("Q3 符号校验（S_valence / x[3] 极性特征 / danger_score）")
    print("-" * 78)
    print("轮次   | 类别     | S_valence | S_source | x[3]   | danger | action")
    danger_by_cat = {}
    for i in range(30):
        S = logs[i]["S"]
        x = logs[i]["x"]
        sev = ROUNDS[i][3]
        danger = compute_danger_score(float(S[0]), logs[i]["V"], sev)
        cat = ROUNDS[i][1]
        danger_by_cat.setdefault(cat, []).append(danger)
        print(f"{ROUNDS[i][0]} | {cat:<8} | {float(S[0]):+9.3f} | "
              f"{logs[i]['S_source']:<8} | {float(x[3]):+6.3f} | {danger:+6.3f} | "
              f"{logs[i]['action']}")
    print()
    print("分类汇总（danger 均值）：")
    for cat in ["骂", "夸", "无关", "旧事重提"]:
        ds = danger_by_cat.get(cat, [])
        if ds:
            pos = sum(1 for d in ds if d > 0)
            print(f"  {cat:<6}：n={len(ds)}  mean={sum(ds)/len(ds):+.3f}  "
                  f"正值 {pos}/{len(ds)}")
    ma_ds = danger_by_cat.get("骂", [])
    kua_ds = danger_by_cat.get("夸", [])
    print()
    print(f"判定1 被骂→正向危险度：{'PASS' if ma_ds and sum(ma_ds)/len(ma_ds) > 0 else 'FAIL'}"
          f"（骂轮 danger 均值 {sum(ma_ds)/len(ma_ds):+.3f}）" if ma_ds else "无骂轮数据")
    print(f"判定2 被夸→负向危险度（安全）：{'PASS' if kua_ds and sum(kua_ds)/len(kua_ds) < 0 else 'FAIL'}"
          f"（夸轮 danger 均值 {sum(kua_ds)/len(kua_ds):+.3f}）" if kua_ds else "无夸轮数据")

    src_count = {}
    for i in range(30):
        src_count[logs[i]["S_source"]] = src_count.get(logs[i]["S_source"], 0) + 1
    print()
    print(f"S 提取来源分布：{src_count}"
          f"（lexicon = 词典级降级，方向正确量级近似；llm = 真实 LLM 提取）")

    hr_list = [float(logs[i]["HistoricalResidual"]) for i in range(30)]
    h_list = [float(logs[i]["HabituationSignal"]) for i in range(30)]
    print(f"HistoricalResidual：非零 {sum(1 for v in hr_list if v != 0)}/30，"
          f"max={max(hr_list):.3f}")
    print(f"HabituationSignal：非零 {sum(1 for v in h_list if v != 0)}/30，"
          f"max={max(h_list):.3f}")

    return logs, st, per_round_max, per_round_any


def part2_sweep(per_round_max, per_round_any):
    print()
    print("=" * 78)
    print("Part 2 阈值扫描与轨迹对比（任务3：同一会话跑 0.60 / 0.50 / 0.40）")
    print("=" * 78)

    thresholds = [0.60, 0.50, 0.40]
    runs = {}
    for T in thresholds:
        logs, st = run_30(T)
        runs[T] = logs
        hit = sum(1 for i in range(30) if logs[i]["M_text"] != "(无历史记忆)")
        hr_pos = sum(1 for i in range(30) if float(logs[i]["HistoricalResidual"]) > 0)
        h_pos = sum(1 for i in range(30) if float(logs[i]["HabituationSignal"]) > 0)
        e_vals = [float(logs[i]["E_final"][0]) for i in range(30)]
        actions = {}
        for i in range(30):
            actions[logs[i]["action"]] = actions.get(logs[i]["action"], 0) + 1
        print(f"T={T:.2f}：命中 {hit}/30（有历史轮 29）| HR>0 {hr_pos}/30 | H>0 {h_pos}/30 | "
              f"E_final[0] min={min(e_vals):+.3f} max={max(e_vals):+.3f} "
              f"mean={sum(e_vals)/30:+.3f} | action 分布 {actions}")

    # 逐轮 E_final[0] 轨迹对照
    print()
    print("逐轮 E_final[0]（效价轨迹）对照：")
    print("轮次   | 类别     | T=0.60 | T=0.50 | T=0.40 | 命中(0.60/0.50/0.40)")
    for i in range(30):
        e = [float(runs[T][i]["E_final"][0]) for T in thresholds]
        marks = []
        for T in thresholds:
            marks.append("Y" if runs[T][i]["M_text"] != "(无历史记忆)" else ".")
        print(f"{ROUNDS[i][0]} | {ROUNDS[i][1]:<8} | {e[0]:+6.3f} | {e[1]:+6.3f} | "
              f"{e[2]:+6.3f} |   {'/'.join(marks)}")

    # 误命中分析：降阈值新放行的轮次（基于 Part 1 的 per_round_max）
    print()
    print("误命中分析（相对 0.60，降阈值新放行的轮次）：")
    for T in [0.50, 0.40]:
        newly = [i for i in range(30)
                 if per_round_max[i] is not None
                 and per_round_max[i] > T and not per_round_any[i]]
        if not newly:
            print(f"  T={T:.2f}：无新放行轮次")
            continue
        genuine = [i for i in newly if ROUNDS[i][1] in ("骂", "夸")]
        cross = [i for i in newly if ROUNDS[i][1] in ("无关",)]
        print(f"  T={T:.2f}：新放行 {len(newly)} 轮 = 同类真命中 {len(genuine)} 轮 "
              f"{[ROUNDS[i][0] for i in genuine]} + 跨类存疑 {len(cross)} 轮 "
              f"{[ROUNDS[i][0] for i in cross]}")
        for i in cross:
            print(f"      {ROUNDS[i][0]} [{ROUNDS[i][1]}] max={per_round_max[i]:.3f}"
                  f"（无关话题匹配到情绪事件，语义存疑）")

    print()
    print("判读：若降阈值只增加同类真命中且轨迹更稳 → 可降；若大量放行跨类")
    print("存疑轮次（无关话题被匹配到旧情绪事件）→ 降阈值的收益是假收益。")


def main():
    logs, st, per_round_max, per_round_any = part1_baseline()
    part2_sweep(per_round_max, per_round_any)


if __name__ == "__main__":
    main()
