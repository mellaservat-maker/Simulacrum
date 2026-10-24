# ============================================================
# probe_t54_semantic_res.py · 054 号工作 · 任务 2
#
# 问题：u_s@M 是近常数。实测（053）18 条真实 BGE 向量上
#   0.1·ΣM ≈ -0.075，全库波动只有 0.047 —— 1024 维语义进慢通路
#   只剩一个偏置，M 对 z_s 几乎没有区分能力。
#
# 根因（本探针要验证的假设）：BGE 向量高度各向异性，所有向量都带一个
#   共同方向（Σv 恒为负、量级相近）。均分权重 u_s = 0.1·ones(1024)
#   恰好把 M 投影到"全 1 方向"，而该方向上的分量对任何话题都差不多，
#   于是 u_s@M 被这个共模项锁死。
#
# 验证方式：构造 6 个话题 × 5 条 = 30 条语料，对每条语料模拟海马体
#   topk（取最相似的 3 条求均值并归一化）得到 M，再比较四种读出的
#   区分度：
#     A 现状        u_s = 0.1·ones，M 原始
#     B 减均值基    u_s = 0.1·ones，M_c = M - mean_basis
#     C 归一化读出   u_s = ones/√1024，M_c（把 B 的尺度归一）
#     D 随机投影    u_s = 固定种子的零均值单位向量，M_c
#
# 判定指标（关键）：
#   std_total      全部 30 条的标准差（越大越好）
#   std_between    话题间标准差（真正想放大的信号）
#   std_within     话题内标准差（噪声）
#   信噪比         std_between / std_within（>1 才算真的能区分话题）
#
# 用法：python scripts/probe_t54_semantic_res.py
# ============================================================

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config                       # noqa: E402
from embedder import get_embedder    # noqa: E402

TOPICS = {
    "代码报错": [
        "这段代码报错了，说 KeyError。",
        "为什么我的 Python 脚本一运行就抛异常？",
        "这个函数返回 None，是不是哪里写错了。",
        "安装依赖的时候报版本冲突。",
        "调试了半天，还是找不到 bug 在哪。",
    ],
    "情绪心情": [
        "今天心情特别低落，什么都不想干。",
        "我最近压力很大，晚上睡不着。",
        "有点难过，但说不清为什么。",
        "感觉很累，想找个人说说话。",
        "刚才被夸了，心情一下子好了。",
    ],
    "天气": [
        "今天天气怎么样，适合出门吗？",
        "外面下雨了，记得带伞。",
        "这几天降温，注意保暖。",
        "明天是晴天还是阴天？",
        "台风要来了，航班可能取消。",
    ],
    "做饭": [
        "红烧肉要炖多久才软？",
        "番茄炒蛋先放蛋还是先放番茄？",
        "这个菜太咸了，怎么补救。",
        "推荐一个简单的家常菜做法。",
        "煮米饭水和米的比例是多少。",
    ],
    "股票": [
        "今天大盘涨了还是跌了？",
        "这只股票的市盈率高不高？",
        "最近行情不好，要不要减仓。",
        "帮我看看这个板块的走势。",
        "财报出来了，营收增长多少？",
    ],
    "电影": [
        "推荐一部最近好看的科幻电影。",
        "这部电影的结局是什么意思？",
        "导演的手法我特别喜欢。",
        "有没有类似题材的片子？",
        "这个演员演技真不错。",
    ],
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _std(a):
    return float(np.std(np.asarray(a, dtype=float)))


def main():
    emb = get_embedder("bge")
    texts, labels = [], []
    for topic, items in TOPICS.items():
        for t in items:
            texts.append(t)
            labels.append(topic)
    V = np.stack([_unit(emb.encode(t)) for t in texts])       # (30, 1024)
    labels = np.array(labels)

    # 会话均值基（生产里是 memory_graph 全部事件向量的均值）
    mean_basis = V.mean(axis=0)
    print("语料 %d 条 / %d 个话题" % (len(texts), len(TOPICS)))
    print("均值基 ‖mean_basis‖ = %.4f，Σ(mean_basis) = %.4f"
          % (np.linalg.norm(mean_basis), mean_basis.sum()))
    print("单条向量 Σv：min %.3f / mean %.3f / max %.3f"
          % (V.sum(axis=1).min(), V.sum(axis=1).mean(), V.sum(axis=1).max()))

    # 模拟海马体 topk：每条语料取最相似的 3 条（含自身）求均值归一化
    Ms = []
    for i in range(len(V)):
        sims = V @ V[i]
        top = np.argsort(-sims)[:3]
        Ms.append(_unit(V[top].mean(axis=0)))
    M = np.stack(Ms)
    Mc = np.stack([_unit(m - mean_basis) for m in M])   # 减均值基后再归一化

    u_uniform = np.full(1024, 0.1)
    u_norm = np.full(1024, 1.0 / np.sqrt(1024.0))
    rng = np.random.RandomState(42)
    u_rand = rng.randn(1024)
    u_rand = u_rand - u_rand.mean()          # 零均值
    u_rand = u_rand / np.linalg.norm(u_rand)

    variants = [
        ("A 现状（u=0.1·ones, M 原始）", M @ u_uniform),
        ("B 减均值基（u=0.1·ones, M_c）", Mc @ u_uniform),
        ("C 归一化读出（u=ones/√1024, M_c）", Mc @ u_norm),
        ("D 随机零均值投影（u_rand, M_c）", Mc @ u_rand),
    ]

    print("\n%-34s %9s %9s %9s %9s %9s"
          % ("方案", "mean", "std_all", "std_btw", "std_win", "SNR"))
    print("-" * 82)
    rows = {}
    for name, vals in variants:
        vals = np.asarray(vals, dtype=float)
        means = [vals[labels == tp].mean() for tp in TOPICS]
        withins = [vals[labels == tp].std() for tp in TOPICS]
        std_all = _std(vals)
        std_btw = _std(means)
        std_win = float(np.mean(withins))
        snr = (std_btw / std_win) if std_win > 1e-12 else float("inf")
        rows[name] = dict(std_all=std_all, std_btw=std_btw, std_win=std_win,
                          snr=snr, vals=vals)
        print("%-34s %9.5f %9.5f %9.5f %9.5f %9.2f"
              % (name, float(vals.mean()), std_all, std_btw, std_win, snr))

    base = rows["A 现状（u=0.1·ones, M 原始）"]
    print("\n相对现状 A 的改善倍数：")
    for name, r in rows.items():
        if name == "A 现状（u=0.1·ones, M 原始）":
            continue
        print("  %-34s std_all ×%.1f   SNR ×%s"
              % (name,
                 (r["std_all"] / base["std_all"]) if base["std_all"] > 1e-12 else float("nan"),
                 ("%.1f" % (r["snr"] / base["snr"])) if base["snr"] > 1e-12 else "inf"))

    print("\n各话题的读出值（B 方案，看话题是否真的分开）：")
    vals = rows["B 减均值基（u=0.1·ones, M_c）"]["vals"]
    for tp in TOPICS:
        seg = vals[labels == tp]
        print("  %-8s mean %+0.5f  min %+0.5f  max %+0.5f  std %.5f"
              % (tp, seg.mean(), seg.min(), seg.max(), seg.std()))

    print("\n各话题的读出值（A 现状，对照）：")
    vals = rows["A 现状（u=0.1·ones, M 原始）"]["vals"]
    for tp in TOPICS:
        seg = vals[labels == tp]
        print("  %-8s mean %+0.5f  min %+0.5f  max %+0.5f  std %.5f"
              % (tp, seg.mean(), seg.min(), seg.max(), seg.std()))

    print("\n结论口径：std_all 至少放大一个量级、且 SNR > 1，才算恢复语义分辨率。")
    print("当前 config: u_s 均分 0.1 = %s，SEMANTIC_CENTERING = %s"
          % (bool(np.allclose(config.u_s, 0.1)),
             getattr(config, "SEMANTIC_CENTERING", "（未实现）")))

    # ---- 实现后实测：直接读 config，证明落地的就是 D 方案 ----
    print("\n实现后实测（直接读 config.u_s / config.U_v）")
    print("-" * 82)
    u_cfg = np.asarray(config.u_s, dtype=float)
    centered = bool(getattr(config, "SEMANTIC_CENTERING", False))
    X = Mc if centered else M
    vals = X @ u_cfg
    means = [vals[labels == tp].mean() for tp in TOPICS]
    withins = [vals[labels == tp].std() for tp in TOPICS]
    std_all = _std(vals)
    std_btw = _std(means)
    std_win = float(np.mean(withins))
    snr = (std_btw / std_win) if std_win > 1e-12 else float("inf")
    print("  ‖u_s‖ = %.4f   u_s 均值 = %+.6f（应为 0）" % (np.linalg.norm(u_cfg), u_cfg.mean()))
    print("  SEMANTIC_CENTERING = %s" % centered)
    print("  mean %+.5f   std_all %.5f   std_btw %.5f   std_win %.5f   SNR %.2f"
          % (float(vals.mean()), std_all, std_btw, std_win, snr))
    if base["std_all"] > 1e-12:
        print("  相对 053 口径（A）：std_all ×%.1f，SNR %.2f -> %.2f"
              % (std_all / base["std_all"], base["snr"], snr))

    # U_v 两行是否还共模（053 的问题：两行同为均分 → 完全相关）
    v0 = X @ np.asarray(config.U_v[0], dtype=float)
    v1 = X @ np.asarray(config.U_v[1], dtype=float)
    corr = float(np.corrcoef(v0, v1)[0, 1]) if v0.std() > 1e-12 and v1.std() > 1e-12 else 0.0
    old0 = M @ np.full(1024, 0.1)
    old1 = M @ np.full(1024, 0.1)
    print("  U_v 两行相关性：旧 %.2f -> 新 %.2f（接近 0 才算解除共模）"
          % (float(np.corrcoef(old0, old1)[0, 1]), corr))
    print("  U_v 行内 std：行0 %.5f / 行1 %.5f" % (float(v0.std()), float(v1.std())))
    # 两行相关性由数据（话题子空间低秩）决定，正交权重也压不下去；
    # 真正影响 danger 的是 V[0]-V[1]，直接量它的波动。
    diff_new = v0 - v1
    diff_old = old0 - old1
    print("  V[0]-V[1] 的 std：旧 %.5f -> 新 %.5f（这才是 danger 的输入）"
          % (float(diff_old.std()), float(diff_new.std())))

    print("\n各话题读出值（实现后）：")
    for tp in TOPICS:
        seg = vals[labels == tp]
        print("  %-8s mean %+0.5f  min %+0.5f  max %+0.5f  std %.5f"
              % (tp, seg.mean(), seg.min(), seg.max(), seg.std()))


if __name__ == "__main__":
    main()
