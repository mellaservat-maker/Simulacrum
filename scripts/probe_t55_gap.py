# ============================================================
# probe_t55_gap.py · 055 号工作 · 任务 3
#
# 目的：把「隔 3 天」和「隔 30 轮」放到同一把尺子上，证明它们是不同权重。
#
# 背景：053/054 的时间衰减用"轮"，跨会话完全没有时间感。055 接入 saved_at
#   之后，四条通道各自用真实天数衰减：
#     EMOTION  E_prev 空档消退（tau=1 天）
#     FATIGUE  w_PFC 空档恢复（tau=2 天）
#     HABIT    H 空档消退（tau=3 天）
#     RECALL   包级置信度折扣（tau=7 天，全局同乘数，不改排序）
#
# 对照口径：会话内 1 轮的情绪衰减是 ×(LAMBDA - GAMMA_HOME) = ×0.85。
#   于是可以把"空档天数"换算成"等效轮数"：
#     n 轮后 E 剩 0.85^n；d 天空档后 E 剩 exp(-d/tau)
#   两者相等时 n = d / (tau × -ln0.85)。
#
# 用法：python scripts/probe_t55_gap.py
# ============================================================

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config    # noqa: E402

KEEP = float(config.LAMBDA[0]) - float(config.E_HOMEOSTASIS_GAMMA)   # 0.85


def _equiv_rounds(days, tau):
    """d 天空档 ≡ 多少轮连续对话（情绪通道口径）。"""
    if tau <= 0:
        return float("inf")
    return days / (tau * (-math.log(KEEP)))


def main():
    print("会话内每轮情绪保留系数 KEEP = %.2f" % KEEP)
    print("（1 轮 = ×%.2f，即 1 轮消掉 %.1f%%）" % (KEEP, (1 - KEEP) * 100))
    print()

    print("一、四条跨天通道的衰减常数与剩余比例")
    print("-" * 70)
    print("  %-8s %-10s %8s %8s %8s %8s" %
          ("通道", "tau(天)", "1天", "3天", "7天", "30天"))
    rows = [
        ("EMOTION", config.TAU_GAP_DAYS_EMOTION),
        ("FATIGUE", config.TAU_GAP_DAYS_FATIGUE),
        ("HABIT", config.TAU_GAP_DAYS_HABIT),
        ("RECALL", config.TAU_GAP_DAYS_RECALL),
    ]
    for name, tau in rows:
        cells = [math.exp(-d / tau) if tau > 0 else 1.0 for d in (1, 3, 7, 30)]
        print("  %-8s %-10s %8.3f %8.3f %8.3f %8.3f"
              % (name, tau, cells[0], cells[1], cells[2], cells[3]))

    print()
    print("二、「隔 3 天」等效于多少轮连续对话（情绪通道口径）")
    print("-" * 70)
    for d in (1, 3, 7, 30):
        n = _equiv_rounds(d, config.TAU_GAP_DAYS_EMOTION)
        print("  隔 %-3d 天 ≈ 连续聊 %5.1f 轮的情绪消退量（剩 %.3f）"
              % (d, n, math.exp(-d / config.TAU_GAP_DAYS_EMOTION)))

    print()
    print("三、30 轮 vs 3 天的直接对比（这才是问题所在）")
    print("-" * 70)
    e_30r = KEEP ** 30
    e_3d = math.exp(-3.0 / config.TAU_GAP_DAYS_EMOTION)
    print("  连续 30 轮（同一会话，中间没有空档）：E 剩 %.5f" % e_30r)
    print("  隔 3 天（一轮都没聊）：            E 剩 %.5f" % e_3d)
    print("  隔 3 天 ≈ 连续聊 %.1f 轮（见第二节），比 30 轮消得少 %.1f 倍"
          % (_equiv_rounds(3.0, config.TAU_GAP_DAYS_EMOTION), e_3d / e_30r))
    print("  关键不在谁大谁小，而在 055 之前「隔 3 天」这一栏是 %.5f："
          % 1.0)
    print("  你不在的时候系统一轮都没跑，E_prev / w_PFC / H 原封不动躺着，")
    print("  于是「气三天」和「没气过」在系统里是同一件事。现在它变成了 0.05。")

    print()
    print("四、包级匹配的跨天折扣（不改排序，只压低置信度）")
    print("-" * 70)
    for d in (0, 3, 7, 30):
        f = math.exp(-d / config.TAU_GAP_DAYS_RECALL) if d else 1.0
        print("  空档 %-3d 天：matched_sim ×%.3f（argmax 排序不变）" % (d, f))


if __name__ == "__main__":
    main()
