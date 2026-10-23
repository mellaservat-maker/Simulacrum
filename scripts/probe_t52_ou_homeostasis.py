# ============================================================
# probe_t52_ou_homeostasis.py · 050 · 情绪基线回拉探针
#
# 目的：只看数据，不改代码。回答一个问题：
#   持续 0.02 的微小负向输入下，情绪会不会被「拖偏」到一个非零静息点？
#   加一个向零的回归项（OU 过程）后，静息点和恢复速度会怎么变？
#
# 模型（与 emotion.py 同口径，隔离掉 B 偏离项，只看效价一维）：
#   E_{t+1} = clip( LAMBDA * E_t + impact - gamma * E_t, -1, 1 )
# 其中 LAMBDA = 0.9（config），impact 为每轮释放冲击（本探针固定 -0.02）。
#
# 注意一个代数事实（本探针顺带验证）：
#   LAMBDA*E - gamma*E = (LAMBDA - gamma)*E
# 因此「向零回归」在数学上等价于把保留系数从 0.9 调低到 0.9-gamma。
#   它的作用是降低稳态偏移与缩短尾巴，而不是把基线拉回某个「个人基线」。
#   若未来要的是「回到个人基线 e_base」，需要的是 E_{t+1} = e_base +
#   (LAMBDA-gamma)*(E_t - e_base) + impact，那是另一个机制，需单独裁决。
#
# 两段式场景：
#   阶段一 30 轮：每轮 impact = -0.02（模拟连续说「你好无聊」这类轻微负反馈）
#   阶段二 20 轮：impact = 0（停止输入，观察回血速度）
#
# 输出：每个 gamma 的完整 E[0] 曲线、阶段末稳态、停止输入后回到
#   |E| <= 0.05 与 <= 0.01 所需轮数。
# [051] 追加「真实链路复核」段：用 emotion.run_emotion（线上同一个函数）
#   跑同场景，与纯数学推导逐轮对比，确认落地后曲线未偏离。
#
# 运行：
#   d:\aiqg\venv\Scripts\python.exe scripts\probe_t52_ou_homeostasis.py
#   可选：--impact -0.02 --rounds 30 --recover 20 --gammas 0,0.02,0.05,0.1
# ============================================================

import argparse
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config  # noqa: E402

LAMBDA = float(config.LAMBDA[0]) if hasattr(config.LAMBDA, "__len__") else float(config.LAMBDA)


def simulate(gamma, impact, rounds, recover, e0=0.0):
    """返回 (阶段一曲线, 阶段二曲线)。每个元素是 (轮次, E[0])。"""
    E = float(e0)
    curve_in = []
    for t in range(1, rounds + 1):
        E = max(-1.0, min(1.0, LAMBDA * E + impact - gamma * E))
        curve_in.append((t, E))
    curve_out = []
    for t in range(1, recover + 1):
        E = max(-1.0, min(1.0, LAMBDA * E - gamma * E))
        curve_out.append((rounds + t, E))
    return curve_in, curve_out


def rounds_to_threshold(curve_out, start_value, thresh):
    """从停止输入那一刻算起，|E| 首次小于等于 thresh 的轮数；未到返回 None。"""
    if abs(start_value) <= thresh:
        return 0
    for i, (_t, e) in enumerate(curve_out, start=1):
        if abs(e) <= thresh:
            return i
    return None


def _fmt(v):
    return "%+.4f" % v


def verify_real_chain(impact, rounds, recover):
    """[051] 用线上 emotion.run_emotion 跑同场景，与解析式逐轮比对。

    B 取 B_INIT（B 偏离项为 0），冲击只加在效价一维，其他三维为 0。
    返回 (max_abs_diff, 末值)。
    """
    import emotion

    gamma = float(config.E_HOMEOSTASIS_GAMMA)
    E = np.zeros(4, dtype=float)
    formula = np.zeros(4, dtype=float)
    max_diff = 0.0
    for t in range(rounds + recover):
        imp = np.array([impact, 0.0, 0.0, 0.0]) if t < rounds else np.zeros(4)
        E = emotion.run_emotion(E, imp, config.B_INIT)
        formula = np.clip((config.LAMBDA - gamma) * formula + imp, -1.0, 1.0)
        max_diff = max(max_diff, float(np.max(np.abs(E - formula))))
    return gamma, max_diff, float(E[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description="T52 情绪基线回拉探针（只出数据）")
    ap.add_argument("--impact", type=float, default=-0.02, help="每轮冲击（默认 -0.02）")
    ap.add_argument("--rounds", type=int, default=30, help="阶段一持续输入轮数")
    ap.add_argument("--recover", type=int, default=20, help="阶段二停止输入轮数")
    ap.add_argument("--gammas", default="0,0.02,0.05,0.1",
                    help="回归系数列表，逗号分隔")
    args = ap.parse_args(argv)

    gammas = [float(x) for x in args.gammas.split(",") if x.strip() != ""]

    print("=" * 74)
    print("T52 情绪基线回拉探针（051 / 050 号工作）")
    print("=" * 74)
    print("模型   : E[t+1] = clip(LAMBDA*E[t] + impact - gamma*E[t], -1, 1)")
    print("参数   : LAMBDA=%.2f  impact=%+.4f  连续 %d 轮，然后停止输入 %d 轮"
          % (LAMBDA, args.impact, args.rounds, args.recover))
    print("等价性 : gamma 项等价于把保留系数变为 %.2f-gamma，见文件头注释" % LAMBDA)
    print("")

    summary = []
    for g in gammas:
        curve_in, curve_out = simulate(g, args.impact, args.rounds, args.recover)
        steady = curve_in[-1][1]
        low = min(e for _t, e in curve_in)
        n05 = rounds_to_threshold(curve_out, steady, 0.05)
        n01 = rounds_to_threshold(curve_out, steady, 0.01)
        summary.append((g, steady, low, n05, n01, curve_out[-1][1]))

        print("-" * 74)
        print("gamma = %.3f    （有效保留系数 = %.3f）" % (g, LAMBDA - g))
        print("-" * 74)
        print("阶段一：连续 %d 轮 impact=%+.3f（每行 10 轮）" % (args.rounds, args.impact))
        for i in range(0, len(curve_in), 10):
            chunk = curve_in[i:i + 10]
            print("  第 %2d-%2d 轮: %s"
                  % (chunk[0][0], chunk[-1][0],
                     "  ".join(_fmt(e) for _t, e in chunk)))
        print("阶段二：停止输入 %d 轮（同 gamma，无冲击）" % args.recover)
        for i in range(0, len(curve_out), 10):
            chunk = curve_out[i:i + 10]
            print("  第 %2d-%2d 轮: %s"
                  % (chunk[0][0], chunk[-1][0],
                     "  ".join(_fmt(e) for _t, e in chunk)))
        print("")
        print("  阶段末静息值 (第 %d 轮) : %s" % (args.rounds, _fmt(steady)))
        print("  全程最低点              : %s" % _fmt(low))
        print("  停止输入后回到 |E|<=0.05: %s"
              % ("%d 轮" % n05 if n05 is not None else "未达到"))
        print("  停止输入后回到 |E|<=0.01: %s"
              % ("%d 轮" % n01 if n01 is not None else "未达到"))
        print("  阶段二结束时            : %s" % _fmt(curve_out[-1][1]))
        print("")

    print("=" * 74)
    print("汇总")
    print("=" * 74)
    print("gamma   稳态E0    最低点    回0.05轮数   回0.01轮数   终值")
    for g, steady, low, n05, n01, final in summary:
        print("%.3f   %s   %s   %-10s   %-10s   %s"
              % (g, _fmt(steady), _fmt(low),
                 str(n05) if n05 is not None else "未达到",
                 str(n01) if n01 is not None else "未达到",
                 _fmt(final)))
    print("")
    print("读法提示：")
    print("1. gamma=0 时稳态 = impact / (1 - LAMBDA) = 冲击的 10 倍，这就是静息点被拖偏。")
    print("2. gamma>0 时稳态 = impact / (1 - LAMBDA + gamma)，偏移随 gamma 单调减小。")
    print("3. 停止输入后，gamma=0 也能靠乘性遗忘衰减；加 gamma 只是把尾巴缩短。")
    print("4. 因此「10-20 轮回血到 0 附近」在 gamma=0 时也可能成立，")
    print("   真正被 OU 项改善的是持续输入期间的稳态偏移，不是停后的恢复。")
    print("")

    # [051] 落地后复核：线上 emotion.run_emotion 与解析式是否一致
    print("=" * 74)
    print("真实链路复核（051 落地后）")
    print("=" * 74)
    print("config.E_HOMEOSTASIS_GAMMA = %.4f" % config.E_HOMEOSTASIS_GAMMA)
    print("调用 emotion.run_emotion（线上同一函数），B=B_INIT 隔离 B 偏离项；")
    print("场景同上：%d 轮 impact=%+.3f，再 %d 轮停输入。"
          % (args.rounds, args.impact, args.recover))
    gamma_cfg, max_diff, end_val = verify_real_chain(
        args.impact, args.rounds, args.recover)
    print("  线上函数与解析式逐轮最大偏差：%.3e" % max_diff)
    print("  阶段二末 E[0]（线上函数）：%s" % _fmt(end_val))
    ok = max_diff < 1e-12
    print("  结论：%s" % ("一致（偏差在浮点容差内）" if ok
                       else "不一致，需排查 emotion.py"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
