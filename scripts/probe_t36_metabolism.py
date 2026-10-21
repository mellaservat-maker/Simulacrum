# ============================================================
# 014.5 任务 2 探针：区分场景的疲劳代谢
# 链路级验证（真 S 链路，pipeline 默认行为，不强制 zeros）：
#   段 1：连骂 10 轮 —— 受压期间疲劳只涨不跌
#   段 2：连夸 20 轮 —— 安全期间疲劳每轮自然消退 W_PFC_RECOVERY_RATE
# 断言：
#   A. 段 1 每轮 w_PFC_new >= w_PFC（受压不降，绝无恢复混入）
#   B. 段 2 每轮 w_PFC_new == w_PFC - W_PFC_RECOVERY_RATE（安全恢复精确线性）
#   C. 段 2 恢复总量 == 20 * W_PFC_RECOVERY_RATE（未被受压轮打断）
#   D. 公式恒等：每轮用 config 参数重算 w_PFC_new 与 log 写回一致
#   E. 段 1 全部轮次判定为受压（I>0 且 D_error>=阈值），段 2 全部为安全轮
# 注：被夸场景的真 S 链路 E_raw[0] 转正（evaluator valence>0 → c_E 注入），
#     neg_intensity=0 → D_error=0 → 安全轮；S=zeros 对照组被夸 E 仍为负
#     （011 效价轴恒负问题），会误判受压——故本探针必须走真 S。
# ============================================================
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from config import (W_PFC_RECOVERY_RATE, W_PFC_RECOVERY_THRESHOLD, RHO, DELTA,
                    KAPPA, W_MAX)

SCENARIOS = ["被骂"] * 10 + ["被夸"] * 20


def check(name, cond, detail=""):
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    print("=== 014.5 任务 2：区分场景的疲劳代谢 ===")
    print("代谢参数：W_PFC_RECOVERY_RATE=%.3f，W_PFC_RECOVERY_THRESHOLD=%.2f" %
          (W_PFC_RECOVERY_RATE, W_PFC_RECOVERY_THRESHOLD))
    print("规则：安全轮（D_error<阈值 或 I==0）→ max(0, w-0.01)；"
          "受压轮（I>0 且 D_error>=阈值）→ clip(RHO*w+DELTA*I, 0, W_MAX)")
    print()

    recs, st = pipeline.run_session(SCENARIOS, feature_mode="fixture", seed=42)
    assert len(recs) == 30, "应跑 30 轮，实际 %d" % len(recs)

    w_prev = [0.0] + [float(r["w_PFC"]) for r in recs[:-1]]  # 每轮进入时的 w_PFC（首轮为 W_PFC_INIT=0.0）
    w_series = [float(r["w_PFC"]) for r in recs]              # 每轮写回后的 w_PFC
    de_series = [float(r["D_error"]) for r in recs]
    i_series = [float(r["I_inhibit"]) for r in recs]
    act_series = [r["action"] for r in recs]

    # ---------- 段 1：被骂 10 轮 ----------
    print("=== 段 1：连骂 10 轮（受压期间只涨不跌）===")
    print("轮次 | D_error | I_inhibit | w_PFC(入→出)   | 判定   | action")
    phase1_ok = True
    for t in range(10):
        r = recs[t]
        pressured = i_series[t] > 0.0 and de_series[t] >= W_PFC_RECOVERY_THRESHOLD
        verdict = "受压" if pressured else "安全"
        if not pressured:
            phase1_ok = False
        print("%4d | %7.4f | %9.4f | %.4f → %.4f | %s | %s"
              % (t + 1, de_series[t], i_series[t], w_prev[t], w_series[t],
                 verdict, act_series[t]))
    print()

    # ---------- 段 2：连夸 20 轮 ----------
    print("=== 段 2：连夸 20 轮（安全期间每轮消退）===")
    print("轮次 | D_error | I_inhibit | w_PFC(入→出)   | 判定   | action")
    phase2_ok = True
    for t in range(10, 30):
        r = recs[t]
        safe = de_series[t] < W_PFC_RECOVERY_THRESHOLD or i_series[t] == 0.0
        verdict = "安全" if safe else "受压"
        if not safe:
            phase2_ok = False
        print("%4d | %7.4f | %9.4f | %.4f → %.4f | %s | %s"
              % (t + 1, de_series[t], i_series[t], w_prev[t], w_series[t],
                 verdict, act_series[t]))
    print()

    # ---------- 汇总 ----------
    peak = max(w_series)
    peak_round = w_series.index(peak) + 1
    end1 = w_series[9]
    end2 = w_series[-1]
    print("=== 汇总 ===")
    print("疲劳峰值：%.4f（第 %d 轮）" % (peak, peak_round))
    print("段 1 终点（第 10 轮写回）：%.4f" % end1)
    print("段 2 终点（第 30 轮写回）：%.4f，恢复总量 %.4f（= 20 x %.2f = %.2f）"
          % (end2, end1 - end2, W_PFC_RECOVERY_RATE, 20 * W_PFC_RECOVERY_RATE))
    print()

    # ---------- 断言 ----------
    print("=== 断言 ===")
    # A. 段 1 受压不降
    no_drop_p1 = all(w_series[t] >= w_prev[t] - 1e-12 for t in range(10))
    check("A. 段 1（挨骂 10 轮）w_PFC 每轮不减", no_drop_p1 and phase1_ok,
          "单调性=%s，全部受压=%s" % (no_drop_p1, phase1_ok))
    # B. 段 2 每轮精确恢复
    linear = all(abs(w_series[t] - (w_prev[t] - W_PFC_RECOVERY_RATE)) < 1e-9
                 for t in range(10, 30))
    check("B. 段 2（被夸 20 轮）每轮精确 -%.2f" % W_PFC_RECOVERY_RATE, linear and phase2_ok,
          "线性=%s，全部安全=%s" % (linear, phase2_ok))
    # C. 恢复总量
    total_recovered = end1 - end2
    check("C. 段 2 恢复总量 == 20 x %.2f = %.2f" % (W_PFC_RECOVERY_RATE, 20 * W_PFC_RECOVERY_RATE),
          abs(total_recovered - 20 * W_PFC_RECOVERY_RATE) < 1e-9,
          "实际 %.4f" % total_recovered)
    # D. 公式恒等（按 log 的 D_error/I 重算每轮 w_PFC_new）
    formula_ok = True
    for t in range(30):
        de, i_v, w_in = de_series[t], i_series[t], w_prev[t]
        if de < W_PFC_RECOVERY_THRESHOLD or i_v == 0.0:
            expect = max(0.0, w_in - W_PFC_RECOVERY_RATE)
        else:
            expect = float(np.clip(RHO * w_in + DELTA * i_v, 0.0, W_MAX))
        if abs(expect - w_series[t]) > 1e-9:
            formula_ok = False
            print("  第 %d 轮不一致：重算 %.6f vs log %.6f" % (t + 1, expect, w_series[t]))
    check("D. 30 轮 w_PFC_new 与代谢公式重算完全一致", formula_ok)
    # E. 代谢后行为恢复：段 2 的 action 应以 reappraise/task_first 为主
    #    （疲劳消退后抑制阈值变化的行为签名，非硬断言阈值，只报占比）
    late_actions = act_series[10:]
    reapp = sum(1 for a in late_actions if a == "reappraise")
    tf = sum(1 for a in late_actions if a == "task_first")
    sp = sum(1 for a in late_actions if a == "suppress")
    print("[INFO] E. 段 2 action 分布：reappraise %d / task_first %d / suppress %d（20 轮）"
          % (reapp, tf, sp))
    check("E. 段 2 无 suppress（疲劳消退后不再防御）", sp == 0,
          "suppress 占比 %d/20" % sp)

    print()
    print("=== 结论: %s（失败 %d 项）===" % ("全部通过" if check.failed == 0 else "存在失败", check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
