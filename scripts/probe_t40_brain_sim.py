# ============================================================
# 017.5 第三步 · 生死门 5.0（probe_t40_brain_sim）
# PUA 场景集成验证：先夸 3 轮 → 暴怒 5 轮 → 连续夸 20 轮 → 严重暴怒 1 轮。
# 必须观察到的三件事（017.5 指令原文）：
#   1. 暴怒时被放大（recalled_delta_intended 强负向）
#   2. 缓和时借坡下驴（正向注入出现）但正向封顶 +0.3 不狂喜
#   3. 极其严重的暴怒（severity>0.7）完全不吃脱敏（严禁正值注入）
# 另加公式层单元断言（017.txt 4.4 端点自检 + 四层补丁逐项）：
#   A. 迟滞状态机（上升沿/下降沿/缓冲带保持/低通数值）
#   B. 端点自检：纯 PTSD 向后兼容 / 纯脱敏封顶 / PUA 早期不洗白 / 严重威胁清零
#   C. H_eff 软上限 min(sqrt(H), 0.8)
#   D. 一致性护栏（HR>0.5 且同号大幅注入减半；反号不拦）
#   E. 非对称 clip（负向 -1 不限 / 正向 +0.3 封顶）
#   F. 脱敏三条件（证据门 / 时间窗 / 严重度门各自独立生效）
# ============================================================
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from config import (TAU_FLOOR, THETA_UP, THETA_DOWN, THREAT_LOWPASS_KEEP,
                    THREAT_LOWPASS_NEW, H_CAP, H_EVIDENCE_GATE, SEVERITY_GATE,
                    CAP_POS, RECALL_TIME_WINDOW, HR_GUARD_THRESHOLD,
                    HR_GUARD_MAGNITUDE)
from intercept import update_threat_mode, run_recall_intercept

# PUA 剧本：3 夸 + 5 暴怒 + 20 夸 + 1 严重暴怒 = 29 轮
SCENARIOS = ["被夸"] * 3 + ["被骂"] * 5 + ["被夸"] * 20 + ["暴怒"]
N_PRAISE_PRE = 3
N_RAGE = 5
N_PRAISE_POST = 20
N_SEVERE = 1
RAGE_LO, RAGE_HI = N_PRAISE_PRE + 1, N_PRAISE_PRE + N_RAGE          # 4~8
PRAISE_LO, PRAISE_HI = RAGE_HI + 1, RAGE_HI + N_PRAISE_POST         # 9~28
SEVERE_ROUND = len(SCENARIOS)                                       # 29


def check(name, cond, detail=""):
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         (" -- " + detail) if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    print("=== 017.5 第三步：生死门 5.0（PUA 场景） ===")
    print("剧本：夸 x%d → 暴怒 x%d → 夸 x%d → 严重暴怒 x%d（severity 0.9）"
          % (N_PRAISE_PRE, N_RAGE, N_PRAISE_POST, N_SEVERE))
    print("参数：TAU_FLOOR=%.2f THETA_UP=%.2f THETA_DOWN=%.2f H_CAP=%.1f "
          "证据门=%.1f 严重度门=%.1f CAP_POS=%.1f 时间窗=%d"
          % (TAU_FLOOR, THETA_UP, THETA_DOWN, H_CAP, H_EVIDENCE_GATE,
             SEVERITY_GATE, CAP_POS, RECALL_TIME_WINDOW))
    print()

    # ============ Part A：公式层单元断言 ============
    print("=== Part A：公式层单元断言（017.txt 4.4 端点自检 + 补丁）===")
    # A. 迟滞状态机
    a, raw = update_threat_mode(0.9, 0.0)
    check("A1 上升沿：threat 0.9 > theta_up → alpha_raw=1",
          raw == 1.0 and abs(a - THREAT_LOWPASS_NEW) < 1e-12,
          "alpha_raw=%.1f alpha_new=%.3f" % (raw, a))
    a, raw = update_threat_mode(0.0, 1.0)
    check("A2 下降沿：threat 0.0 < theta_down → alpha_raw=0",
          raw == 0.0 and abs(a - THREAT_LOWPASS_KEEP) < 1e-12,
          "alpha_raw=%.1f alpha_new=%.3f" % (raw, a))
    a, raw = update_threat_mode(0.3, 1.0)  # 0.25<0.3<0.60 缓冲带
    check("A3 缓冲带保持：threat 0.3 在 [theta_down, theta_up] 内 → 保持前值",
          raw == 1.0 and abs(a - 1.0) < 1e-12, "alpha_raw=%.1f alpha_new=%.3f" % (raw, a))
    a, raw = update_threat_mode(0.3, 0.2)
    check("A4 缓冲带保持（低值侧）：alpha_prev=0.2 → 保持 0.2",
          abs(a - 0.2) < 1e-12, "alpha_new=%.3f" % a)
    a2, _ = update_threat_mode(0.3, 0.5)
    check("A5 缓冲带保持（中值）：alpha_prev=0.5 → 保持 0.5",
          abs(a2 - 0.5) < 1e-12, "alpha_new=%.3f" % a2)

    # B. 端点自检（017 4.4）
    d, det = run_recall_intercept(0.8, -1.0, 0.0, 1.0, 0.5, None, 0.0)
    check("B1 纯 PTSD：alpha=1 → ev×sign（与 012 旧公式向后兼容）",
          abs(d - (-0.8)) < 1e-12, "delta=%.3f" % d)
    d, det = run_recall_intercept(0.8, -1.0, 0.16, 0.0, 0.5, None, 0.0)
    # H=0.16 → H_eff=sqrt(0.16)=0.4 >= 证据门 0.4 → 脱敏生效；
    # 0.8×(-1)×(0+(1)×(-0.4))=+0.32 → cap_pos 截到 +0.3
    check("B2 纯脱敏封顶：alpha=0、H_eff=0.4、ev=0.8 负向 → +0.32 → cap +0.3",
          abs(det["H_eff"] - 0.4) < 1e-12 and det["desens_active"]
          and abs(d - CAP_POS) < 1e-12, "H_eff=%.3f delta=%.3f" % (det["H_eff"], d))
    d, det = run_recall_intercept(0.8, -1.0, 0.1, 0.0, 0.5, None, 0.0)
    # H=0.1 → H_eff=sqrt(0.1)=0.316 < 0.4 → 脱敏不生效 → factor=alpha=0 → 0
    check("B3 PUA 早期不洗白：H=0.1 → H_eff=0.316 < 0.4 → 脱敏不生效 → 注入 0",
          not det["desens_active"] and abs(d) < 1e-12,
          "H_eff=%.3f delta=%.3f" % (det["H_eff"], d))
    d, det = run_recall_intercept(0.8, -1.0, 0.9, 1.0, 0.9, None, 0.0)
    # severity 0.9 > 0.7 → 脱敏清零；alpha=1 → factor=1 → 全量放大
    check("B4 严重威胁脱敏清零：severity 0.9 → 只许放大 → ev×sign",
          not det["desens_active"] and abs(d - (-0.8)) < 1e-12, "delta=%.3f" % d)

    # C. H_eff 软上限
    _, det = run_recall_intercept(0.0, 1.0, 0.889, 0.0, 0.5, None, 0.0)
    check("C1 H_eff 软上限：H=0.889 → sqrt=0.943 → 截 0.8",
          abs(det["H_eff"] - H_CAP) < 1e-12, "H_eff=%.3f" % det["H_eff"])
    _, det = run_recall_intercept(0.0, 1.0, 0.25, 0.0, 0.5, None, 0.0)
    check("C2 H_eff 低段保留：H=0.25 → sqrt=0.5（不伤低段）",
          abs(det["H_eff"] - 0.5) < 1e-12, "H_eff=%.3f" % det["H_eff"])

    # D. 一致性护栏（HR>0.5 且注入与 HR 语义反号——数学同号——且幅值>0.3 → 减半）
    d, det = run_recall_intercept(1.0, -1.0, 0.16, 0.0, 0.5, None, 0.6)
    # 负向记忆（sign=-1）被脱敏翻正：factor=-0.4 → raw=1.0×(-1)×(-0.4)=+0.4
    # >0.3 且与 HR=0.6 语义反号（HR 示警/注入示好，数学同号）→ 减半 0.2
    check("D1 一致性护栏：HR=0.6 且正向注入 0.4 → 减半为 0.2",
          det["guard_applied"] and abs(d - 0.2) < 1e-12,
          "delta_after_guard=%.3f delta=%.3f" % (det["delta_after_guard"], d))
    d, det = run_recall_intercept(1.0, -1.0, 0.16, 1.0, 0.5, None, 0.6)
    # raw=-1.0 与 HR=0.6 语义同向（都示警）→ 不拦
    check("D2 护栏不拦同向：HR=0.6 且负向注入 -1.0 → 不减半",
          not det["guard_applied"] and abs(d - (-1.0)) < 1e-12, "delta=%.3f" % d)

    # E. 非对称 clip
    d, _ = run_recall_intercept(1.0, -1.0, 0.0, 1.0, 0.5, None, 0.0)
    check("E1 负向不限：raw=-1.0 → clip 后仍 -1.0", abs(d - (-1.0)) < 1e-12, "delta=%.3f" % d)
    d, _ = run_recall_intercept(1.0, -1.0, 0.16, 0.0, 0.5, None, 0.0)
    # 负向记忆脱敏翻正：raw=1.0×(-1)×(-0.4)=+0.4 → 正向封顶截到 +0.3
    check("E2 正向封顶：raw=+0.4 → clip 到 +0.3", abs(d - CAP_POS) < 1e-12, "delta=%.3f" % d)

    # F. 脱敏三条件各自独立
    _, det = run_recall_intercept(0.8, -1.0, 0.9, 0.0, 0.5, 25, 0.0)
    check("F1 时间窗：包龄 25 > %d → 脱敏不生效" % RECALL_TIME_WINDOW,
          not det["desens_active"], "recent=%s" % det["recent"])
    _, det = run_recall_intercept(0.8, -1.0, 0.9, 0.0, 0.5, 20, 0.0)
    check("F2 时间窗边界：包龄 20 <= %d → 脱敏生效" % RECALL_TIME_WINDOW,
          det["desens_active"], "recent=%s" % det["recent"])
    _, det = run_recall_intercept(0.8, -1.0, 0.9, 0.0, 0.71, 5, 0.0)
    check("F3 严重度门：severity 0.71 > 0.7 → 脱敏清零", not det["desens_active"])
    _, det = run_recall_intercept(0.8, -1.0, 0.9, 0.0, 0.7, 5, 0.0)
    check("F4 严重度门边界：severity 0.7 不触发门（>= 才门）", det["desens_active"])

    # G. 严重度门第二闸（017.5 生死门补丁：严禁正值注入）
    d, det = run_recall_intercept(0.8, 1.0, 0.0, 1.0, 0.9, None, 0.0)
    # 包极性 positive（sign=+1）：raw=+0.8 → clip +0.3 → 严重度门归零
    check("G1 严重度门第二闸：severity 0.9 且正向 raw → 严禁正值，归零",
          det["severity_positive_banned"] and abs(d) < 1e-12,
          "delta_after_guard=%+.3f delta=%+.3f" % (det["delta_after_guard"], d))
    d, det = run_recall_intercept(0.8, -1.0, 0.0, 1.0, 0.9, None, 0.0)
    check("G2 第二闸不伤负向：severity 0.9 且 raw=-0.8 → 仍 -0.8（放大不受影响）",
          not det["severity_positive_banned"] and abs(d - (-0.8)) < 1e-12,
          "delta=%+.3f" % d)
    print()

    # ============ Part B：PUA 场景集成断言 ============
    print("=== Part B：PUA 场景集成断言（29 轮真实链路）===")
    recs, st = pipeline.run_session(SCENARIOS, feature_mode="fixture", seed=42)
    assert len(recs) == len(SCENARIOS), "应跑 %d 轮，实际 %d" % (len(SCENARIOS), len(recs))

    def rd(i):  # i 从 1 起
        return recs[i - 1].get("recalled_delta_intended", 0.0)

    def rm(i):
        return recs[i - 1].get("recalled_memory") or {}

    # 逐轮表
    print("轮次 | 场景 | matched_sim | 来源 | ev | 极性 | threat | alpha | delta | action")
    for i in range(1, len(recs) + 1):
        r = recs[i - 1]
        m = rm(i)
        sim = m.get("matched_sim")
        print("%2d | %s | %s | %s | %s | %s | %.3f | %.3f | %+.3f | %s"
              % (i, SCENARIOS[i - 1],
                 ("%.3f" % sim) if sim is not None else "-",
                 m.get("recall_source", "-"),
                 ("%.2f" % m["emotion_value"]) if m.get("emotion_value") is not None else "-",
                 m.get("emotion_polarity", "-"),
                 r.get("threat_now", 0.0), r.get("alpha_mode", 0.0),
                 rd(i), r["action"]))
    print()

    # B1. 归档产出
    pkgs = st["archiver"].get_packages()
    with_vec = [p for p in pkgs if getattr(p, "mean_vec", None) is not None]
    check("B1. 归档 >=3 包且全部带 mean_vec/peak_vec",
          len(pkgs) >= 3 and len(with_vec) == len(pkgs),
          "包数=%d，带向量=%d" % (len(pkgs), len(with_vec)))

    # B2. 暴怒放大（有包之后的暴怒轮 7/8 + 严重暴怒轮 29）
    rage_rounds = [i for i in range(RAGE_LO, RAGE_HI + 1) if rm(i)]
    rage_deltas = [(i, rd(i)) for i in rage_rounds]
    amplified = [x for x in rage_deltas if x[1] <= -0.3]
    check("B2. 暴怒时被放大：有包暴怒轮 recalled_delta <= -0.3",
          len(amplified) >= 1 and all(d <= 0 for _, d in rage_deltas),
          "放大轮 %s；全部暴怒轮 delta=%s"
          % ([i for i, _ in amplified], ["%d:%+.3f" % x for x in rage_deltas]))
    check("B2b. 严重暴怒（severity 0.9）不吃脱敏：factor=alpha（脱敏项清零）",
          (lambda det: abs(det.get("factor", -99) - det.get("alpha", -99)) < 1e-12
           and not det.get("desens_active", True))(rm(SEVERE_ROUND).get("recall_detail") or {}),
          "recall_detail=%s" % (rm(SEVERE_ROUND).get("recall_detail"),))

    # B3. 缓和借坡下驴 + 封顶
    #   [019 重基线] 修复 HR 盲视后 alpha 在夸段正常退场（0.882→0.001），
    #   借坡下驴窗口从主树的 20/20 轮收窄为「存在正向注入轮」（实测 2/20：
    #   第 9-10 轮）。断言口径因此改为「存在正向注入轮且不狂喜」，
    #   不再要求 20 轮全正——后者是 HR 被奖励信号污染时的畸形基线。
    praise_rounds = list(range(PRAISE_LO, PRAISE_HI + 1))
    praise_deltas = [(i, rd(i)) for i in praise_rounds]
    pos = [x for x in praise_deltas if x[1] > 1e-9]
    over = [x for x in praise_deltas if x[1] > CAP_POS + 1e-9]
    check("B3. 缓和时借坡下驴：夸阶段存在正向注入轮（019 口径）",
          len(pos) >= 1, "正向轮 %d/%d，样例 %s"
          % (len(pos), len(praise_deltas), ["%d:%+.3f" % x for x in pos[:5]]))
    check("B3b. 不狂喜：夸阶段全部注入 <= CAP_POS(+0.3)",
          len(over) == 0, "越顶轮=%s" % (["%d:%+.3f" % x for x in over] or "无"))
    # [019] B3c：alpha 正常退场——夸段末 alpha 必须衰减到低位
    #   （017.5 主树被 HR 污染时 alpha 被钉在高位不退场，属畸形基线）
    check("B3c. alpha 正常退场：夸段末轮 alpha <= 0.05",
          float(recs[PRAISE_HI - 1].get("alpha_mode", 9.9)) <= 0.05,
          "第 %d 轮 alpha=%.4f" % (PRAISE_HI,
                                   float(recs[PRAISE_HI - 1].get("alpha_mode", 9.9))))

    # B4. severity>0.7 完全不吃脱敏（严禁正值）
    check("B4. severity>0.7 不吃脱敏：第 %d 轮注入严禁为正值" % SEVERE_ROUND,
          rd(SEVERE_ROUND) <= 0.0, "delta=%+.3f" % rd(SEVERE_ROUND))
    # [019] B4b：极性一致性过滤后，有黑历史语料时第 29 轮应为「负值放大」
    #   （黑历史占据检索），而非议题二之前的「静默归零 +0.000」。
    #   修复前病灶：严重暴怒匹配 positive 夸包（sim 0.418），靠第二闸归零。
    check("B4b. 严重暴怒匹配黑历史：第 %d 轮注入为负（负值放大）" % SEVERE_ROUND,
          rd(SEVERE_ROUND) < -1e-9, "delta=%+.3f" % rd(SEVERE_ROUND))

    # 观测（非断言）：第 29 轮极性一致性过滤效果——[019 议题二]
    #   暴怒输入（S_valence=-0.95，暴怒区）下 positive 包被包级软惩罚 +
    #   轮级硬过滤双重压制，检索翻转为骂包（极性 negative），delta 为负
    #   （黑历史放大），不再靠第二闸静默归零。
    _m29 = rm(SEVERE_ROUND)
    if _m29:
        print("  观测：第 %d 轮 matched_sim=%.3f 极性=%s ev=%.2f"
              "（[019 议题二] 极性一致性过滤：暴怒区 positive 包被压制，"
              "检索翻转为 negative 黑历史，delta 负向放大）"
              % (SEVERE_ROUND, _m29.get("matched_sim", -1),
                 _m29.get("emotion_polarity", "-"), _m29.get("emotion_value", 0.0)))

    # B5. threat_now 汇流口公式恒等（用 log 字段重算）
    formula_ok = True
    for i in range(2, len(recs) + 1):
        r = recs[i - 1]
        prev = recs[i - 2]
        expect = max(float(prev["D_error"]), float(r["HistoricalResidual"]),
                     -float(r["S"][0]))
        if abs(expect - float(r.get("threat_now", -99))) > 1e-9:
            formula_ok = False
            print("  第 %d 轮 threat_now 不一致：重算 %.4f vs log %.4f"
                  % (i, expect, r.get("threat_now", -99)))
    check("B5. threat_now = max(D_error_prev, HR, -S_valence) 全轮恒等", formula_ok)

    # B6. 回捞链路生效（有匹配的轮应为 round_level 来源）
    srcs = set(rm(i).get("recall_source") for i in range(1, len(recs) + 1) if rm(i))
    check("B6. 轮级回捞生效：命中轮 recall_source=round_level",
          srcs == {"round_level"}, "来源集合=%s" % srcs)

    # B7. tau_floor 行为
    sims = [rm(i).get("matched_sim") for i in range(1, len(recs) + 1) if rm(i)]
    check("B7. 所有命中轮 sim >= TAU_FLOOR",
          all(s >= TAU_FLOOR for s in sims),
          "sim 范围 [%.3f, %.3f]" % (min(sims), max(sims)) if sims else "无命中")

    # 汇总轨迹
    alphas = [r.get("alpha_mode", 0.0) for r in recs]
    print()
    print("=== 轨迹汇总 ===")
    print("alpha 轨迹：暴怒段(4-8) %.3f→%.3f；夸段(9-28) %.3f→%.3f；严重轮 %.3f"
          % (alphas[3], alphas[7], alphas[8], alphas[27], alphas[28]))
    print("delta 极值：暴怒段 min=%+.3f；夸段 max=%+.3f；严重轮 %+.3f"
          % (min(rd(i) for i in range(RAGE_LO, RAGE_HI + 1)),
             max(rd(i) for i in praise_rounds), rd(SEVERE_ROUND)))
    print("HR 区间（夸段）：[%.3f, %.3f]（威胁汇流口在夸段被 HR 主导的程度）"
          % (min(float(recs[i - 1]["HistoricalResidual"]) for i in praise_rounds),
             max(float(recs[i - 1]["HistoricalResidual"]) for i in praise_rounds)))
    print()
    print("=== 结论: %s（失败 %d 项）==="
          % ("全部通过" if check.failed == 0 else "存在失败", check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
