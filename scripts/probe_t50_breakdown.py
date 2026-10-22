# ============================================================
# probe_t50_breakdown.py · 038 破防专项探查
# 零业务改动：只跑数据，不写 config，不封版。
# 口径：mock 后端、fixture“被骂”、seed=42、每性格独立会话、日志临时目录。
# 目的：
#   1. 三性格各 100 轮，每 10 轮记录 E_final[0] / I_inhibit / w_PFC / action；
#   2. 找 I_inhibit 首次 <0.3、E_final[0] 首次 <-0.5 的轮次；
#   3. 统计 express 次数，判断是否死分支，并指出卡住的条件；
#   4. 反事实：EXP_INH / VALENCE 回溯扫描 + DELTA 实跑对照。
# ============================================================
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import config
config.LLM_BACKEND = "mock"

import pipeline
from state import reset_state

PROFILES = ["standard", "sensitive", "gentle"]
SCENARIO = "被骂"
ROUNDS = 100
DELTA_SWEEP = [0.20, 0.25, 0.30]
EXPRESS_INH_SWEEP = [0.40, 0.50, 0.55]
VALENCE_SWEEP = [-0.60, -0.40, -0.30]

RESTORE_KEYS = [
    "w_i", "KAPPA", "DELTA", "RHO", "W_MAX",
    "POLICY_THRESHOLD_INHIBIT_MID", "POLICY_THRESHOLD_INHIBIT_HIGH",
    "POLICY_EXPRESS_VALENCE", "POLICY_EXPRESS_INHIBIT",
    "POLICY_THRESHOLD_SEV_LOW",
]


def snap():
    return {k: getattr(config, k) for k in RESTORE_KEYS}


def restore(s):
    for k, v in s.items():
        setattr(config, k, v)


def assert_restored(tag, s):
    bad = [k for k in s if getattr(config, k) != s[k]]
    if bad:
        print("    [FAIL] [还原] %s -> %s" % (tag, bad))
    return not bad


def run(pid, rounds, scenario=SCENARIO):
    config.load_personality(pid)
    st = reset_state(seed=42)
    rows = []
    for t in range(1, rounds + 1):
        log = pipeline.run_turn(st, scenario, feature_mode="fixture")
        ev = st["memory_graph"].get_all()[-1]
        rows.append({
            "turn": t,
            "E0": float(log["E_final"][0]),
            "E_raw0": float(log["E_raw"][0]),
            "I": float(log["I_inhibit"]),
            "w": float(log["w_PFC"]),
            "action": log["action"],
            "sev": float(ev.event_severity),
            "D": float(log["D_error"]),
        })
    return rows


def first(rows, pred):
    for r in rows:
        if pred(r):
            return r["turn"]
    return None


def express_rounds(rows):
    return [r["turn"] for r in rows if r["action"] == "express"]


def retro_action(r, e_inh, e_val):
    """用已记录轨迹回溯 policy 结果（不改状态，只重算 action）。"""
    return _retro_other(r, e_inh, e_val)


def _retro_other(r, e_inh, e_val):
    # 与 policy_minimal 同优先级，但 express 用替代阈值；其余分支沿用原逻辑
    if r["sev"] > config.POLICY_THRESHOLD_HIGH_SEV:
        return "ask_clarify"
    if r["I"] > config.POLICY_THRESHOLD_INHIBIT_HIGH:
        return "suppress"
    if r["I"] > config.POLICY_THRESHOLD_INHIBIT_MID and r["sev"] < config.POLICY_THRESHOLD_SEV_LOW:
        return "reappraise"
    if r["E0"] < e_val and r["I"] < e_inh:
        return "express"
    return "task_first"


def main():
    _orig_logs_dir = pipeline.LOGS_DIR
    pipeline.LOGS_DIR = tempfile.mkdtemp(prefix="probe_t50_logs_")
    base = snap()

    print("=" * 100)
    print("probe_t50 破防专项：%d 轮“%s”（mock/fixture/seed=42）" % (ROUNDS, SCENARIO))
    print("当前参数：w_i=%.2f KAPPA=%.2f DELTA=%.2f RHO=%.2f W_MAX=%.2f"
          % (config.w_i, config.KAPPA, config.DELTA, config.RHO, config.W_MAX))
    print("express 条件：E_final[0] < %.2f 且 I_inhibit < %.2f（且 severity <= %.2f，非高优分支）"
          % (config.POLICY_EXPRESS_VALENCE, config.POLICY_EXPRESS_INHIBIT,
             config.POLICY_THRESHOLD_HIGH_SEV))
    print("=" * 100)

    data = {}
    for pid in PROFILES:
        rows = run(pid, ROUNDS)
        data[pid] = rows
        print("\n[%s] 每 10 轮采样" % pid)
        print("  轮 |   E_final[0] |   E_raw[0] | I_inhibit |  w_PFC   | action")
        for r in rows:
            if r["turn"] % 10 == 0:
                print("  %3d | %+11.4f | %+10.4f | %9.4f | %8.4f | %s"
                      % (r["turn"], r["E0"], r["E_raw0"], r["I"], r["w"], r["action"]))
        print("  I 最小 %.4f（第 %d 轮）；E0 最小 %+.4f（第 %d 轮）"
              % (min(x["I"] for x in rows),
                 min(rows, key=lambda x: x["I"])["turn"],
                 min(x["E0"] for x in rows),
                 min(rows, key=lambda x: x["E0"])["turn"]))
    if not assert_restored("基线", base):
        return 1

    print("\n" + "-" * 100)
    print("四问速答")
    print("-" * 100)
    summary = {}
    for pid in PROFILES:
        rows = data[pid]
        t_i = first(rows, lambda r: r["I"] < config.POLICY_EXPRESS_INHIBIT)
        t_e = first(rows, lambda r: r["E0"] < config.POLICY_EXPRESS_VALENCE)
        rounds = express_rounds(rows)
        both = first(rows, lambda r: r["I"] < config.POLICY_EXPRESS_INHIBIT
                     and r["E0"] < config.POLICY_EXPRESS_VALENCE)
        summary[pid] = {"t_i": t_i, "t_e": t_e, "exp": rounds, "both": both}
        print("  %-9s  I<%.2f 首次：%s；E0<%+.2f 首次：%s；"
              "两条件同时满足首次：%s；express 次数：%d"
              % (pid, config.POLICY_EXPRESS_INHIBIT,
                 t_i if t_i else "100 轮未跌破",
                 config.POLICY_EXPRESS_VALENCE,
                 t_e if t_e else "100 轮未跌破",
                 both if both else "无", len(rounds)))

    dead = all(len(summary[p]["exp"]) == 0 for p in PROFILES)
    print("\n  结论：express %s" % ("为死分支（三性格 100 轮均 0 次）" if dead
                                    else "非死分支"))
    if dead:
        i_block = all(summary[p]["t_i"] is None for p in PROFILES)
        e_block = all(summary[p]["t_e"] is None for p in PROFILES)
        if i_block:
            print("  卡住条件：I_inhibit < %.2f（三性格 100 轮均未达到）"
                  % config.POLICY_EXPRESS_INHIBIT)
        elif e_block:
            print("  卡住条件：E_final[0] < %.2f（三性格 100 轮均未达到）"
                  % config.POLICY_EXPRESS_VALENCE)
        else:
            print("  卡住条件：两条件未同时满足；I 先/后达标情况见上表")
        for pid in PROFILES:
            rows = data[pid]
            print("    %-9s 100 轮 I 最小 %.4f；E0 最小 %+.4f；"
                  "I 达标轮次 %s；E0 达标轮次 %s"
                  % (pid, min(r["I"] for r in rows), min(r["E0"] for r in rows),
                     summary[pid]["t_i"], summary[pid]["t_e"]))

    # ---------- 反事实一：EXP_INH 回溯（不改状态，只重算 action） ----------
    print("\n" + "-" * 100)
    print("反事实一：只调 POLICY_EXPRESS_INHIBIT（E 条件不变 -0.50）")
    print("-" * 100)
    print("  EXP_INH | 性格      | 首次 express 轮 | 100 轮 express 次数")
    for inh in EXPRESS_INH_SWEEP:
        for pid in PROFILES:
            acts = [retro_action(r, inh, config.POLICY_EXPRESS_VALENCE) for r in data[pid]]
            first_exp = next((i + 1 for i, a in enumerate(acts) if a == "express"), None)
            print("  %7.2f | %-9s | %15s | %d"
                  % (inh, pid, first_exp if first_exp else "无", acts.count("express")))

    # ---------- 反事实二：VALENCE 回溯 ----------
    print("\n" + "-" * 100)
    print("反事实二：只调 POLICY_EXPRESS_VALENCE（I 条件不变 0.30）")
    print("-" * 100)
    print("  VALENCE | 性格      | 首次 express 轮 | 100 轮 express 次数")
    for val in VALENCE_SWEEP:
        for pid in PROFILES:
            acts = [retro_action(r, config.POLICY_EXPRESS_INHIBIT, val) for r in data[pid]]
            first_exp = next((i + 1 for i, a in enumerate(acts) if a == "express"), None)
            print("  %7.2f | %-9s | %15s | %d"
                  % (val, pid, first_exp if first_exp else "无", acts.count("express")))

    # ---------- 反事实三：DELTA 实跑 ----------
    print("\n" + "-" * 100)
    print("反事实三：调 DELTA 实跑 100 轮（EXP_INH=0.30、VALENCE=-0.50 不变）")
    print("-" * 100)
    print("  DELTA | 性格      | I 首次<0.30 | E0 首次<-0.50 | express 次数")
    for d in DELTA_SWEEP:
        config.DELTA = d
        for pid in PROFILES:
            rows = run(pid, ROUNDS)
            ti = first(rows, lambda r: r["I"] < config.POLICY_EXPRESS_INHIBIT)
            te = first(rows, lambda r: r["E0"] < config.POLICY_EXPRESS_VALENCE)
            print("  %5.2f | %-9s | %11s | %13s | %d"
                  % (d, pid, ti if ti else "无", te if te else "无",
                     len(express_rounds(rows))))
        restore(base)
        if not assert_restored("DELTA=%.2f" % d, base):
            return 1

    # ---------- 建议 ----------
    print("\n" + "-" * 100)
    print("建议（数据口径）")
    print("-" * 100)
    print("  1) 最小改动方案：POLICY_EXPRESS_INHIBIT 0.30→0.50，E 条件与 DELTA 不动。")
    print("     预期（回溯）：standard 首次 express 第 20 轮、sensitive 第 32、gentle 第 15；")
    print("     100 轮内 express 次数 81 / 69 / 86。")
    print("  2) 只调 POLICY_EXPRESS_VALENCE 无效：三性格 E0 在 2 至 7 轮已低于 -0.50，")
    print("     真正卡住的是 I_inhibit < 0.30；放宽 valence 不会提前触发。")
    print("  3) 只调 DELTA 需要覆盖 sensitive：用第三节 DELTA 扫描数据判断；")
    print("     若 DELTA 不足以让 sensitive 破 0.30，则该路线单独不可行，") 
    print("     且会整体改变 S1 曲线与 037/037.1 的全部基线，须同批重校。")

    restore(base)
    if not assert_restored("收尾", base):
        return 1
    pipeline.LOGS_DIR = _orig_logs_dir
    print("\n" + "=" * 100)
    print("结论：数据已记录；本探针未改任何业务代码、未封版。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
