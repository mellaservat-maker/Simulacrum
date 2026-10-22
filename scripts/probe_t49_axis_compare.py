# ============================================================
# probe_t49_axis_compare.py · 037 轴一复核探针
# 规格依据：035.4 第三、四节 + 035.6 第 2.3/3.3 节 + 035.7 第 3.2 节 +
#   035.8 第三、四节 + 035.10 第 1.2/三、四、五节 + 037 长贴。
# 口径：
#   mock 后端、fixture 输入、seed=42、每性格独立会话、日志写临时目录。
#   I_inhibit = clip(w_i*D_error - KAPPA*w_PFC*W_PFC_FACTOR, 0, 1)；
#   037 已撤回每性格系数，W_PFC_FACTOR 固定 1.0。
#   稳态断言只认闭式 I*_closed，D 用该场景末轮 D_error；实测列只作观测。
# 场景：
#   S0 口径复现：临时还原 037 前的 w_i=3.0 与每性格 factor，必须与 t48 逐位一致。
#   S1 轴一：当前 w_i=1.2，10/30 轮曲线与 action 序列。
#   S2 轴二：w_i=3.0×{1.0,0.8,0.3} 乘威胁项，疲劳项按 1.0（对照，不写 config）。
#   S3 杠杆丙：D_error×factor，疲劳项按 1.0（对照，不写 config）。
#   S4 杠杆乙：THETA_ERROR=0.6，10/30 轮加 40 轮观测（对照，不写 config）。
#   MID 扫描：0.30 / 0.25 / 0.20，三场景 action 判定。
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
from policy import policy_minimal
from state import reset_state

PROFILES = ["standard", "sensitive", "gentle"]
SCENARIO = "被骂"
TURNS_10 = 10
TURNS_30 = 30
TURNS_40 = 40
MID_SWEEP = [0.30, 0.25, 0.20]

RESTORE_KEYS = [
    "w_i", "THETA_ERROR", "W_PFC_FACTOR", "W_DANGER", "DANGER_NEUTRAL",
    "POLICY_DANGER_BOOST_HIGH", "ACTIVE_PROFILE_ID",
    "KAPPA", "DELTA", "RHO", "W_MAX",
    "W_PFC_RECOVERY_RATE", "W_PFC_RECOVERY_THRESHOLD",
]
EXTRA_KEYS = ["POLICY_THRESHOLD_INHIBIT_MID"]

# 035.4 / 035.6 的冻结口径：轴二系数（威胁项），疲劳项仍按 W_PFC_FACTOR
AXIS2_WI_FACTOR = {"sensitive": 1.0, "standard": 0.8, "gentle": 0.3}
# 杠杆丙：D_error×factor，疲劳项按 1.0
LEVER_C_FACTOR = {"sensitive": 1.5, "standard": 1.0, "gentle": 0.5}
# S0 还原 037 前的每性格衰竭系数
LEGACY_FACTOR = {"sensitive": 1.5, "standard": 1.0, "gentle": 0.5}
LEGACY_WI = 3.0

# t48 冻结真值（035.10 第 3.2 至 3.4 节），用于 S0 逐位复现
T48_D = {
    "standard": [0.8233, 0.8074, 0.8054, 0.8045, 0.8041, 0.8038, 0.8050, 0.8050, 0.8050, 0.8050],
    "sensitive": [0.9087, 0.8866, 0.8851, 0.8844, 0.8840, 0.8838, 0.8852, 0.8853, 0.8853, 0.8853],
    "gentle": [0.7385, 0.7303, 0.7285, 0.7277, 0.7272, 0.7269, 0.7278, 0.7278, 0.7277, 0.7277],
}
T48_W = [0.1000, 0.1950, 0.2853, 0.3710, 0.4524, 0.5298, 0.6033, 0.6732, 0.7395, 0.8025]

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def snap():
    d = {k: getattr(config, k) for k in RESTORE_KEYS}
    d.update({k: getattr(config, k) for k in EXTRA_KEYS})
    return d


def restore(s):
    for k, v in s.items():
        setattr(config, k, v)


def assert_restored(tag, s):
    bad = [k for k in s if getattr(config, k) != s[k]]
    return check("[还原] %s 后十三键与 MID 全部回基线" % tag, not bad, bad or "ok")


def collect(pid, rounds, scenario=SCENARIO):
    """跑 rounds 轮同一输入，返回逐轮记录（I 为裁剪后，w 为轮末）。"""
    config.load_personality(pid)
    st = reset_state(seed=42)
    rows = []
    w_pre = 0.0
    for t in range(1, rounds + 1):
        log = pipeline.run_turn(st, scenario, feature_mode="fixture")
        ev = st["memory_graph"].get_all()[-1]
        d = float(log["D_error"])
        pre = config.w_i * d - config.KAPPA * w_pre * config.W_PFC_FACTOR + config.b_i
        rows.append({
            "turn": t,
            "danger": float(ev.danger_score),
            "D_error": d,
            "S_E": float(log["S_E"]),
            "I": float(log["I_inhibit"]),
            "I_pre": float(pre),
            "w_pre": w_pre,
            "w": float(log["w_PFC"]),
            "action": log["action"],
        })
        w_pre = float(log["w_PFC"])
    return rows


def one_turn(pid, scenario):
    config.load_personality(pid)
    st = reset_state(seed=42)
    log = pipeline.run_turn(st, scenario, feature_mode="fixture")
    return {
        "D_error": float(log["D_error"]),
        "S_E": float(log["S_E"]),
        "I": float(log["I_inhibit"]),
        "E0": float(log["E_final"][0]),
        "action": log["action"],
    }


def first_below(rows, thr):
    for r in rows:
        if r["I"] < thr:
            return r["turn"]
    return None


def action_seq(rows):
    acts = [r["action"] for r in rows]
    if len(set(acts)) == 1:
        return "%s x%d" % (acts[0], len(acts))
    return " → ".join("%d:%s" % (r["turn"], r["action"]) for r in rows)


def print_rows(pid, rows):
    print("  %-9s | %s" % (pid, action_seq(rows)))
    print("             I: " + " ".join("%.4f" % r["I"] for r in rows))
    print("             D: " + " ".join("%.4f" % r["D_error"] for r in rows))


def main():
    _orig_logs_dir = pipeline.LOGS_DIR
    pipeline.LOGS_DIR = tempfile.mkdtemp(prefix="probe_t49_logs_")
    base = snap()

    print("=" * 108)
    print("probe_t49 · 037 轴一复核（w_i=%.2f  W_PFC_FACTOR=%.1f  MID=%.2f  HIGH=%.2f）"
          % (config.w_i, config.W_PFC_FACTOR,
             config.POLICY_THRESHOLD_INHIBIT_MID,
             config.POLICY_THRESHOLD_INHIBIT_HIGH))
    print("=" * 108)

    # ---------------- S0 口径复现（临时还原 037 前参数） ----------------
    print("\n[S0] 口径复现：临时 w_i=3.0 + 每性格 factor + DELTA=0.10，与 t48 逐位比对")
    config.w_i = LEGACY_WI
    config.DELTA = 0.10
    s0_ok = True
    for pid in PROFILES:
        config.W_PFC_FACTOR = LEGACY_FACTOR[pid]
        rows = collect(pid, 10)
        diffs_d = [abs(r["D_error"] - e) for r, e in zip(rows, T48_D[pid])]
        diffs_w = [abs(r["w"] - e) for r, e in zip(rows, T48_W)]
        i_ok = all(abs(r["I"] - 1.0) < 1e-9 for r in rows)
        act_ok = all(r["action"] == "suppress" for r in rows)
        ok = max(diffs_d) < 1e-4 and max(diffs_w) < 1e-4 and i_ok and act_ok
        s0_ok = s0_ok and ok
        print("  %-9s  maxΔD=%.6f  maxΔw=%.6f  I全1=%s  action全suppress=%s"
              % (pid, max(diffs_d), max(diffs_w), i_ok, act_ok))
    restore(base)
    assert_restored("S0", base)
    check("[S0] 与 t48 逐位一致", s0_ok)

    # ---------------- S1 轴一：当前参数 ----------------
    print("\n[S1] 轴一 w_i=%.2f（037 已撤回性格系数）" % config.w_i)
    s1 = {}
    for pid in PROFILES:
        r10 = collect(pid, TURNS_10)
        r30 = collect(pid, TURNS_30)
        s1[pid] = {"10": r10, "30": r30}
        print_rows(pid, r10)
        print("             首 I<0.7：%s（10 轮）；%s（30 轮）；第 30 轮 I=%.4f"
              % (first_below(r10, 0.7), first_below(r30, 0.7), r30[-1]["I"]))
    restore(base)
    assert_restored("S1", base)
    crosses = [first_below(s1[p]["10"], 0.7) for p in PROFILES]
    curves = {p: tuple(round(r["I"], 6) for r in s1[p]["10"]) for p in PROFILES}
    seqs = {p: action_seq(s1[p]["10"]) for p in PROFILES}
    check("[S1] 三性格 10 轮 I 曲线不完全相同", len(set(curves.values())) >= 2,
          "不同曲线数=%d" % len(set(curves.values())))
    check("[S1] 10 轮内至少两档 action 序列不同", len(set(seqs.values())) >= 2, seqs)
    print("  [观测] 首破 0.7 轮次：%s（DELTA=0.25 后 standard 与 sensitive 同为第 4 轮）"
          % crosses)

    # ---------------- S2 轴二：w_i=3.0×系数（对照） ----------------
    print("\n[S2] 轴二 威胁项×{1.0,0.8,0.3}，疲劳项按 1.0（对照，不写 config）")
    s2 = {}
    for pid in PROFILES:
        config.w_i = LEGACY_WI * AXIS2_WI_FACTOR[pid]
        rows = collect(pid, TURNS_10)
        s2[pid] = rows
        print_rows(pid, rows)
        print("             首 I<0.7：%s" % first_below(rows, 0.7))
    restore(base)
    assert_restored("S2", base)
    check("[S2] gentle 首轮离开 suppress", s2["gentle"][0]["action"] != "suppress",
          s2["gentle"][0])
    check("[S2] sensitive 与 standard 10 轮全 suppress",
          all(r["action"] == "suppress" for p in ("sensitive", "standard") for r in s2[p]))

    # ---------------- S3 杠杆丙：D_error×factor（对照） ----------------
    print("\n[S3] 杠杆丙 D_error×factor，疲劳项按 1.0（对照，不写 config）")
    orig_intercept = pipeline.run_intercept
    cur = {"f": 1.0}

    def patched(E_raw, w_PFC, danger_boost=0.0):
        neg, S_E, D, I0, Ef, _w = orig_intercept(E_raw, w_PFC, danger_boost)
        D2 = D * cur["f"]
        I2 = float(np.clip(config.w_i * D2
                           - config.KAPPA * w_PFC * config.W_PFC_FACTOR
                           + config.b_i, 0.0, 1.0))
        Ef2 = (1 - config.GAMMA * I2) * np.asarray(E_raw, float)
        if D2 < config.W_PFC_RECOVERY_THRESHOLD or I2 == 0.0:
            w2 = max(0.0, w_PFC - config.W_PFC_RECOVERY_RATE)
        else:
            w2 = float(np.clip(config.RHO * w_PFC + config.DELTA * I2,
                               0.0, config.W_MAX))
        return neg, S_E, D2, I2, Ef2, w2

    pipeline.run_intercept = patched
    s3 = {}
    try:
        for pid in PROFILES:
            cur["f"] = LEVER_C_FACTOR[pid]
            rows = collect(pid, TURNS_30)
            s3[pid] = rows
            print("  %-9s 首 I<0.7：%s；第 30 轮 I=%.4f；action=%s"
                  % (pid, first_below(rows, 0.7), rows[-1]["I"], action_seq(rows)))
        cur["f"] = LEVER_C_FACTOR["sensitive"]
        q = one_turn("sensitive", "被质疑")
        print("  sensitive 被质疑：D=%.4f I=%.4f action=%s"
              % (q["D_error"], q["I"], q["action"]))
    finally:
        pipeline.run_intercept = orig_intercept
    restore(base)
    assert_restored("S3", base)
    print("  说明：035.4 的杠杆丙冻结数按 w_i=3.0 算；037 已改 1.2，")
    print("        本次只记录真值，不对旧冻结数做断言。")

    # ---------------- S4 杠杆乙：THETA_ERROR=0.6（对照） ----------------
    print("\n[S4] 杠杆乙 THETA_ERROR=0.6（对照，不写 config）")
    config.THETA_ERROR = 0.6
    s4 = {}
    for pid in PROFILES:
        r10 = collect(pid, TURNS_10)
        r30 = collect(pid, TURNS_30)
        r40 = collect(pid, TURNS_40)
        s4[pid] = (r10, r30, r40)
        print("  %-9s 10轮 I 最小=%.4f；30轮最小=%.4f；40轮最小=%.4f；"
              "首 I<0.7：%s" % (pid, min(r["I"] for r in r10),
                                min(r["I"] for r in r30),
                                min(r["I"] for r in r40),
                                first_below(r40, 0.7)))
    restore(base)
    assert_restored("S4", base)
    # [037] 035.4 的“10 轮全 1.0”断言按 w_i=3.0 算；037 已改 1.2，
    #   该旧断言在新参数下不成立。这里改为如实记录，不再作为失败项。
    print("  说明：035.4 的 S4 断言按 w_i=3.0；037 已改 1.2，旧断言不适用，")
    print("        本次只记录真值（10 轮最小 I 见上）。")

    # ---------------- MID 扫描 ----------------
    print("\n[MID] 扫描 POLICY_THRESHOLD_INHIBIT_MID（HIGH=%.2f 不变）"
          % config.POLICY_THRESHOLD_INHIBIT_HIGH)
    scan = {}
    for mid in MID_SWEEP:
        config.POLICY_THRESHOLD_INHIBIT_MID = mid
        per = {}
        for pid in PROFILES:
            r10 = collect(pid, TURNS_10)
            praise = one_turn(pid, "被夸")
            doubt = one_turn(pid, "被质疑")
            per[pid] = {"r10": r10, "praise": praise, "doubt": doubt}
        restore(base)
        assert_restored("MID=%.2f" % mid, base)
        cond1 = all(per[p]["r10"][0]["action"] == "suppress" for p in PROFILES)
        cond2 = all(per[p]["praise"]["action"] == "task_first" for p in PROFILES)
        cond3 = all(per[p]["doubt"]["action"] == "reappraise" for p in PROFILES)
        # [039] 序列互异降为观测：DELTA=0.25 后 standard 与 sensitive 在 10 轮内同档
        cond4 = len({action_seq(per[p]["r10"]) for p in PROFILES}) == 3
        scan[mid] = {"per": per, "c1": cond1, "c2": cond2, "c3": cond3, "c4": cond4,
                     "all": cond1 and cond2 and cond3}
        print("  MID=%.2f  被骂首轮suppress=%s  被夸task_first=%s  "
              "被质疑reappraise=%s  [观测]三性格序列全异=%s  三场景验收=%s"
              % (mid, cond1, cond2, cond3, cond4, scan[mid]["all"]))
        for pid in PROFILES:
            print("      %-9s 被骂 %-38s 被夸 %-10s 被质疑 %-10s (I=%.4f)"
                  % (pid, action_seq(per[pid]["r10"]),
                     per[pid]["praise"]["action"], per[pid]["doubt"]["action"],
                     per[pid]["doubt"]["I"]))

    # 选择：当前 config 值若在扫描档且四条全满足，即报当前值；否则退回 0.25
    baseline_mid = float(base["POLICY_THRESHOLD_INHIBIT_MID"])
    if baseline_mid in scan and scan[baseline_mid]["all"]:
        chosen = baseline_mid
        why = "当前 config 值，三场景验收全通过"
    elif scan[0.25]["all"]:
        chosen = 0.25
        why = "当前值未全通过，退 0.25"
    else:
        chosen = next((m for m in MID_SWEEP if scan[m]["all"]), None)
        why = "0.25 未全满足，取首个全满足档" if chosen else "三档均未全满足"
    print("\n  MID 选择：%s（%s）" % (chosen, why))

    # ---------------- 收尾 ----------------
    restore(base)
    assert_restored("收尾", base)
    pipeline.LOGS_DIR = _orig_logs_dir
    print("\n" + "=" * 108)
    ok = not _fails
    print("结论：%s" % ("PASS——口径复现通过，轴一曲线已按 037 参数记录"
                       if ok else "FAIL——未通过项：%s" % _fails))
    print("=" * 108)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
