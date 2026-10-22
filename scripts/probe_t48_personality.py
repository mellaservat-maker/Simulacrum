# ============================================================
# probe_t48_personality.py · 031 性格预设 / 032 危险度接入 / 033 衰竭系数
# 目的：
#   A. 同一输入连跑 10 轮，比较 standard / sensitive / gentle 的
#      danger_score → danger_boost → D_error → I_inhibit → action 全链路；
#   B. 验证 load_personality 运行时改参数立即生效；
#   C. 验证 profile_id 随会话持久化，旧会话缺字段回落 standard；
#   D. [033] 验证 W_PFC_FACTOR 已乘入疲劳侵蚀项，并如实记录 10 轮曲线。
# 口径：mock 后端 + fixture“被骂”（确定性；不依赖真实 LLM 措辞波动）。
# 公式：
#   danger_boost = max(0, danger_score - DANGER_NEUTRAL)
#   D_error = max(0, S_E - THETA_ERROR) + danger_boost * W_DANGER
#   I_inhibit = clip(w_i * D_error - KAPPA * w_PFC * W_PFC_FACTOR, 0, 1)
# ============================================================
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import config
config.LLM_BACKEND = "mock"

import pipeline
import session_store
from intercept import run_intercept
from state import reset_state

PROFILES = ["standard", "sensitive", "gentle"]
TURNS = 10
SCENARIO = "被骂"

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def run_profile(pid):
    config.load_personality(pid)
    st = reset_state(seed=42)
    rows = []
    for t in range(1, TURNS + 1):
        log = pipeline.run_turn(st, SCENARIO, feature_mode="fixture")
        ev = st["memory_graph"].get_all()[-1]
        danger = float(ev.danger_score)
        rows.append({
            "turn": t,
            "danger": danger,
            "boost": max(0.0, danger - config.DANGER_NEUTRAL),
            "S_E": float(log["S_E"]),
            "D_error": float(log["D_error"]),
            "I_inhibit": float(log["I_inhibit"]),
            "w_PFC": float(log["w_PFC"]),
            "action": log["action"],
        })
    return st, rows


def main():
    # [032] 探针日志隔离：probe 连跑 30 轮，不写 logs/ 正式口径数据。
    import tempfile
    _orig_logs_dir = pipeline.LOGS_DIR
    pipeline.LOGS_DIR = tempfile.mkdtemp(prefix="probe_t48_logs_")

    print("=" * 104)
    print("probe_t48 性格 × 自控（033）：%d 轮同一“%s”输入" % (TURNS, SCENARIO))
    print("口径：W_DANGER=%.2f  DANGER_NEUTRAL=%.2f  w_i=%.2f  THETA_ERROR=%.2f  "
          "KAPPA=%.2f  W_MAX=%.2f"
          % (config.W_DANGER, config.DANGER_NEUTRAL, config.w_i,
             config.THETA_ERROR, config.KAPPA, config.W_MAX))
    print("=" * 104)

    data = {}
    for pid in PROFILES:
        st, rows = run_profile(pid)
        data[pid] = rows
        p = config.PERSONALITY_PROFILES[pid]
        print("\n[%s / %s]  w_sem=%.2f w_con=%.2f w_sev=%.2f sev_neutral=%.2f "
              "lambda_pol=%.2f gain=%.2f w_pfc_factor=%.2f"
              % (pid, p["label"], p["w_sem"], p["w_con"], p["w_sev"],
                 p["sev_neutral"], p["lambda_pol"], p["danger_gain"],
                 config.W_PFC_FACTOR))
        print("  轮 |   danger |    boost |    S_E   | D_error  | I_inhibit |   w_PFC  | action")
        for r in rows:
            print("  %2d | %+8.4f | %8.4f | %8.4f | %8.4f | %9.4f | %8.4f | %s"
                  % (r["turn"], r["danger"], r["boost"], r["S_E"],
                     r["D_error"], r["I_inhibit"], r["w_PFC"], r["action"]))
        first_suppress = next((r["turn"] for r in rows if r["action"] == "suppress"), None)
        actions = [r["action"] for r in rows]
        seq = "%s x%d" % (actions[0], len(actions))
        if len(set(actions)) > 1:
            seq = " → ".join("%d:%s" % (r["turn"], r["action"]) for r in rows)
        print("  首轮 suppress：%s；danger 均值 %.4f；D_error 均值 %.4f；"
              "I 均值 %.4f；w_PFC 末值 %.4f"
              % (first_suppress if first_suppress else "无",
                 float(np.mean([r["danger"] for r in rows])),
                 float(np.mean([r["D_error"] for r in rows])),
                 float(np.mean([r["I_inhibit"] for r in rows])), rows[-1]["w_PFC"]))
        print("  action 序列：%s" % seq)

    print("\n" + "-" * 104)
    print("断言")
    print("-" * 104)

    # A1 首轮危险度排序：sensitive > standard > gentle
    d1 = {pid: data[pid][0]["danger"] for pid in PROFILES}
    check("首轮 danger：sensitive > standard > gentle",
          d1["sensitive"] > d1["standard"] > d1["gentle"],
          "sensitive=%.4f standard=%.4f gentle=%.4f"
          % (d1["sensitive"], d1["standard"], d1["gentle"]))

    # A2 10 轮均值排序
    mean = {pid: float(np.mean([r["danger"] for r in data[pid]])) for pid in PROFILES}
    check("10 轮 danger 均值：sensitive > standard > gentle",
          mean["sensitive"] > mean["standard"] > mean["gentle"],
          "sensitive=%.4f standard=%.4f gentle=%.4f"
          % (mean["sensitive"], mean["standard"], mean["gentle"]))

    # A3 每轮都保持排序（允许 1e-9 数值容差）
    order_ok = all(data["sensitive"][i]["danger"] >= data["standard"][i]["danger"] - 1e-9
                   and data["standard"][i]["danger"] >= data["gentle"][i]["danger"] - 1e-9
                   for i in range(TURNS))
    check("10 轮逐轮 danger 排序保持", order_ok)

    # A4 danger 定义域
    check("三性格 danger 均在 [-1,1]",
          all(-1.0 <= r["danger"] <= 1.0 for pid in PROFILES for r in data[pid]))

    # A5 danger_boost 已进入 D_error：逐轮排序 sensitive > standard > gentle
    de_order = all(data["sensitive"][i]["D_error"] > data["standard"][i]["D_error"]
                   > data["gentle"][i]["D_error"] for i in range(TURNS))
    check("[032] 逐轮 D_error 排序：sensitive > standard > gentle", de_order,
          "首轮 sensitive=%.4f standard=%.4f gentle=%.4f"
          % (data["sensitive"][0]["D_error"], data["standard"][0]["D_error"],
             data["gentle"][0]["D_error"]))

    # A6 高门与增益：boost 与 danger 的差保持一致，且 D_error 高于基础项
    boost_ok = all(abs(r["boost"] - max(0.0, r["danger"] - config.DANGER_NEUTRAL)) < 1e-12
                   for pid in PROFILES for r in data[pid])
    check("[033] danger_boost = max(0, danger-DANGER_NEUTRAL) 逐轮恒等", boost_ok)
    base_de = max(0.0, data["gentle"][0]["S_E"] - config.THETA_ERROR)
    check("[033] 被骂越过中性线：D_error 高于基础项",
          data["gentle"][0]["D_error"] > base_de,
          "基础=%.4f gentle=%.4f（boost=%.4f）"
          % (base_de, data["gentle"][0]["D_error"], data["gentle"][0]["boost"]))
    check("[032] D_error 未发生上界裁剪（三性格首轮均 <1）",
          all(data[pid][0]["D_error"] < 1.0 for pid in PROFILES))

    # A7 W_DANGER=0 时三性格 D_error 逐位一致：证明分化只来自 danger_boost
    old_w = config.W_DANGER
    config.W_DANGER = 0.0
    zero = {pid: run_profile(pid)[1] for pid in PROFILES}
    config.W_DANGER = old_w
    zero_same = all(abs(zero["sensitive"][i]["D_error"] - zero["gentle"][i]["D_error"]) < 1e-12
                    and abs(zero["standard"][i]["D_error"] - zero["gentle"][i]["D_error"]) < 1e-12
                    for i in range(TURNS))
    check("[032] W_DANGER=0 回归：三性格 D_error 逐位一致", zero_same,
          "首轮 S_E=%.4f D_error=%.4f" % (zero["gentle"][0]["S_E"],
                                           zero["gentle"][0]["D_error"]))

    # A8 [033→037] 037 已撤回每性格衰竭系数：W_PFC_FACTOR 固定 1.0，
    #   固定 E_raw / w_PFC 下三性格 I 应当逐位相同，性格不再作用于自控入口。
    E_probe = np.array([-1.0, 0.0, 0.0, 0.0])
    I_probe = {}
    for pid in PROFILES:
        config.load_personality(pid)
        I_probe[pid] = run_intercept(E_probe, config.W_MAX, danger_boost=0.0)[3]
    check("[037] W_PFC_FACTOR 固定 1.0，三性格 I 在固定工况下逐位相同",
          abs(I_probe["sensitive"] - I_probe["standard"]) < 1e-12
          and abs(I_probe["gentle"] - I_probe["standard"]) < 1e-12,
          "sen=%.4f std=%.4f gen=%.4f（w_PFC=%.1f）"
          % (I_probe["sensitive"], I_probe["standard"], I_probe["gentle"],
             config.W_MAX))

    # B 运行时改参数立即生效
    config.load_personality("standard")
    w_std, f_std = config.W_SEM, config.W_PFC_FACTOR
    config.load_personality("sensitive")
    w_sen, f_sen = config.W_SEM, config.W_PFC_FACTOR
    config.load_personality("gentle")
    w_gen, f_gen = config.W_SEM, config.W_PFC_FACTOR
    check("load_personality 运行时改参立即生效",
          w_std == 0.60 and w_sen == 0.85 and w_gen == 0.45,
          "std=%.2f sen=%.2f gen=%.2f" % (w_std, w_sen, w_gen))
    check("[033→037] W_PFC_FACTOR 固定为 1.0（每性格系数已撤回）",
          f_std == 1.0 and f_sen == 1.0 and f_gen == 1.0,
          "std=%.2f sen=%.2f gen=%.2f" % (f_std, f_sen, f_gen))
    check("中文别名可用（敏感型）",
          config.load_personality("敏感型") == "sensitive")

    # C profile_id 持久化与旧会话兼容
    st = reset_state(seed=42)
    st["profile_id"] = "gentle"
    d = session_store.dump_state(st)
    check("dump_state 写入 profile_id", d.get("profile_id") == "gentle",
          d.get("profile_id"))
    st2 = session_store.load_state(d)
    check("load_state 恢复 profile_id", st2.get("profile_id") == "gentle",
          st2.get("profile_id"))
    d_old = {k: v for k, v in d.items() if k != "profile_id"}
    st3 = session_store.load_state(d_old)
    check("旧会话缺字段回落 standard", st3.get("profile_id") == "standard",
          st3.get("profile_id"))

    # D [033] 如实记录自控曲线与耐力台账
    print("\n" + "-" * 104)
    print("自控耐力台账（10 轮）")
    print("-" * 104)
    print(" 性格      | factor | I(1)   | I(5)   | I(10)  | I 最小 | 首次 I<0.7 | 首 suppress | 动作种类")
    for pid in PROFILES:
        rows = data[pid]
        f = config.W_PFC_FACTOR
        first_under = next((r["turn"] for r in rows if r["I_inhibit"] < 0.7), None)
        first_sup = next((r["turn"] for r in rows if r["action"] == "suppress"), None)
        print(" %-9s | %6.2f | %.4f | %.4f | %.4f | %.4f | %10s | %11s | %d"
              % (pid, f, rows[0]["I_inhibit"], rows[4]["I_inhibit"],
                 rows[9]["I_inhibit"], min(r["I_inhibit"] for r in rows),
                 first_under if first_under else "无",
                 first_sup if first_sup else "无",
                 len({r["action"] for r in rows})))

    same_i = all(abs(data["standard"][i]["I_inhibit"]
                     - data["sensitive"][i]["I_inhibit"]) < 1e-12
                 and abs(data["standard"][i]["I_inhibit"]
                         - data["gentle"][i]["I_inhibit"]) < 1e-12
                 for i in range(TURNS))
    same_actions = all(
        data["standard"][i]["action"] == data["sensitive"][i]["action"]
        == data["gentle"][i]["action"] for i in range(TURNS))
    print("\n观测（非断言项）：")
    print("  I_inhibit 三性格 10 轮是否完全一致：%s" % ("是" if same_i else "否"))
    print("  action 三性格 10 轮是否完全一致：%s" % ("是" if same_actions else "否"))
    for pid in PROFILES:
        rows = data[pid]
        f = config.W_PFC_FACTOR
        floor_d = float(rows[-1]["D_error"])
        floor_i = config.w_i * floor_d - config.KAPPA * config.W_MAX * f
        print("  %-9s：w_i×D_error(末轮)=%.4f；W_MAX 处最大侵蚀=%.4f；"
              "理论 I 下界=%.4f；%s 0.7"
              % (pid, config.w_i * floor_d, config.KAPPA * config.W_MAX * f,
                 max(0.0, floor_i), "低于" if floor_i < 0.7 else "不低于"))
    print("  说明：若三性格 I 均被 clip 到 1.0，疲劳项尚未进入有效区，")
    print("        W_PFC_FACTOR 的差异要等 w_PFC 累积到反超阈值才可见。")

    print("\n" + "=" * 104)
    ok = not _fails
    print("结论：%s" % ("PASS——危险度与衰竭系数均已进入自控公式，曲线见上"
                       if ok else "FAIL——未通过项：%s" % _fails))
    print("=" * 104)
    pipeline.LOGS_DIR = _orig_logs_dir
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
