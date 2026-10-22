# ============================================================
# probe_t51_breakdown_cycle.py · 039 破防周期性验证
# 口径：mock 后端、fixture“被骂”、seed=42、每性格独立会话、日志临时目录。
# 验证：express 首次轮次、100 轮触发次数、冷却期内的 action、
#       两次 express 的最小间隔（应为 11 轮：1 次破防 + 10 轮冷却）。
# ============================================================
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
config.LLM_BACKEND = "mock"

import pipeline
from state import reset_state

PROFILES = ["standard", "sensitive", "gentle"]
SCENARIO = "被骂"
ROUNDS = 100

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def run(pid):
    config.load_personality(pid)
    st = reset_state(seed=42)
    rows = []
    for t in range(1, ROUNDS + 1):
        log = pipeline.run_turn(st, SCENARIO, feature_mode="fixture")
        rows.append({
            "turn": t,
            "E0": float(log["E_final"][0]),
            "I": float(log["I_inhibit"]),
            "w": float(log["w_PFC"]),
            "action": log["action"],
            "cooldown_after": int(st["express_cooldown"]),
        })
    return rows


def compress(rows):
    out = []
    for r in rows:
        if out and out[-1][0] == r["action"]:
            out[-1][1] += 1
            out[-1][2] = r["turn"]
        else:
            out.append([r["action"], 1, r["turn"]])
    return " → ".join("%s x%d(至%d)" % (a, n, end) for a, n, end in out)


def main():
    _orig_logs_dir = pipeline.LOGS_DIR
    pipeline.LOGS_DIR = tempfile.mkdtemp(prefix="probe_t51_logs_")

    print("=" * 104)
    print("probe_t51 破防周期：%d 轮“%s”（DELTA=%.2f  MID=%.2f  EXPRESS_INH=%.2f  "
          "VALENCE=%.2f）"
          % (ROUNDS, SCENARIO, config.DELTA, config.POLICY_THRESHOLD_INHIBIT_MID,
             config.POLICY_EXPRESS_INHIBIT, config.POLICY_EXPRESS_VALENCE))
    print("=" * 104)

    data = {}
    for pid in PROFILES:
        rows = run(pid)
        data[pid] = rows
        ex = [r["turn"] for r in rows if r["action"] == "express"]
        gaps = [b - a for a, b in zip(ex, ex[1:])]
        print("\n[%s] express 轮次：%s" % (pid, ex if ex else "无"))
        print("  次数=%d；最小间隔=%s；首次=%s；末次=%s"
              % (len(ex), min(gaps) if gaps else "无",
                 ex[0] if ex else "无", ex[-1] if ex else "无"))
        print("  action 全程：%s" % compress(rows))
        print("  冷却检查：%s" % ("通过" if (not gaps or min(gaps) >= 11) else "存在间隔 <11"))
        # 抽样打印冷却段
        if len(ex) >= 2:
            a = ex[0]
            seg = rows[a:a + 11]
            print("  首次破防后 10 轮："
                  + " ".join("%d:%s(cd=%d)" % (r["turn"], r["action"], r["cooldown_after"])
                             for r in seg))

    print("\n" + "-" * 104)
    print("断言")
    print("-" * 104)
    for pid in PROFILES:
        rows = data[pid]
        ex = [r["turn"] for r in rows if r["action"] == "express"]
        gaps = [b - a for a, b in zip(ex, ex[1:])]
        check("%s 100 轮内至少破防 1 次" % pid, len(ex) >= 1, ex)
        check("%s 两次破防间隔 >= 11" % pid, all(g >= 11 for g in gaps), gaps)
        ex_set = set(ex)
        blocked_ok = all((t + k) not in ex_set for t in ex for k in range(1, 11))
        check("%s 每次破防后 10 轮内无 express" % pid, blocked_ok)
        check("%s 冷却期降级动作只含 suppress/task_first" % pid,
              all(r["action"] in ("suppress", "task_first") for r in rows
                  if r["cooldown_after"] > 0 and r["action"] != "express"))
    counts = {p: len([r for r in data[p] if r["action"] == "express"]) for p in PROFILES}
    print("\n  express 次数：standard %d、sensitive %d、gentle %d（预期 6 至 10）"
          % (counts["standard"], counts["sensitive"], counts["gentle"]))

    pipeline.LOGS_DIR = _orig_logs_dir
    print("\n" + "=" * 104)
    ok = not _fails
    print("结论：%s" % ("PASS——破防按周期出现，冷却生效" if ok
                       else "FAIL——未通过项：%s" % _fails))
    print("=" * 104)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
