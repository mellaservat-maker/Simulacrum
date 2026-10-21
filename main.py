# ============================================================
# main.py · V4.2 · M1 入口
# 跑三场景 fixture + encoder 冒烟
# ============================================================

import os
import sys

from pipeline import run_scenario

SCENARIOS = ["被骂", "被夸", "被质疑"]
EXPECTED = {"被骂": "suppress", "被夸": "task_first", "被质疑": "reappraise"}


def main():
    print("=" * 80)
    print("M1 三场景验收 · fixture 模式")
    print("=" * 80)
    all_pass = True
    for sc in SCENARIOS:
        log = run_scenario(sc, feature_mode="fixture")
        ok = log["action"] == EXPECTED[sc]
        all_pass = all_pass and ok
        print(f"{sc:<6} action={log['action']:<12} expected={EXPECTED[sc]:<12} "
              f"{'PASS' if ok else 'FAIL'}")

    print("-" * 80)
    print(f"fixture 验收：{'全部通过' if all_pass else '失败'}")
    print()

    print("=" * 80)
    print("M1 encoder 模式冒烟（不断言 action，只检查字段完整）")
    print("=" * 80)
    smoke_ok = True
    required = ["x", "x_fast", "B", "D_fast", "E_raw", "S_E", "I_inhibit", "E_final", "action"]
    for sc in SCENARIOS:
        try:
            log = run_scenario(sc, feature_mode="encoder")
        except Exception as e:
            print(f"{sc:<6} 冒烟异常：{e}")
            smoke_ok = False
            continue
        missing = [f for f in required if f not in log]
        if missing:
            print(f"{sc:<6} 字段缺失：{missing}")
            smoke_ok = False
        else:
            print(f"{sc:<6} 冒烟通过（action={log['action']}）")

    print("-" * 80)
    print(f"encoder 冒烟：{'全部通过' if smoke_ok else '失败'}")
    print()
    print(f"M1 总验收：{'通过' if (all_pass and smoke_ok) else '失败'}")
    return 0 if (all_pass and smoke_ok) else 1


if __name__ == "__main__":
    sys.exit(main())