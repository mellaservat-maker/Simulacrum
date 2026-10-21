# ============================================================
# probe_session.py · M2-D 修正探针
# 验证会话模式下写入侧真的跨轮积累
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import run_session


def check(label, condition, detail=""):
    if condition:
        print(f"  {label} ✓")
        return True
    print(f"  {label} ✗  {detail}")
    return False


def main():
    all_ok = True

    # ===== 实验一：7 轮 =====
    print("=== 实验一：7 轮会话 ===")
    logs, st = run_session(["被骂"] * 7, feature_mode="fixture")

    print(f"  len(logs)                              = {len(logs)}")
    print(f"  len(st['memory_buffer'])               = {len(st['memory_buffer'])}")
    print(f"  len(st['raw_log'])                     = {len(st['raw_log'])}")
    print(f"  len(st['archiver'].get_packages())     = {len(st['archiver'].get_packages())}")
    print(f"  st['round_counter']                     = {st['round_counter']}")
    print()

    all_ok &= check("len(logs) == 7", len(logs) == 7, f"实际 = {len(logs)}")
    all_ok &= check("memory_buffer 剩 3 轮", len(st["memory_buffer"]) == 3,
                    f"实际 = {len(st['memory_buffer'])}")
    all_ok &= check("raw_log 有 4 轮", len(st["raw_log"]) == 4,
                    f"实际 = {len(st['raw_log'])}")
    all_ok &= check("archiver 有 1 个包", len(st["archiver"].get_packages()) == 1,
                    f"实际 = {len(st['archiver'].get_packages())}")
    all_ok &= check("round_counter == 7", st["round_counter"] == 7,
                    f"实际 = {st['round_counter']}")

    if not all_ok:
        print("TODO_REVIEW: 实验一断言失败")
        return 1

    # ===== 实验二：13 轮，验证两次归档 =====
    print()
    print("=== 实验二：13 轮会话（两次归档） ===")
    logs2, st2 = run_session(["被骂"] * 13, feature_mode="fixture")

    pkg_count = len(st2["archiver"].get_packages())
    raw_len = len(st2["raw_log"])
    buf_len = len(st2["memory_buffer"])
    rnd = st2["round_counter"]

    print(f"  len(logs)                              = {len(logs2)}")
    print(f"  len(st['memory_buffer'])               = {buf_len}")
    print(f"  len(st['raw_log'])                     = {raw_len}")
    print(f"  len(st['archiver'].get_packages())     = {pkg_count}")
    print(f"  st['round_counter']                     = {rnd}")
    print()

    # 第 6 轮归档一次（刷掉前 4 轮，raw_log += 4，buffer 剩 2）
    # 第 7-11 轮追加后 buffer 到 7（2+5=7），第 12 轮追加到 8 → 触发第二次归档
    # 第二次归档刷掉前 6 轮（2+5-1=6？算一下）
    # 实际：第 6 轮归档后 buffer = 2（round 5,6）
    # 第 7 轮追加 → 3
    # 第 8 轮 → 4
    # 第 9 轮 → 5
    # 第 10 轮 → 6 → should_archive=True → 归档 → 刷掉前 4 轮（round 5,6,7,8），剩 2（round 9,10）
    # 第 11 轮 → 3
    # 第 12 轮 → 4
    # 第 13 轮 → 5
    # 所以 13 轮后：
    #   归档次数 = 2（第 6 轮和第 10 轮）
    #   raw_log = 4 + 4 = 8 轮
    #   buffer = 5 轮（round 9,10,11,12,13）
    #   packages = 2

    all_ok &= check("len(logs) == 13", len(logs2) == 13, f"实际 = {len(logs2)}")
    all_ok &= check("archiver 有 2 个包", pkg_count == 2, f"实际 = {pkg_count}")
    all_ok &= check("raw_log 有 8 轮", raw_len == 8, f"实际 = {raw_len}")
    all_ok &= check("memory_buffer 剩 5 轮", buf_len == 5, f"实际 = {buf_len}")
    all_ok &= check("round_counter == 13", rnd == 13, f"实际 = {rnd}")

    # 验证 raw_log 里的轮次 id
    all_ids = st2["raw_log"].all_ids()
    print(f"  raw_log all_ids = {all_ids}")
    # 两次归档：第一次归档 round 1~4，第二次归档 round 5~8
    all_ok &= check("raw_log 含 8 条 id: 1~8", all_ids == list(range(1, 9)),
                    f"实际 = {all_ids}")

    print()
    if all_ok:
        print("探针完成，全部断言通过")
        return 0
    else:
        print("TODO_REVIEW: 断言失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
