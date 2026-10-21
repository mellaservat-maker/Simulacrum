# ============================================================
# probe_read_side.py · 读取侧验证探针
# 验证会话模式下，归档后读取侧能真正拿到记忆包
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from pipeline import run_session


def main():
    all_ok = True

    print("=== 读取侧开关状态 ===")
    print(f"MEMORY_READ_ENABLED = {config.MEMORY_READ_ENABLED}")
    if config.MEMORY_READ_ENABLED is not True:
        print("  警告：读取侧未开启")
    print()

    print("=== 运行 8 轮被骂会话 ===")
    logs, st = run_session(["被骂"] * 8, feature_mode="fixture")

    print(f"总轮数: {len(logs)}")
    print(f"归档包数: {len(st['archiver'].get_packages())}")
    print()

    # 打印每轮的 recalled_memory 状态
    for i, log in enumerate(logs):
        recalled = log.get("recalled_memory")
        if recalled is None:
            status = "None"
        else:
            status = f"有 (emotion_value={recalled.get('emotion_value')}, polarity={recalled.get('emotion_polarity')})"
        print(f"  第 {i+1} 轮: recalled_memory = {status}")
    print()

    # 断言 1：第 6 轮 recalled_memory 为 None
    # （第 6 轮结束时触发归档，但第 6 轮读取的是归档前的状态，所以应该还是 None）
    r6 = logs[5].get("recalled_memory")
    check1 = r6 is None
    print(f"断言 1：第 6 轮 recalled_memory 为 None → {'通过' if check1 else '失败'}")
    if not check1:
        print(f"  实际值: {r6}")
    all_ok &= check1

    # 断言 2：第 7 轮 recalled_memory 非 None，且含 emotion_value 和 emotion_polarity
    r7 = logs[6].get("recalled_memory")
    check2 = (r7 is not None and
              "emotion_value" in r7 and
              "emotion_polarity" in r7)
    print(f"断言 2：第 7 轮 recalled_memory 非 None 且含字段 → {'通过' if check2 else '失败'}")
    if not check2:
        print(f"  实际值: {r7}")
    all_ok &= check2

    # 断言 3：第 8 轮也非 None
    r8 = logs[7].get("recalled_memory")
    check3 = r8 is not None
    print(f"断言 3：第 8 轮 recalled_memory 非 None → {'通过' if check3 else '失败'}")
    if not check3:
        print(f"  实际值: {r8}")
    all_ok &= check3

    print()
    print(f"最终 MEMORY_READ_ENABLED = {config.MEMORY_READ_ENABLED}")
    print()

    if all_ok:
        print("探针完成，全部断言通过")
        return 0
    else:
        print("TODO_REVIEW: 断言失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
