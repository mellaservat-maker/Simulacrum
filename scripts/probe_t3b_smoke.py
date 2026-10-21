# ============================================================
# probe_t3b_smoke.py · M3-T3B 冒烟探针
# 验证 T3-B 维度解耦：x = 8 维、M = 1024 维、x_fast = 4 维
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import run_scenario
from hippocampal_evaluator import evaluate as hippocampal_evaluate
from event_parser import parse as event_parse
from state import reset_state
from xinguang import update_xinguang_budget


def main():
    print("=== 1. 跑一轮完整链路（encoder 模式）===")
    log = run_scenario("被骂", feature_mode="encoder")
    print("run_scenario('被骂', feature_mode='encoder') 完成")

    x = np.asarray(log["x"])
    x_fast = np.asarray(log["x_fast"])

    print()
    print("=== 2. 维度实测 ===")

    # M 不在 log 里，单独走一遍海马体评估取M 的真实维度
    st = reset_state(seed=42)
    event = event_parse("你这个AI真蠢，什么都做不好。",
                        event_severity=0.7,
                        importance=0.8,
                        emotion_weight=0.7,
                        unfinished=0.3,
                        danger_score=0.8)
    B_xg = update_xinguang_budget(st["B"][0], st["B"][1])
    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, event.timestamp)
    M = np.asarray(hippo["M"])

    print("x维度      = %d" % x.shape[0])
    print("M 维度      = %d" % M.shape[0])
    print("x_fast 维度 = %d" % x_fast.shape[0])
    print("event_embedding 维度 = %d" % np.asarray(event.event_embedding).shape[0])

    print()
    print("=== 3. 断言 ===")

    problems = []

    if x.shape[0] != 8:
        problems.append("x 维度应为 8，实际 %d" % x.shape[0])
    else:
        print("x = 8: PASS")

    if M.shape[0] != 1024:
        problems.append("M 维度应为 1024，实际 %d" % M.shape[0])
    else:
        print("M = 1024: PASS")

    if x_fast.shape[0] != 4:
        problems.append("x_fast 维度应为 4，实际 %d" % x_fast.shape[0])
    else:
        print("x_fast = 4: PASS")

    if np.asarray(event.event_embedding).shape[0] != 1024:
        problems.append("event_embedding 维度应为 1024，实际 %d"
                        % np.asarray(event.event_embedding).shape[0])
    else:
        print("event_embedding = 1024: PASS")

    print()
    if problems:
        print("TODO_REVIEW")
        for p in problems:
            print("  " + p)
        return 1

    print("全部断言通过：x 与 M 已解耦，x=8 / M=1024 / x_fast=4")
    return 0


if __name__ == "__main__":
    sys.exit(main())