# ============================================================
# probe_t24.py · T24 探针
# 验证 recalled_memory["emotion_value"] 确实影响本轮慢通路 D_slow
#
# 做法：跑 8 轮「被骂」（run_session，跨轮共享 state）
#   写入侧阈值=6，第 6 轮末触发归档 -> 第 7 轮起读取侧能召回
# 打印第 7、8 轮的 recalled_memory / ReframedContext / D_slow / action
#
# 对照方式：包装 pipeline.run_slow_path，在同一轮的同一组输入上
#   再算一次「假装没有召回」（ReframedContext 减掉 delta）的 D_slow，
#   两个值在同一轮内比较，排除海马体历史累积的干扰。
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from pipeline import run_session
from slow_path import run_slow_path as _real_slow_path

# 记录每一轮慢通路的真实输入，用于事后做同轮 A/B 对照
records_raw = []


def recording_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S=None):
    """包装慢通路：记录本轮真实输入，行为与原函数完全一致。

    [012] 新增 S 参数透传（pipeline 两处链现传硬编码占位 S）。
    """
    real_D, real_V = _real_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S)
    records_raw.append({
        "x": np.asarray(x).copy(),
        "RC": np.asarray(ReframedContext).copy(),
        "M": np.asarray(M).copy(),
        "B": np.asarray(B).copy(),
        "w_PFC": np.asarray(w_PFC).copy(),
        "hab": float(HabituationSignal),
        "D": real_D,
    })
    return real_D, real_V


def fmt(v, n=6):
    return np.array2string(np.asarray(v), precision=n, separator=", ")


def main():
    # 接管 pipeline 里的慢通路引用，只记录不改行为
    pipeline.run_slow_path = recording_slow_path

    print("=== 1. 跑 8 轮「被骂」（run_session，fixture 模式）===")
    logs, st = run_session(["被骂"] * 8, feature_mode="fixture", seed=42)
    print("8 轮完成")
    print()

    print("=== 2. 归档包数量 ===")
    print("archiver.get_packages() 数量 = %d" % len(st["archiver"].get_packages()))
    print()

    print("=== 3. 逐轮概览 ===")
    print("轮次 | recalled_memory | delta | D_slow | action")
    for i, lg in enumerate(logs, start=1):
        rm = lg["recalled_memory"]
        rm_s = "None" if rm is None else ("ev=%.4f" % rm["emotion_value"])
        print("%3d | %-14s | %.4f | %.6f | %s"
              % (i, rm_s, lg["recalled_emotion_delta"], lg["D_slow"], lg["action"]))
    print()

    print("=== 4. 第 7 轮详情（首个有召回的轮次）===")
    lg7 = logs[6]
    print("recalled_memory          = %s" % (lg7["recalled_memory"],))
    print("recalled_emotion_delta   = %.6f" % lg7["recalled_emotion_delta"])
    print("ReframedContext          = %s" % fmt(lg7["ReframedContext"]))
    print("D_slow                   = %.8f" % lg7["D_slow"])
    print("action                   = %s" % lg7["action"])
    print()

    print("=== 5. 第 8 轮详情 ===")
    lg8 = logs[7]
    print("recalled_memory          = %s" % (lg8["recalled_memory"],))
    print("recalled_emotion_delta   = %.6f" % lg8["recalled_emotion_delta"])
    print("ReframedContext          = %s" % fmt(lg8["ReframedContext"]))
    print("D_slow                   = %.8f" % lg8["D_slow"])
    print("action                   = %s" % lg8["action"])
    print()

    print("=== 6. 同轮 A/B 对照（同一 M、同一 B，只撤销召回增量）===")
    cf7 = _real_slow_path(records_raw[6]["x"], records_raw[6]["RC"] - lg7["recalled_emotion_delta"],
                          records_raw[6]["M"], records_raw[6]["B"],
                          records_raw[6]["w_PFC"], records_raw[6]["hab"])
    cf8 = _real_slow_path(records_raw[7]["x"], records_raw[7]["RC"] - lg8["recalled_emotion_delta"],
                          records_raw[7]["M"], records_raw[7]["B"],
                          records_raw[7]["w_PFC"], records_raw[7]["hab"])
    print("第 7 轮 D_slow 有召回 = %.8f，无召回对照 = %.8f，差 = %.3e"
          % (lg7["D_slow"], cf7[0], abs(lg7["D_slow"] - cf7[0])))
    print("第 8 轮 D_slow 有召回 = %.8f，无召回对照 = %.8f，差 = %.3e"
          % (lg8["D_slow"], cf8[0], abs(lg8["D_slow"] - cf8[0])))
    print("第 7 轮 V 有召回 = %s" % fmt(lg7["V"]))
    print("第 7 轮 V 无召回 = %s" % fmt(cf7[1]))
    print()

    print("=== 7. 前 6 轮 delta 全部为 0 ===")
    pre_deltas = [lg["recalled_emotion_delta"] for lg in logs[:6]]
    print("前 6 轮 recalled_emotion_delta = %s" % pre_deltas)
    print()

    print("=== 8. 断言 ===")
    problems = []

    if not all(d == 0.0 for d in pre_deltas):
        problems.append("前 6 轮 delta 应全为 0.0，实际 %s" % pre_deltas)
    else:
        print("前 6 轮 delta 全为 0.0: PASS")

    if lg7["recalled_memory"] is None:
        problems.append("第 7 轮应有召回，实际为 None")
    elif lg7["recalled_emotion_delta"] == 0.0:
        problems.append("第 7 轮 delta 应非 0，实际 0.0（emotion_value 恰为 0）")
    else:
        print("第 7 轮召回 emotion_value=%.4f -> delta=%.4f: PASS"
              % (lg7["recalled_memory"]["emotion_value"], lg7["recalled_emotion_delta"]))

    if lg8["recalled_memory"] is None:
        problems.append("第 8 轮应有召回，实际为 None")
    else:
        print("第 8 轮同样有召回: PASS")

    if abs(lg7["D_slow"] - cf7[0]) < 1e-12:
        problems.append("第 7 轮撤销召回增量后 D_slow 完全不变，召回未进入慢通路")
    else:
        print("第 7 轮撤销召回增量后 D_slow 改变（差 %.3e）: PASS" % abs(lg7["D_slow"] - cf7[0]))

    if abs(lg8["D_slow"] - cf8[0]) < 1e-12:
        problems.append("第 8 轮撤销召回增量后 D_slow 完全不变，召回未进入慢通路")
    else:
        print("第 8 轮撤销召回增量后 D_slow 改变（差 %.3e）: PASS" % abs(lg8["D_slow"] - cf8[0]))

    # delta 是否真的进了 ReframedContext：还原海马体原始输出后应落在 tanh 值域内
    rc_raw7 = records_raw[6]["RC"] - lg7["recalled_emotion_delta"]
    if np.all(np.abs(rc_raw7) <= 1.0 + 1e-9):
        print("撤销 delta 后 ReframedContext 落在 tanh 值域内（逐维 |x|<=1）: PASS")
    else:
        print("撤销 delta 后 ReframedContext = %s，逐维绝对值最大 %.6f"
              % (fmt(rc_raw7), float(np.max(np.abs(rc_raw7)))))

    if problems:
        print()
        print("TODO_REVIEW")
        for p in problems:
            print("  " + p)
        return 1

    print()
    print("全部断言通过：召回记忆的 emotion_value 已作为标量增量加进 ReframedContext，")
    print("并经由 x + ReframedContext 进入慢通路 z_s，影响 D_slow。")
    return 0


if __name__ == "__main__":
    sys.exit(main())