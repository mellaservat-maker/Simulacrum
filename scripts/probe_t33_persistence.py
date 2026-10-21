# ============================================================
# probe_t33_persistence.py · T33 探针
# 验证五个跨轮状态量（E_prev / w_PFC / B / M_prev / HabituationSignal_prev）
# 在 run_turn 末尾写回共享 state 后，跨轮真实累积。
#
# 做法：跑 8 轮「被骂」（run_session，fixture 模式，跨轮共享 state）。
#   包装 pipeline.run_turn：每轮返回后记录该轮结束时的 st 快照
#   （写回后的值），行为与原函数完全一致，只观测不改计算。
#   写入侧阈值=6，第 6 轮末归档，第 7 轮起有召回（emotion_value 由 LLM
#   现场生成，不可复现，仅影响第 7、8 轮）。
#
# 观测：
#   1. 逐轮打印五个状态量（E_prev[0]、w_PFC、B[1]、M_prev 范数、
#      HabituationSignal_prev），附 I_inhibit 与 D_fast 供参照。
#   2. w_PFC 第 1 轮到第 8 轮增长轨迹。
#   3. 第 8 轮 I_inhibit 与第 1 轮对比（压制力是否累积）。
#   4. 快通路唤醒分支（D_fast > THETA_FAST=0.5 时 B[1] += 0.3）是否触发，
#      未触发则如实记录，不硬改公式。
# 断言：E_prev 与 w_PFC 在第 2 轮开始发生变化（证明写回生效）。
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline

SCENARIO = "被骂"
ROUNDS = 8

# 每轮结束时的 st 快照（写回后的值）
snapshots = []

_original_run_turn = pipeline.run_turn


def observing_run_turn(state, scenario, feature_mode="fixture"):
    """包装 run_turn：行为不变，返回后记录 st 快照。"""
    log = _original_run_turn(state, scenario, feature_mode=feature_mode)
    snapshots.append({
        "round": len(snapshots) + 1,
        "E_prev_0": float(state["E_prev"][0]),
        "w_PFC": float(state["w_PFC"]),
        "B_1": float(state["B"][1]),
        "M_prev_norm": float(np.linalg.norm(state["M_prev"])),
        "HabituationSignal_prev": float(state["HabituationSignal_prev"]),
    })
    return log


def main():
    # 接管 pipeline 里的 run_turn 引用，只记录不改行为
    pipeline.run_turn = observing_run_turn

    print("=== 1. 跑 %d 轮「被骂」（run_session，fixture 模式，seed=42）===" % ROUNDS)
    logs, st = pipeline.run_session([SCENARIO] * ROUNDS, feature_mode="fixture", seed=42)

    pipeline.run_turn = _original_run_turn
    print("%d 轮完成" % ROUNDS)
    print()

    print("=== 2. 五个跨轮状态量逐轮变化（写回后的 st 值）===")
    print("轮次 | E_prev[0]  | w_PFC     | B[1]      | M_prev范数 | HabituationPrev | I_inhibit | D_fast   | action")
    for i in range(ROUNDS):
        s = snapshots[i]
        lg = logs[i]
        print("%4d | %10.6f | %9.6f | %9.6f | %10.6f | %15.6f | %9.6f | %8.6f | %s"
              % (s["round"], s["E_prev_0"], s["w_PFC"], s["B_1"],
                 s["M_prev_norm"], s["HabituationSignal_prev"],
                 lg["I_inhibit"], lg["D_fast"], lg["action"]))
    print()

    print("=== 3. w_PFC 增长轨迹（第 1 轮到第 %d 轮）===" % ROUNDS)
    traj = " -> ".join("%.6f" % snapshots[i]["w_PFC"] for i in range(ROUNDS))
    print(traj)
    print()

    print("=== 4. I_inhibit 第 8 轮 vs 第 1 轮（压制力是否累积）===")
    i1 = float(logs[0]["I_inhibit"])
    i8 = float(logs[ROUNDS - 1]["I_inhibit"])
    print("第 1 轮 I_inhibit = %.6f" % i1)
    print("第 8 轮 I_inhibit = %.6f" % i8)
    if i8 > i1:
        print("对比：第 8 轮更高（差 %.6f），压制力在累积" % (i8 - i1))
    elif i8 < i1:
        print("对比：第 8 轮更低（差 %.6f）" % (i1 - i8))
    else:
        print("对比：持平")
    print()

    print("=== 5. 快通路唤醒分支记录（D_fast > 0.5 时 B[1] += 0.3）===")
    d_fasts = [float(lg["D_fast"]) for lg in logs]
    b1s = [s["B_1"] for s in snapshots]
    triggered = [i + 1 for i in range(ROUNDS) if d_fasts[i] > 0.5]
    if triggered:
        print("触发轮次：%s" % triggered)
    else:
        print("全程未触发：8 轮 D_fast 最大值为 %.6f，均未超过 THETA_FAST=0.5" % max(d_fasts))
    print("B[1] 逐轮值：%s" % " -> ".join("%.6f" % v for v in b1s))
    print("（如实记录，不硬改公式）")
    print()

    print("=== 6. 断言 ===")
    # 写回生效的核心证据：第 2 轮状态必须与第 1 轮不同
    assert snapshots[1]["E_prev_0"] != snapshots[0]["E_prev_0"], (
        "E_prev 第 2 轮未变化（%.6f -> %.6f），写回可能未生效"
        % (snapshots[0]["E_prev_0"], snapshots[1]["E_prev_0"]))
    print("E_prev 第 2 轮起变化（%.6f -> %.6f）: PASS"
          % (snapshots[0]["E_prev_0"], snapshots[1]["E_prev_0"]))

    assert snapshots[1]["w_PFC"] != snapshots[0]["w_PFC"], (
        "w_PFC 第 2 轮未变化（%.6f -> %.6f）"
        % (snapshots[0]["w_PFC"], snapshots[1]["w_PFC"]))
    print("w_PFC 第 2 轮起变化（%.6f -> %.6f）: PASS"
          % (snapshots[0]["w_PFC"], snapshots[1]["w_PFC"]))

    # 附加核对：快照值应与本轮 log 的 E_final / w_PFC 一致（写回行正确执行）
    for i in range(ROUNDS):
        assert abs(snapshots[i]["E_prev_0"] - float(logs[i]["E_final"][0])) < 1e-12, (
            "第 %d 轮快照 E_prev[0] 与本轮 E_final[0] 不一致" % (i + 1))
        assert abs(snapshots[i]["w_PFC"] - float(logs[i]["w_PFC"])) < 1e-12, (
            "第 %d 轮快照 w_PFC 与本轮 log w_PFC 不一致" % (i + 1))
    print("逐轮核对：快照 E_prev[0]/w_PFC 与本轮 E_final/w_PFC 一致: PASS")
    print()

    print("全部断言通过：五个跨轮状态量已在 run_turn 末尾写回，")
    print("第 2 轮起 E_prev 与 w_PFC 真实累积，情绪惯性与自控强化两条")
    print("设计意图首次具备运行条件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
