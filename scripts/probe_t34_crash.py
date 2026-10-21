# ============================================================
# probe_t34_crash.py · T34 生死门探针（009.5 批次丁创建，012/013 演进）
# 在总阀门（T33）已通的基础上，验证：
#   L23 召回映射公式 012 定版：recalled_delta_intended = emotion_value * sign
#     （强度×方向；废弃 (v*2-1)*sign——该式 negative 且 v<0.5 时产出正向增量）
#   L28 clamp 防爆（ReframedContext 截断到 [-1,1]）
#   双记录（intended / applied）
#   012 损耗型反转：I_inhibit = clip(w_i*D_error - KAPPA*w_PFC + b_i, 0, 1)，
#     w_PFC 从"防御力"变为"意志力疲劳值"，累积越高抑制越弱（破防）；
#     KAPPA=0.5（014 裁决：0.2→0.5 看真实防线崩溃），W_MAX=2.0；
#     不动点 I* = w_i*D_error/(1+KAPPA·DELTA/(1-RHO))，详见 probe_t35_collapse。
#   012 语义信号 S：pipeline 两处链接入 evaluator 真提取（013 号工作），
#     S=[valence∈-1~1, arousal, dominance, intensity]；被骂实测 valence≈-0.9，
#     经 c_E=1.0 进入 raw_emotion_impact[0]（负效价 → E_raw[0] 压负 → 抑制启动）。
#     （013 前为硬编码占位 S=[0.8,1.0,0.5,0.5]，把攻击当正效价注入，曾使被骂
#       E_raw[0] 转正、D_error 恒 0、抑制熄火——该配置的观测见 012.1 第 4.3 节。）
#
# 做法：跑 15 轮「被骂」（run_session，fixture 模式，跨轮共享 state）。
#   包装 pipeline.run_turn：每轮返回后记录 st 快照与本轮 log，
#   行为与原函数完全一致，只观测不改计算。
#
# 时间线：写入侧阈值=6，缓冲区满即归档——第 6 轮末首次归档（覆盖第 1~4
#   轮），第 10 轮末第二次归档（覆盖第 5~8 轮），第 14 轮末第三次归档
#   （覆盖第 9~12 轮）；第 7~10 轮读第一个包，第 11~14 轮读第二个包，
#   第 15 轮读第三个包。emotion_value 由 LLM 现场生成，不可复现。
#   故第 1~6 轮完全确定可复现，第 7 轮起依赖 LLM 归档值（三段各一个值）。
#
# 观测：
#   1. 逐轮打印 E_prev[0] / w_PFC / B[1] / S_E / D_error / I_inhibit / action。
#   2. 破防分析（012 损耗型）：反事实 clip(w_i*D_error + b_i)（无疲劳项）
#      仍为 1 而实测 I_inhibit < 1 → 疲劳侵蚀生效；报告破防首轮与
#      KAPPA*w_PFC 每轮侵蚀量。
#   3. 召回介入轮（第 7 轮起）：打印归档原始值、intended（映射后）、
#      applied（截断后实际注入），两者差异即 clamp 实际触顶的证据；
#      附 ReframedContext 逐维 min/max 观察是否贴 [-1,1] 边界。
#   4. 极端输入单轮（fresh state）：记录传入 S=[真提取值] 时的
#      z_s / z_v / E_raw[0] / I_inhibit；被骂语义下 E_raw[0] 应为负且
#      I_inhibit 启动（正效价占位时代的"转正"预期已随 013 真提取废弃）。
#
# 断言（数据完整性，不对结果下结论）：
#   A. 15 轮齐全、字段齐全、写回一致（快照 vs log）。
#   B. 召回轮 intended 等于 归档值*sign(polarity)（L23 公式 012 定版）。
#   C. ReframedContext 每轮每维都在 [-1,1] 内（L28 守卫自证）。
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from config import (w_i, b_i, KAPPA, W_MAX, S_DIM, c_E,
                    w_s, u_s, v_s, W_v, U_v, v_v, b_v, b_s, ETA, W_H, e_s, f_v)
from state import reset_state
from slow_path import run_slow_path as _real_slow_path

SCENARIO = "被骂"
ROUNDS = 15

snapshots = []
logs_captured = []

_original_run_turn = pipeline.run_turn


def observing_run_turn(state, scenario, feature_mode="fixture"):
    """包装 run_turn：行为不变，返回后记录 st 快照与 log。"""
    log = _original_run_turn(state, scenario, feature_mode=feature_mode)
    snapshots.append({
        "round": len(snapshots) + 1,
        "E_prev_0": float(state["E_prev"][0]),
        "w_PFC": float(state["w_PFC"]),
        "B_1": float(state["B"][1]),
    })
    logs_captured.append(log)
    return log


def main():
    pipeline.run_turn = observing_run_turn

    print("=== 1. 跑 %d 轮「被骂」（run_session，fixture 模式，seed=42，S 由 evaluator 真提取）===" % ROUNDS)
    logs, st = pipeline.run_session([SCENARIO] * ROUNDS, feature_mode="fixture", seed=42)

    pipeline.run_turn = _original_run_turn
    print("%d 轮完成（第 6/10/14 轮末三次归档，第 7 轮起召回介入，第 15 轮读第三个包）" % ROUNDS)
    print()

    print("=== 2. 逐轮状态表（写回后的 st 快照 + 本轮 log）===")
    print("轮次 | E_prev[0]  | w_PFC     | B[1]      | S_E      | D_error  | I_inhibit | action")
    for i in range(ROUNDS):
        s = snapshots[i]
        lg = logs_captured[i]
        print("%4d | %10.6f | %9.6f | %9.6f | %8.6f | %8.6f | %9.6f | %s"
              % (s["round"], s["E_prev_0"], s["w_PFC"], s["B_1"],
                 lg["S_E"], lg["D_error"], lg["I_inhibit"], lg["action"]))
    print()

    print("=== 3. 破防分析（012 损耗型：疲劳侵蚀是否拖垮 I_inhibit）===")
    # 反事实：I_inhibit = clip(w_i*D_error - KAPPA*w_PFC + b_i, 0, 1)
    # 去掉疲劳项后 clip(w_i*D_error + b_i) 仍为 1 而实测 < 1 → 疲劳侵蚀生效
    print("轮次 | I_inhibit | 反事实(无疲劳项) | KAPPA*w_PFC | 归因")
    first_break = None
    for i in range(ROUNDS):
        lg = logs_captured[i]
        de = float(lg["D_error"])
        ii = float(lg["I_inhibit"])
        wp = float(lg["w_PFC"])
        cf = float(np.clip(w_i * de + b_i, 0.0, 1.0))
        kap = KAPPA * wp
        if ii < 1.0 - 1e-9 and first_break is None:
            first_break = i + 1
        if ii >= 1.0 - 1e-9:
            attr = "未破防（满抑制）"
        elif cf >= 1.0 - 1e-9:
            attr = "疲劳侵蚀破防"
        else:
            attr = "D_error 自然回落"
        print("%4d | %9.6f | %16.6f | %11.6f | %s"
              % (i + 1, ii, cf, kap, attr))
    print()
    if first_break is not None:
        print("破防首轮：第 %d 轮（I_inhibit 首次跌破 1.0）" % first_break)
    else:
        print("全程未破防（I_inhibit 恒为 1.0）")
    print("参数注：KAPPA=%.2f，W_MAX=%.1f，疲劳项最大侵蚀 = KAPPA*W_MAX = %.2f"
          % (KAPPA, W_MAX, KAPPA * W_MAX))
    print("（损耗型语义：w_PFC 累积 → 抑制减弱 → E_final 幅值回升 → 情绪冲破防线）")
    print()

    print("=== 4. 召回介入轮：L23 映射 + L28 clamp + 双记录 ===")
    recall_rounds = 0
    for i in range(ROUNDS):
        lg = logs_captured[i]
        rm = lg["recalled_memory"]
        if rm is None:
            continue
        recall_rounds += 1
        raw_v = float(rm["emotion_value"])
        intended = float(lg["recalled_delta_intended"])
        applied = float(lg["recalled_delta_applied"])
        rc = np.asarray(lg["ReframedContext"], dtype=float)
        diff = intended - applied
        print("第 %2d 轮：归档值=%.6f polarity=%s  intended=%.6f（-1~1）  applied=%.6f"
                  % (i + 1, raw_v, rm["emotion_polarity"], intended, applied))
        print("       差异 intended-applied = %+.6f  %s"
              % (diff, "→ clamp 实际触顶" if abs(diff) > 1e-9 else "→ 未触顶（全额注入）"))
        print("       ReframedContext 逐维 min=%+.6f max=%+.6f（边界 ±1.0）"
              % (float(np.min(rc)), float(np.max(rc))))
    if recall_rounds == 0:
        print("（无召回轮次）")
    print()

    print("=== 5. E_prev 轨迹（总阀门 + 召回注入 + S 真提取后的情绪惯性）===")
    print(" -> ".join("%.6f" % snapshots[i]["E_prev_0"] for i in range(ROUNDS)))
    print()

    print("=== 6. 极端输入单轮（fresh state，记录 z_s / z_v）===")
    # 包装 pipeline.run_slow_path 记录慢通路内部量（同 probe_t24 手法）
    slow_records = []

    def recording_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S=None):
        if S is None:
            S = np.zeros(S_DIM)
        z_s = (w_s @ (x + ReframedContext)
               + u_s @ M
               + v_s @ B
               - ETA * w_PFC
               + W_H * HabituationSignal
               + b_s
               + e_s @ S)
        z_v = W_v @ (x + ReframedContext) + U_v @ M \
            + v_v * (1.0 / (1.0 + np.exp(-z_s))) + b_v + f_v @ S
        real_D, real_V = _real_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S)
        slow_records.append({
            "S": np.asarray(S, dtype=float).copy(),
            "z_s": float(z_s),
            "z_v": np.asarray(z_v, dtype=float).copy(),
            "D": real_D,
            "V": np.asarray(real_V, dtype=float).copy(),
        })
        return real_D, real_V

    pipeline.run_slow_path = recording_slow_path
    st_fresh = reset_state(seed=42)
    lg = pipeline.run_turn(st_fresh, SCENARIO, feature_mode="fixture")
    pipeline.run_slow_path = _real_slow_path

    rec = slow_records[-1]
    print("传入 S = %s（S_valence=%.1f，c_E=%.1f）" % (np.round(rec["S"], 4), rec["S"][0], c_E))
    print("z_s = %+.6f   z_v = %s" % (rec["z_s"], np.round(rec["z_v"], 6)))
    print("D_slow = %.6f   V = %s" % (rec["D"], np.round(rec["V"], 6)))
    print("E_raw[0] = %+.6f   E_final[0] = %+.6f" % (lg["E_raw"][0], lg["E_final"][0]))
    print("I_inhibit = %.6f   w_PFC_new = %.6f   action = %s"
          % (lg["I_inhibit"], lg["w_PFC"], lg["action"]))
    print("raw_emotion_impact[0] = %+.6f（对照：无 S 时约为 %.6f 量级的恒负危险偏置）"
          % (lg["raw_emotion_impact"][0], lg["raw_emotion_impact"][0] - c_E * rec["S"][0]))
    print("观察：E_raw[0] = %+.6f %s；I_inhibit = %.6f %s"
          % (lg["E_raw"][0],
             "为负（敌意语义正确：负效价注入压低了危险偏置）" if lg["E_raw"][0] < 0 else "转正（异常：敌意消息不应转正）",
             lg["I_inhibit"],
             "抑制启动" if lg["I_inhibit"] > 0 else "抑制未启动"))
    print()

    print("=== 7. 断言（数据完整性）===")
    # A. 轮数齐全 + 写回一致
    assert len(snapshots) == ROUNDS and len(logs_captured) == ROUNDS, "轮数不齐"
    for i in range(ROUNDS):
        assert abs(snapshots[i]["E_prev_0"] - float(logs_captured[i]["E_final"][0])) < 1e-12, (
            "第 %d 轮快照 E_prev[0] 与本轮 E_final[0] 不一致" % (i + 1))
        assert abs(snapshots[i]["w_PFC"] - float(logs_captured[i]["w_PFC"])) < 1e-12, (
            "第 %d 轮快照 w_PFC 与本轮 log w_PFC 不一致" % (i + 1))
    print("A. 15 轮齐全，写回一致（快照 vs log）: PASS")

    # B. L23 公式（含极性符号）：intended == 归档值 * sign(polarity)
    #    （012 号工作定版：强度×方向；废弃 (v*2-1)*sign）
    for i in range(ROUNDS):
        lg2 = logs_captured[i]
        if lg2["recalled_memory"] is not None:
            rm = lg2["recalled_memory"]
            v = float(rm["emotion_value"])
            pol = rm["emotion_polarity"]
            sign = 1.0 if pol == "positive" else (-1.0 if pol == "negative" else 0.0)
            expect = v * sign
            assert abs(float(lg2["recalled_delta_intended"]) - expect) < 1e-12, (
                "第 %d 轮 intended 与 v*sign 不符" % (i + 1))
    print("B. 召回轮 intended == 归档值*sign(polarity)（L23 公式 012 定版）: PASS")

    # C. L28 守卫：ReframedContext 每轮每维都在 [-1,1]
    for i in range(ROUNDS):
        rc = np.asarray(logs_captured[i]["ReframedContext"], dtype=float)
        assert float(np.max(np.abs(rc))) <= 1.0 + 1e-12, (
            "第 %d 轮 ReframedContext 超出 [-1,1]：max|rc|=%.6f" % (i + 1, float(np.max(np.abs(rc)))))
    print("C. ReframedContext 全部轮次全部维度都在 [-1,1]（L28 守卫自证）: PASS")

    # D. 极端输入段：慢通路记录存在且 z_s/z_v 为有限值
    assert len(slow_records) >= 1, "极端输入段未记录到慢通路内部量"
    assert np.isfinite(rec["z_s"]) and np.all(np.isfinite(rec["z_v"])), "z_s/z_v 非有限值"
    print("D. 极端输入段 z_s/z_v 记录完整且有限: PASS")
    print()

    print("探针完成。结论以运行输出的观察为准，断言仅保证数据完整性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
