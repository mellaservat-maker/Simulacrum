# ============================================================
# probe_t35_collapse.py · 013 探针：损耗型破防演示（012 任务 4 补完）
#
# 背景：012 把 w_PFC 从"防御力"反转为"意志力疲劳值"：
#   I_inhibit = clip(w_i*D_error - KAPPA*w_PFC + b_i, 0, 1)
#   w_PFC_new = clip(RHO*w_PFC + DELTA*I_inhibit, 0, W_MAX)
# probe_t34_crash 的 15 轮「被骂」因 S 占位把 E_raw[0] 推正 → D_error 恒 0
# → 抑制从未启动 → 疲劳无从累积 → 破防路径未被触发（已在 012.1 如实记录）。
# 本探针补上该路径的观测：
#   Part 1 链路级：三场景各 15 轮，S 强制置 zeros（012 前的对照行为），
#     观察 I_inhibit 是否启动、w_PFC 如何累积、是否发生破防。
#   Part 2 公式级：D_error 网格扫描（0~0.7），每格模拟 60 轮疲劳动态，
#     求崩塌轮数与终态，核对解析阈值 D_error* = KAPPA*W_MAX/w_i。
#
# 数值动力学（本探针实测确认，014 起 KAPPA=0.5，初版解析阈值猜想已被数据推翻）：
#   RHO=0.95 强持续、DELTA=0.1 弱累积 → 疲劳不动点 w* = DELTA*I*/(1-RHO) = 2.0*I*
#   联合不动点：I* = w_i*D_error / (1 + KAPPA*DELTA/(1-RHO))
#     KAPPA=0.2 时分母 1.4；KAPPA=0.5（014 裁决）时分母 2.0——侵蚀深度加倍。
#   → 任何 D_error>0 都有 I*>0：疲劳侵蚀是渐近的，永远磨不到零；
#     线性稳定性：RHO=0.95/DELTA=0.1 下特征方程 λ²-RHO·λ+KAPPA·DELTA=0 对任意
#     KAPPA 均 |λ|<1（K>2.26 起复振荡、K≈10 临界），不存在疲劳致全崩塌。
#     I*→0 只有连续加大 KAPPA 一条路（014 已到 0.5，是否再加码待所有者裁决）。
#
# 断言（数据完整性 + 公式恒等，不对结果下结论）：
#   A. 三场景各 15 轮齐全、写回一致（快照 vs log）
#   B. 每轮 I_inhibit / w_PFC_new 与 config 参数重算值一致（公式恒等自证）
#   C. S=zeros 对照确认：慢通路/释放器实际收到的 S 全为零（排除占位干扰）
#   D. 公式级扫描每格满足两条公式且 I∈[0,1]、w∈[0,W_MAX]，终态贴合不动点
#   E. 不动点恒等：各格终态 I* = min(1, w_i*D_error/(1+KAPPA·DELTA/(1-RHO)))
#      （014 起 KAPPA=0.5，阻尼分母=2.0）；D_error=0.70 全程 I≡1.0；
#      D_error=0.05 不崩塌（I*>0）——线性稳定性保证无疲劳致全崩塌
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from config import (w_i, b_i, KAPPA, W_MAX, RHO, DELTA, THETA_ERROR, S_DIM)
from state import reset_state
from slow_path import run_slow_path as _real_slow_path
from emotion_releaser import run_releaser as _real_releaser

SCENARIOS = ["被骂", "被夸", "被质疑"]
ROUNDS = 15

all_ok = True


def check(name, cond, detail=""):
    global all_ok
    all_ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ---- S=zeros 强制注入（对照组：还原 012 接入 S 之前的行为）----
s_seen = {"slow": [], "releaser": []}


def forced_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, S=None):
    injected = np.zeros(S_DIM)
    s_seen["slow"].append(injected.copy())
    return _real_slow_path(x, ReframedContext, M, B, w_PFC, HabituationSignal, injected)


def forced_releaser(V, D_fast, HabituationSignal, S_valence=0.0):
    s_seen["releaser"].append(0.0)
    return _real_releaser(V, D_fast, HabituationSignal, 0.0)


def run_control_session(scenario, rounds=ROUNDS):
    """以 S=zeros 对照跑 rounds 轮，返回 (快照列表, log列表)。"""
    snaps, logs = [], []
    orig_turn = pipeline.run_turn

    def observing(state, sc, feature_mode="fixture"):
        log = orig_turn(state, sc, feature_mode=feature_mode)
        snaps.append({
            "round": len(snaps) + 1,
            "E_prev_0": float(state["E_prev"][0]),
            "w_PFC": float(state["w_PFC"]),
            "B_1": float(state["B"][1]),
        })
        logs.append(log)
        return log

    pipeline.run_slow_path = forced_slow_path
    pipeline.run_releaser = forced_releaser
    pipeline.run_turn = observing
    try:
        pipeline.run_session([scenario] * rounds, feature_mode="fixture", seed=42)
    finally:
        pipeline.run_turn = orig_turn
        pipeline.run_slow_path = _real_slow_path
        pipeline.run_releaser = _real_releaser
    return snaps, logs


def main():
    print("=== 013 探针：损耗型破防演示（S=zeros 对照 + 公式级扫描）===")
    print("参数：w_i=%.1f b_i=%.1f KAPPA=%.2f RHO=%.2f DELTA=%.2f W_MAX=%.1f THETA_ERROR=%.2f"
          % (w_i, b_i, KAPPA, RHO, DELTA, W_MAX, THETA_ERROR))
    print("不动点预言：I* = w_i*D_error/(1+KAPPA*DELTA/(1-RHO)) = w_i*D_error/%.2f（封顶 1.0）；"
          "w* = DELTA*I*/(1-RHO) = %.1f*I*（当前参数下无疲劳致全崩塌，只有渐近侵蚀）"
          % (1 + KAPPA * DELTA / (1 - RHO), DELTA / (1 - RHO)))
    print()

    # ============ Part 1 链路级 ============
    print("=== Part 1 链路级：三场景各 15 轮，S 强制 zeros（012 前对照行为）===")
    control = {}
    for sc in SCENARIOS:
        snaps, logs = run_control_session(sc)
        control[sc] = (snaps, logs)
        first_break = None
        for i in range(ROUNDS):
            if float(logs[i]["I_inhibit"]) < 1.0 - 1e-9 and first_break is None:
                first_break = i + 1
        print()
        print("【%s】破防首轮：%s"
              % (sc, ("第 %d 轮" % first_break) if first_break else "全程未破防（I_inhibit 恒 1.0 或恒 0）"))
        print("轮次 | E_prev[0] | w_PFC     | S_E      | D_error  | I_inhibit | E_final[0] | action")
        for i in range(ROUNDS):
            s, lg = snaps[i], logs[i]
            print("%4d | %9.6f | %9.6f | %8.6f | %8.6f | %9.6f | %10.6f | %s"
                  % (s["round"], s["E_prev_0"], s["w_PFC"], lg["S_E"], lg["D_error"],
                     lg["I_inhibit"], lg["E_final"][0], lg["action"]))
    print()

    # ============ Part 2 公式级扫描 ============
    print("=== Part 2 公式级：D_error 网格扫描（120 轮疲活动态）===")
    print("不动点预言：I* = w_i*D_error/(1+KAPPA*DELTA/(1-RHO)) = w_i*D_error/%.2f，I 封顶 1.0"
          % (1 + KAPPA * DELTA / (1 - RHO)))
    print("D_error | 首轮I | 崩塌轮数 | 终w_PFC | 终I   | 预言I* | 判定")
    sweep = {}
    for de in [0.0, 0.05, 0.10, 0.13, 0.1333, 0.14, 0.20, 0.30, 0.50, 0.70]:
        w = 0.0
        collapse_round = None
        i_first = None
        for t in range(1, 121):
            i_t = float(np.clip(w_i * de - KAPPA * w + b_i, 0.0, 1.0))
            if i_first is None:
                i_first = i_t
            if i_t < 0.05 and collapse_round is None:
                collapse_round = t
            w = float(np.clip(RHO * w + DELTA * i_t, 0.0, W_MAX))
        i_last = float(np.clip(w_i * de - KAPPA * w + b_i, 0.0, 1.0))
        i_star = min(1.0, w_i * de / (1 + KAPPA * DELTA / (1 - RHO)))
        verdict = "永固（磨不穿）" if collapse_round is None else "第 %d 轮崩塌" % collapse_round
        sweep[de] = (i_first, collapse_round, w, i_last, i_star)
        print("%7.4f | %6.4f | %8s | %8.4f | %6.4f | %6.4f | %s"
              % (de, i_first, (str(collapse_round) if collapse_round else "-"), w, i_last, i_star, verdict))
    print()

    # ============ Part 3 断言 ============
    print("=== Part 3 断言 ===")
    # A. 三场景各 15 轮齐全 + 写回一致
    for sc in SCENARIOS:
        snaps, logs = control[sc]
        assert len(snaps) == ROUNDS and len(logs) == ROUNDS, f"{sc} 轮数不齐"
        for i in range(ROUNDS):
            assert abs(snaps[i]["E_prev_0"] - float(logs[i]["E_final"][0])) < 1e-12, (
                f"{sc} 第 {i+1} 轮 E_prev 与 E_final 不一致")
            assert abs(snaps[i]["w_PFC"] - float(logs[i]["w_PFC"])) < 1e-12, (
                f"{sc} 第 {i+1} 轮 w_PFC 写回不一致")
    check("A. 三场景各 15 轮齐全、写回一致", True)

    # B. 公式恒等：log 的 I_inhibit / w_PFC 与参数重算一致
    n_checked = 0
    for sc in SCENARIOS:
        snaps, logs = control[sc]
        for i in range(ROUNDS):
            lg = logs[i]
            de = float(lg["D_error"])
            wp_prev = 0.0 if i == 0 else float(logs[i - 1]["w_PFC"])
            ii_expect = float(np.clip(w_i * de - KAPPA * wp_prev + b_i, 0.0, 1.0))
            wp_expect = float(np.clip(RHO * wp_prev + DELTA * ii_expect, 0.0, W_MAX))
            assert abs(ii_expect - float(lg["I_inhibit"])) < 1e-12, (
                f"{sc} 第 {i+1} 轮 I_inhibit 公式恒等失败")
            assert abs(wp_expect - float(lg["w_PFC"])) < 1e-12, (
                f"{sc} 第 {i+1} 轮 w_PFC 公式恒等失败")
            n_checked += 1
    check("B. 45 轮 I_inhibit/w_PFC 与 config 参数重算完全一致（公式恒等）", True,
          f"核对 {n_checked} 轮")

    # C. S=zeros 对照确认：记录的是实际注入值，应全部为 0
    n_slow = len(s_seen["slow"])
    n_rel = len(s_seen["releaser"])
    zeros_ok = all(np.allclose(s, 0.0) for s in s_seen["slow"]) and all(v == 0.0 for v in s_seen["releaser"])
    check("C. 对照组实际注入 S 全为零（慢通路 %d 次 + 释放器 %d 次 = %d，符合 3 场景×15 轮）"
          % (n_slow, n_rel, n_slow + n_rel),
          n_slow == 45 and n_rel == 45 and zeros_ok)

    # D. 扫描每格公式自洽 + 终态贴合不动点
    for de, (i_first, collapse_round, w_end, i_last, i_star) in sweep.items():
        assert 0.0 <= i_first <= 1.0 and 0.0 <= i_last <= 1.0, f"D_error={de} I 越界"
        assert 0.0 <= w_end <= W_MAX + 1e-12, f"D_error={de} w 越界"
        assert abs(i_last - i_star) < 0.01, (
            f"D_error={de} 终态 I={i_last:.4f} 偏离不动点预言 {i_star:.4f}")
        # 终态 w 应等于 DELTA*I*/(1-RHO)（I*=0 时 w 已衰减尽，为 0）
        w_star = DELTA * i_star / (1 - RHO) if i_star > 0 else 0.0
        assert abs(w_end - w_star) < 0.05, (
            f"D_error={de} 终态 w={w_end:.4f} 偏离不动点 {w_star:.4f}")
    check("D. 扫描 10 格全部满足公式、值域与不动点预言", True)

    # E. 阈值语义交叉核对（修正版：无疲劳致全崩塌，只有渐近侵蚀）
    damp = 1 + KAPPA * DELTA / (1 - RHO)
    assert sweep[0.70][1] is None and abs(sweep[0.70][0] - 1.0) < 1e-9, "D_error=0.70 应全程满抑制"
    assert sweep[0.05][1] is None, "D_error=0.05 不应崩塌（不动点 I*>0）"
    assert abs(sweep[0.05][4] - w_i * 0.05 / damp) < 1e-9, "D_error=0.05 不动点预言值"
    check("E. 不动点语义核对：D_error=0.70 全程 I≡1.0；D_error=0.05 不崩塌（I*≈%.3f，阻尼分母=%.2f）"
          % (sweep[0.05][4], damp), True)
    print()
    print("探针结论：%s" % ("全部 PASS" if all_ok else "存在 FAIL，需检查"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
