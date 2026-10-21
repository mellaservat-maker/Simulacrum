# ============================================================
# scripts/verify_scenarios.py · V4.2 · Step 1 数值链路验收
# 目的：在实现完整系统前，先用 config.py 的真实参数跑通数值链路。
# 不接 LLM，不接 memory_graph。
# 表格数字全部来自代码 dump，不允许手填。
# ============================================================

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import m
from state import reset_state
from fast_path import run_fast_path
from slow_path import run_slow_path
from emotion_releaser import run_releaser
from emotion import run_emotion
from intercept import run_intercept
from policy import policy_minimal
from tests.fixtures import get_fixture

# 【方案A裁决】一致性检查：config 与 probe 的 w_s 必须完全一致
from config import w_s as config_ws
print(f"config w_s[:3] = {np.round(config_ws[:3], 6)}")

LOGS_DIR = os.path.join(ROOT, "logs")

SCENARIOS = ["被骂", "被夸", "被质疑"]
LOG_NAMES = {
    "被骂": "verify_insult.json",
    "被夸": "verify_praise.json",
    "被质疑": "verify_doubt.json",
}
EXPECTED = {"被骂": "suppress", "被夸": "task_first", "被质疑": "reappraise"}

# Step 2 附加 WARNING 检查（非阻塞）
WARNING_SPEC = {
    "被骂": ("I_inhibit >= 0.7", lambda v: v >= 0.7),
    "被夸": ("I_inhibit < 0.3", lambda v: v < 0.3),
    "被质疑": ("0.4 < I_inhibit <= 0.7", lambda v: 0.4 < v <= 0.7),
}


def to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    return x


def run_scenario(scenario):
    fx = get_fixture(scenario)
    st = reset_state(seed=42)

    x = fx["x"]
    x_fast = fx["x_fast"]

    # M1 空历史分支（L8）
    M = np.zeros(m)
    ReframedContext = np.tanh(x)
    HabituationSignal = 0.0

    B = st["B"]
    D_fast, B = run_fast_path(x_fast, B)
    D_slow, V = run_slow_path(x, ReframedContext, M, B, st["w_PFC"], HabituationSignal, np.zeros(4))  # [012] 无语义信号，补全 0
    raw_emotion_impact, released_impact, release_rate = run_releaser(V, D_fast, HabituationSignal, 0.0)  # [012] S_valence=0
    E_raw = run_emotion(st["E_prev"], released_impact, B)
    neg_intensity, S_E, D_error, I_inhibit, E_final, w_PFC_new = run_intercept(E_raw, st["w_PFC"])
    action = policy_minimal(E_final, I_inhibit, fx["event_severity"])

    return {
        "scenario": scenario,
        "feature_mode": "fixture",
        "x": to_py(x),
        "x_fast": to_py(x_fast),
        "B": to_py(B),
        "D_fast": to_py(D_fast),
        "D_slow": to_py(D_slow),
        "V": to_py(V),
        "raw_emotion_impact": to_py(raw_emotion_impact),
        "released_impact": to_py(released_impact),
        "release_rate": to_py(release_rate),
        "E_raw": to_py(E_raw),
        "neg_intensity": to_py(neg_intensity),
        "S_E": to_py(S_E),
        "D_error": to_py(D_error),
        "I_inhibit": to_py(I_inhibit),
        "E_final": to_py(E_final),
        "w_PFC": to_py(w_PFC_new),
        "event_severity": to_py(fx["event_severity"]),
        "action": action,
        "expected_action": EXPECTED[scenario],
        "action_match": action == EXPECTED[scenario],
    }


def write_report(rows):
    lines = []
    lines.append("# verify_scenarios 验收报告（V4.2 / Step 1）")
    lines.append("")
    lines.append("参数来源：config.py（局部 RNG default_rng(42)）；特征来源：tests/fixtures.py 手工注入。")
    lines.append("")
    lines.append("## 一、三场景数值链路")
    lines.append("")
    lines.append("| 场景 | D_fast | D_slow | V0 | V1 | E_raw0 | S_E | D_error | I_inhibit | action | expected | match |")
    lines.append("|------|--------|--------|----|----|--------|-----|---------|-----------|--------|----------|-------|")
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['D_fast']:.4f} | {r['D_slow']:.4f} | "
            f"{r['V'][0]:.4f} | {r['V'][1]:.4f} | {r['E_raw'][0]:.4f} | "
            f"{r['S_E']:.4f} | {r['D_error']:.4f} | {r['I_inhibit']:.4f} | "
            f"{r['action']} | {r['expected_action']} | {'PASS' if r['action_match'] else 'FAIL'} |"
        )
    lines.append("")

    all_match = all(r["action_match"] for r in rows)
    lines.append("## 二、action 验收")
    lines.append("")
    lines.append(f"- 三场景 action 全部符合预期：**{'是' if all_match else '否'}**")
    lines.append("")

    lines.append("## 三、附加 WARNING 检查（非阻塞）")
    lines.append("")
    lines.append("| 场景 | 期望 | 实测 I_inhibit | 结果 |")
    lines.append("|------|------|----------------|------|")
    warn_lines = []
    for r in rows:
        desc, fn = WARNING_SPEC[r["scenario"]]
        ok = fn(r["I_inhibit"])
        lines.append(
            f"| {r['scenario']} | {desc} | {r['I_inhibit']:.4f} | {'OK' if ok else 'WARNING'} |"
        )
        if not ok:
            warn_lines.append(f"{r['scenario']}（{desc}，实测 {r['I_inhibit']:.4f}）")
    lines.append("")
    if warn_lines:
        lines.append("WARNING：" + "；".join(warn_lines))
    else:
        lines.append("WARNING：无")
    lines.append("")

    with open(os.path.join(LOGS_DIR, "verify_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return all_match


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        r = run_scenario(scenario)
        rows.append(r)
        with open(os.path.join(LOGS_DIR, LOG_NAMES[scenario]), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    all_match = write_report(rows)

    print("=" * 80)
    print("verify_scenarios（V4.2 / Step 1）")
    print("=" * 80)
    print(f"{'场景':<6}{'D_fast':>9}{'D_slow':>9}{'V0':>9}{'V1':>9}{'E_raw0':>9}"
          f"{'S_E':>9}{'D_err':>9}{'I_inh':>9}{'action':>12}{'match':>7}")
    for r in rows:
        print(
            f"{r['scenario']:<6}{r['D_fast']:>9.4f}{r['D_slow']:>9.4f}"
            f"{r['V'][0]:>9.4f}{r['V'][1]:>9.4f}{r['E_raw'][0]:>9.4f}"
            f"{r['S_E']:>9.4f}{r['D_error']:>9.4f}{r['I_inhibit']:>9.4f}"
            f"{r['action']:>12}{('PASS' if r['action_match'] else 'FAIL'):>7}"
        )
    print("-" * 80)
    print(f"三场景 action 全部符合预期：{all_match}")
    print(f"报告已写入：{os.path.join(LOGS_DIR, 'verify_report.md')}")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())