import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pipeline import run_scenario
from state import reset_state
from tests.fixtures import get_fixture
from event_parser import parse as event_parse
from feature_encoder import encode as feature_encode
from xinguang import update_xinguang_budget
from hippocampal_evaluator import evaluate as hippocampal_evaluate

scenarios = ["被骂", "被夸", "被质疑"]

# 用 run_scenario 拿 log 里的 x, x_fast, D_slow, V
logs = {}
for sc in scenarios:
    logs[sc] = run_scenario(sc, feature_mode="fixture")

# 重新跑前半段拿 ReframedContext 和 M
extras = {}
for sc in scenarios:
    st = reset_state(seed=42)
    fx = get_fixture(sc)
    now = time.time()
    event = event_parse(
        fx["input"],
        event_severity=fx["event_severity"],
        importance=fx["importance"],
        emotion_weight=fx["emotion_weight"],
        unfinished=fx["unfinished"],
        danger_score=fx["danger_score"],
    )
    x, x_fast = feature_encode(event, st["M_prev"], st["HabituationSignal_prev"], st["B"], st["memory_graph"])
    B = st["B"]
    B_xg = update_xinguang_budget(B[0], B[1])
    hippo = hippocampal_evaluate(event, st["memory_graph"], x, B_xg, now)
    extras[sc] = {
        "ReframedContext": np.array(hippo["ReframedContext"]),
        "M": np.array(hippo["M"]),
    }

print("=== 三场景各信号完整值 ===\n")

for sig in ["x", "x_fast", "ReframedContext", "M", "D_slow", "V"]:
    print(f"--- {sig} ---")
    for sc in scenarios:
        if sig in ("ReframedContext", "M"):
            val = extras[sc][sig]
        else:
            val = np.array(logs[sc].get(sig))
        print(f"  {sc}: {val}")
    print()

print("\n=== 汇总表 ===\n")
print(f"{'信号':<25} | {'被骂':<20} | {'被夸':<20} | {'被质疑':<20} | 能否区分")
print("-" * 100)

def is_distinct(vals):
    a, b, c = vals[0], vals[1], vals[2]
    return (abs(a - b) > 1e-9) and (abs(a - c) > 1e-9) and (abs(b - c) > 1e-9)

# x 8维
x_arrs = {sc: np.array(logs[sc]["x"]) for sc in scenarios}
for i in range(8):
    vals = [x_arrs[sc][i] for sc in scenarios]
    d = "是" if is_distinct(vals) else "否"
    print(f"x[{i}]".ljust(25) + f" | {vals[0]:<20.6f} | {vals[1]:<20.6f} | {vals[2]:<20.6f} | {d}")

# x_fast 4维
xf_arrs = {sc: np.array(logs[sc]["x_fast"]) for sc in scenarios}
for i in range(4):
    vals = [xf_arrs[sc][i] for sc in scenarios]
    d = "是" if is_distinct(vals) else "否"
    print(f"x_fast[{i}]".ljust(25) + f" | {vals[0]:<20.6f} | {vals[1]:<20.6f} | {vals[2]:<20.6f} | {d}")

# ReframedContext 8维
rc_arrs = {sc: extras[sc]["ReframedContext"] for sc in scenarios}
for i in range(8):
    vals = [rc_arrs[sc][i] for sc in scenarios]
    d = "是" if is_distinct(vals) else "否"
    print(f"ReframedContext[{i}]".ljust(25) + f" | {vals[0]:<20.6f} | {vals[1]:<20.6f} | {vals[2]:<20.6f} | {d}")

# M 8维
m_arrs = {sc: extras[sc]["M"] for sc in scenarios}
for i in range(8):
    vals = [m_arrs[sc][i] for sc in scenarios]
    d = "是" if is_distinct(vals) else "否"
    print(f"M[{i}]".ljust(25) + f" | {vals[0]:<20.6f} | {vals[1]:<20.6f} | {vals[2]:<20.6f} | {d}")

# D_slow 标量
ds_vals = [logs[sc]["D_slow"] for sc in scenarios]
d = "是" if is_distinct(ds_vals) else "否"
print("D_slow".ljust(25) + f" | {ds_vals[0]:<20.6f} | {ds_vals[1]:<20.6f} | {ds_vals[2]:<20.6f} | {d}")

# V 2维
v_arrs = {sc: np.array(logs[sc]["V"]) for sc in scenarios}
for i in range(2):
    vals = [v_arrs[sc][i] for sc in scenarios]
    d = "是" if is_distinct(vals) else "否"
    print(f"V[{i}]".ljust(25) + f" | {vals[0]:<20.6f} | {vals[1]:<20.6f} | {vals[2]:<20.6f} | {d}")

print()
print("探针完成")
