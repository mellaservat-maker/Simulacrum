# ============================================================
# probe_t47_four_dim.py · 030 四维情绪有值化（第一批）验收探针
# 目的：证明 A_INIT 去共线化 + G_S 语义冲击 + b_E 新符号落地后，
#   被骂 / 被夸 / 被质疑在三到四个维度上呈现不同符号组合，而不是
#   同一条效价线的四个缩放。
# 口径：mock 后端 + fixture 三场景（确定性；不依赖真实 LLM 的措辞波动）。
# 断言：
#   被骂：E0 负、E1 正（高唤醒）、E2 负（低掌控）、E3 正（高焦虑）
#   被夸：E0 正、E1 正、E2 正（高掌控）、E3 接近 0（无焦虑）
#   附带：三场景 action 与 fixture 期望一致（E0 路径未被本批破坏）
# ============================================================
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import config
config.LLM_BACKEND = "mock"          # 确定性：S 走词典级，主模型返回定文本

import pipeline

SCENARIOS = [
    ("被骂", "suppress"),
    ("被夸", "task_first"),
    ("被质疑", "reappraise"),
]
ANX_NEAR_ZERO = 0.15                 # 被夸 E3 “接近 0”的判定阈值（工程口径）

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def show(tag, log):
    S = np.asarray(log["S"], dtype=float)
    Er = np.asarray(log["E_raw"], dtype=float)
    Ef = np.asarray(log["E_final"], dtype=float)
    V = np.asarray(log["V"], dtype=float)
    print("\n%s" % tag)
    print("  S        = [% .4f, % .4f, % .4f, % .4f]  source=%s"
          % (S[0], S[1], S[2], S[3], log.get("S_source")))
    print("  V        = [% .4f, % .4f]   D_fast=% .4f  release=%.4f"
          % (V[0], V[1], log["D_fast"], log["release_rate"]))
    print("  E_raw    = [% .4f, % .4f, % .4f, % .4f]"
          % (Er[0], Er[1], Er[2], Er[3]))
    print("  E_final  = [% .4f, % .4f, % .4f, % .4f]  action=%s"
          % (Ef[0], Ef[1], Ef[2], Ef[3], log["action"]))
    print("  S_E=%.4f  D_error=%.4f  I_inhibit=%.4f  danger=%s"
          % (log["S_E"], log["D_error"], log["I_inhibit"],
             ("n/a" if "danger" not in log else log["danger"])))
    return S, Er, Ef


def main():
    print("=" * 76)
    print("probe_t47 四维情绪有值化（030 第一批）")
    print("A_INIT=%s" % np.array2string(config.A_INIT, precision=2).replace("\n", " "))
    print("b_E=%s  w_e=%s  K_ANX=%.2f"
          % (np.array2string(config.b_E, precision=2), config.w_e, config.K_ANX))
    print("=" * 76)

    logs = {}
    for sc, _ in SCENARIOS:
        logs[sc] = pipeline.run_scenario(sc, feature_mode="fixture")
        show(sc, logs[sc])

    print("\n" + "-" * 76)
    print("断言")
    print("-" * 76)

    # 被骂：负效价 / 高唤醒 / 低掌控 / 高焦虑
    log = logs["被骂"]
    Er = np.asarray(log["E_raw"], dtype=float)
    check("被骂 E0 < 0（负效价）", Er[0] < 0.0, "E0=%.4f" % Er[0])
    check("被骂 E1 > 0（高唤醒）", Er[1] > 0.0, "E1=%.4f" % Er[1])
    check("被骂 E2 < 0（低掌控）", Er[2] < 0.0, "E2=%.4f" % Er[2])
    check("被骂 E3 > 0（高焦虑）", Er[3] > 0.0, "E3=%.4f" % Er[3])
    check("被骂 action=suppress（E0 路径未破坏）", log["action"] == "suppress",
          log["action"])

    # 被夸：正效价 / 高唤醒 / 高掌控 / 焦虑接近 0
    log = logs["被夸"]
    Er = np.asarray(log["E_raw"], dtype=float)
    check("被夸 E0 > 0（正效价）", Er[0] > 0.0, "E0=%.4f" % Er[0])
    check("被夸 E1 > 0（高唤醒）", Er[1] > 0.0, "E1=%.4f" % Er[1])
    check("被夸 E2 > 0（高掌控）", Er[2] > 0.0, "E2=%.4f" % Er[2])
    check("被夸 |E3| <= %.2f（无焦虑）" % ANX_NEAR_ZERO,
          abs(Er[3]) <= ANX_NEAR_ZERO, "E3=%.4f" % Er[3])
    check("被夸 action=task_first（E0 路径未破坏）",
          log["action"] == "task_first", log["action"])

    # 被质疑：中性效价，只做量级观察（不硬断言）
    log = logs["被质疑"]
    Er = np.asarray(log["E_raw"], dtype=float)
    check("被质疑 E0 接近 0", abs(Er[0]) < 0.35, "E0=%.4f" % Er[0])
    check("被质疑 E3 接近 0（无明确威胁）", abs(Er[3]) <= 0.20,
          "E3=%.4f" % Er[3])
    check("被质疑 action=reappraise（E0 路径未破坏）",
          log["action"] == "reappraise", log["action"])

    # 四维全部在定义域内
    for sc, _ in SCENARIOS:
        for key in ("E_raw", "E_final"):
            arr = np.asarray(logs[sc][key], dtype=float)
            check("%s %s 四维均在 [-1,1]" % (sc, key),
                  bool(np.all(arr >= -1.0) and np.all(arr <= 1.0)),
                  np.array2string(arr, precision=4))

    print("\n" + "=" * 76)
    ok = not _fails
    print("结论：%s" % ("PASS——四维在三种输入下呈现差异化符号组合"
                       if ok else "FAIL——未通过项：%s" % _fails))
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
