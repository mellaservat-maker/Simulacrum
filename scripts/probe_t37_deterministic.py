# ============================================================
# 014 任务 3 探针：确定性权重 vs 随机权重对比
# 1. 打印确定性权重设计值（config 当前生效态）
# 2. 进程内复算旧随机冻结值（seed 42 按历史同序五连抽），monkeypatch config 属性
#    跑旧随机组；恢复确定性值跑新组——全程不切 config 文件，无忘切回风险
# 3. 三场景 ×15 轮对比：action 序列 / E_final[0] 轨迹 / I_inhibit 轨迹 / D_slow
# 4. 断言：两组均无 NaN、action 合法、值域合规；差异量化入报告
# ============================================================
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import pipeline

SCENARIOS = ["被骂", "被夸", "被质疑"]
ROUNDS = 15

WEIGHT_NAMES = ["w_s", "u_s", "W_v", "U_v", "v_s"]


def check(name, cond, detail=""):
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def old_random_weights():
    """按历史同序复算旧随机冻结值：w_s→u_s→W_v→U_v→v_s（seed 42）。

    013 起 e_s/f_v 已确定性化不再消耗序列，故此五连抽与历史快照口径一致。
    """
    rng = np.random.RandomState(config.RANDOM_SEED)
    return {
        "w_s": rng.randn(8) * 0.1,
        "u_s": rng.randn(1024) * 0.1,
        "W_v": rng.randn(2, 8) * 0.1,
        "U_v": rng.randn(2, 1024) * 0.1,
        "v_s": rng.randn(4) * 0.1,
    }


def run_group(label):
    """当前 config 属性状态下跑三场景 15 轮，返回 {scenario: recs}。"""
    out = {}
    for sc in SCENARIOS:
        recs, _ = pipeline.run_session([sc] * ROUNDS, feature_mode="fixture", seed=42)
        out[sc] = recs
    return out


def summarize(recs):
    return {
        "actions": [r["action"] for r in recs],
        "E": [float(r["E_final"][0]) for r in recs],
        "I": [float(r["I_inhibit"]) for r in recs],
        "D_slow": [float(r["D_slow"]) for r in recs],
    }


def main():
    print("=== 014 任务 3：确定性权重 vs 随机权重对比 ===")
    print("DETERMINISTIC_WEIGHTS = %s" % config.DETERMINISTIC_WEIGHTS)
    print()
    print("=== 1. 当前生效的确定性权重设计值 ===")
    np.set_printoptions(precision=4, suppress=True)
    print("w_s = %s" % config.w_s.tolist())
    print("W_v = %s" % config.W_v.tolist())
    print("v_s = %s" % config.v_s.tolist())
    print("u_s: 前 8 个 %s ...（1024 维均分 %.2f，sum=%.1f）"
          % (config.u_s[:8].tolist(), config.u_s[0], float(np.sum(config.u_s))))
    print("U_v: 行0 前 8 个 %s ...（2x1024 均分 %.2f）" % (config.U_v[0][:8].tolist(), config.U_v[0][0]))
    print("e_s = %s  f_v = %s（013 设计，本轮不变）" % (config.e_s.tolist(), config.f_v.tolist()))
    print()

    # ---------- 2. 两组运行 ----------
    print("=== 2. 确定性组运行（三场景 x %d 轮）===" % ROUNDS)
    det = run_group("deterministic")
    for sc in SCENARIOS:
        s = summarize(det[sc])
        print("%s action: %s" % (sc, s["actions"]))
    print()

    print("=== 3. 旧随机组运行（进程内 monkeypatch，同 seed）===")
    old = old_random_weights()
    saved = {n: getattr(config, n) for n in WEIGHT_NAMES}
    print("旧随机冻结值（复算）：w_s 前 4 = %s" % old["w_s"][:4].tolist())
    try:
        for n, v in old.items():
            setattr(config, n, v)
        rnd = run_group("random")
        for sc in SCENARIOS:
            s = summarize(rnd[sc])
            print("%s action: %s" % (sc, s["actions"]))
    finally:
        for n, v in saved.items():
            setattr(config, n, v)
    print("config 属性已恢复为确定性权重")
    print()

    # ---------- 4. 量化对比 ----------
    print("=== 4. 量化对比（确定性 vs 旧随机）===")
    print("场景   | 指标     | 旧随机(首→末)      | 确定性(首→末)      | 最大绝对差")
    print("-" * 78)
    max_diff_overall = 0.0
    for sc in SCENARIOS:
        d = summarize(det[sc])
        r = summarize(rnd[sc])
        for key in ("E", "I", "D_slow"):
            rv, dv = r[key], d[key]
            md = max(abs(a - b) for a, b in zip(rv, dv))
            max_diff_overall = max(max_diff_overall, md)
            print("%-6s | %-8s | %+7.4f → %+7.4f | %+7.4f → %+7.4f | %.4f"
                  % (sc, key, rv[0], rv[-1], dv[0], dv[-1], md))
        act_same = r["actions"] == d["actions"]
        print("%-6s | action   | 一致=%s（旧: %d 个 suppress / 新: %d 个 suppress）"
              % (sc, act_same,
                 sum(1 for a in r["actions"] if a == "suppress"),
                 sum(1 for a in d["actions"] if a == "suppress")))
    print("-" * 78)
    print("全局最大轨迹绝对差: %.6f" % max_diff_overall)
    print()

    # ---------- 5. 断言 ----------
    print("=== 5. 断言 ===")
    valid_actions = {"suppress", "reappraise", "task_first", "express"}
    all_finite = True
    all_valid = True
    all_range = True
    for group, g in (("随机", rnd), ("确定性", det)):
        for sc in SCENARIOS:
            s = summarize(g[sc])
            if not all(np.isfinite(v) for v in s["E"] + s["I"] + s["D_slow"]):
                all_finite = False
            if not all(a in valid_actions for a in s["actions"]):
                all_valid = False
            if not all(0.0 <= v <= 1.0 for v in s["I"] + s["D_slow"]):
                all_range = False
            if not all(-1.0 <= v <= 1.0 for v in s["E"]):
                all_range = False
    check("A. 两组全部数值有限（无 NaN/Inf）", all_finite)
    check("B. 两组 action 全部合法（%s）" % sorted(valid_actions), all_valid)
    check("C. 两组值域合规（I/D_slow∈[0,1]，E∈[-1,1]）", all_range)
    # D. 权重层确定性：固定 S + 确定性归档（旁路两层 LLM 温度随机源）后双跑
    #    应一致（容差 1e-5，吸收 embedder CPU 1e-6 级噪声）。
    #    随机源清单（014 实测，均不在 config 权重层，已入 014.1.txt 待裁决）：
    #      (1) evaluator LLM 温度：同输入 4 次 arousal 0.20/0.15/0.10/0.20（~5e-2）
    #      (2) archiver LLM 温度：归档 emotion_value 随机进召回（~1e-2）
    #      (3) embedder CPU 浮点噪声（~1e-6，013 双跑实测）
    #    本断言只验证权重/链路层本身无随机源。
    import evaluator as _ev_mod
    import archiver as _arch_mod
    _fixed_S = np.array([-0.90, 0.85, 0.80, 0.90])  # 固定"被骂"语义
    _orig_extract = _ev_mod.extract_semantic_signals
    _orig_backend = config.LLM_BACKEND
    pipeline.extract_semantic_signals = lambda text, llm=None: (_fixed_S.copy(), "probe_fixed")
    config.LLM_BACKEND = "mock"  # 归档走确定性 mock 分支（仅本进程）
    try:
        run_a = summarize(pipeline.run_session(["被骂"] * ROUNDS, feature_mode="fixture", seed=42)[0])
        run_b = summarize(pipeline.run_session(["被骂"] * ROUNDS, feature_mode="fixture", seed=42)[0])
    finally:
        pipeline.extract_semantic_signals = _orig_extract
        config.LLM_BACKEND = _orig_backend
    fixed_diff = max(abs(a - b) for a, b in zip(run_a["E"] + run_a["I"] + run_a["D_slow"],
                                                run_b["E"] + run_b["I"] + run_b["D_slow"]))
    check("D. 权重层确定性：固定 S + 确定性归档后双跑一致（容差 1e-5）", fixed_diff < 1e-5,
          "最大差 %.2e" % fixed_diff)
    # E. 新旧两组行为不同（证明权重切换真实生效，而非空操作）
    diff_real = max_diff_overall > 1e-9
    check("E. 新旧权重轨迹存在真实差异（切换生效）", diff_real,
          "最大差 %.6f" % max_diff_overall)

    # F. 已知限制量化（informational，不判 PASS/FAIL）：LLM 语义层随机源。
    #    015 温度分层后：evaluator 锁 0.0（本项实测同输入两次差 0.0000，
    #    锁温前为 0.10）；archiver 保留 0.3（记忆创作人味，其 emotion_value
    #    仅以 ~1e-2 进 E）；embedder CPU 噪声 ~1e-6。
    from evaluator import extract_semantic_signals as _ex
    S_a, _ = _ex("你确定这个方案对吗？")
    S_b, _ = _ex("你确定这个方案对吗？")
    temp_diff = float(np.max(np.abs(S_a - S_b)))
    print("[INFO] F. evaluator 锁温 0.0 实证：同输入两次提取 S 最大分量差 %.4f"
          "（锁温前 0.10；archiver 0.3 创作温度保留，embedder 噪声 ~1e-6）"
          % temp_diff)

    print()
    print("=== 结论: %s（失败 %d 项）===" % ("全部通过" if check.failed == 0 else "存在失败", check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
