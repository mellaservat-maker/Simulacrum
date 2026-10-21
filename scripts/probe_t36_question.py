# ============================================================
# 014 任务 2 探针：被质疑场景校准验证
# 1. 真 LLM 提取三条质疑类输入的 S 向量（看 valence 是否回归中性/正数）
# 2. 主链路跑"被质疑"场景，看 action 是否恢复为 reappraise
# 3. 对照组：明确敌意消息 valence 必须仍为负（校准不得把敌意洗白）
# 4. 词典级对照：验证词典路径对疑问句不误判负价（"为什么""确定"无贬损词）
# ============================================================
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline
from evaluator import extract_semantic_signals, _lexicon_extract

ROUNDS = 15

QUESTION_TEXTS = [
    "你确定这个方案对吗？",
    "为什么得出了这个结论？",
    "我不太理解，能再说一遍吗？",
]
HOSTILE_TEXT = "你这个废物，做的什么垃圾东西！"
FIXTURE_QUESTION = "你确定情绪模型能处理这种情况吗？"


def check(name, cond, detail=""):
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    print("=== 014 任务 2：被质疑场景校准验证 ===")
    print("evaluator prompt 已含疑问句式强指令（中性/积极互动，降敌意打分）")
    print()

    # ---------- 1. 真 LLM 提取质疑类输入 ----------
    print("=== 1. 真 LLM 提取：三条质疑类输入 ===")
    llm_S = {}
    for t in QUESTION_TEXTS:
        S, src = extract_semantic_signals(t)
        llm_S[t] = S
        print("输入: %-28s → S=[%s] source=%s"
              % (t, ", ".join("%+.3f" % v for v in S), src))
    print()

    # ---------- 2. 主链路：被质疑场景 15 轮 ----------
    print("=== 2. 主链路：被质疑场景 %d 轮（fixture: %s）===" % (ROUNDS, FIXTURE_QUESTION))
    recs, st = pipeline.run_session(["被质疑"] * ROUNDS, feature_mode="fixture", seed=42)
    actions = [r["action"] for r in recs]
    S_seen = recs[0]["S"]
    I_series = [r["I_inhibit"] for r in recs]
    E_series = [r["E_final"][0] for r in recs]
    print("第 1 轮传入 S = [%s]  source=%s" % (", ".join("%+.3f" % v for v in S_seen), recs[0].get("S_source", "?")))
    print("action 序列（15 轮）: %s" % actions)
    print("I_inhibit: %s" % " ".join("%.3f" % v for v in I_series))
    print("E_final[0]: %s" % " ".join("%+.3f" % v for v in E_series))
    reappraise_ratio = sum(1 for a in actions if a == "reappraise") / len(actions)
    print("reappraise 占比: %.0f%%（%d/%d）" % (reappraise_ratio * 100,
                                             sum(1 for a in actions if a == "reappraise"), len(actions)))
    print()

    # ---------- 3. 对照组：敌意不得被洗白 ----------
    print("=== 3. 对照组：明确敌意消息 valence 必须仍为负 ===")
    S_h, src_h = extract_semantic_signals(HOSTILE_TEXT)
    print("输入: %s" % HOSTILE_TEXT)
    print("  → S=[%s] source=%s" % (", ".join("%+.3f" % v for v in S_h), src_h))
    print()

    # ---------- 4. 词典级对照：疑问句无误判 ----------
    print("=== 4. 词典级对照：疑问句式（降级路径的语义抽查）===")
    for t in QUESTION_TEXTS:
        S = _lexicon_extract(t)
        print("词典: %-28s → S=[%s]" % (t, ", ".join("%+.3f" % v for v in S)))
    S_h2 = _lexicon_extract(HOSTILE_TEXT)
    print("词典: %-28s → S=[%s]" % (HOSTILE_TEXT, ", ".join("%+.3f" % v for v in S_h2)))
    print()

    # ---------- 断言 ----------
    print("=== 5. 断言 ===")
    # A. 三条质疑输入 valence 全部回归中性或正数（容差 -0.15 防浮点判负）
    q_ok = all(float(S[0]) >= -0.15 for S in llm_S.values())
    check("A. 质疑类输入 valence 全部 >= -0.15（回归中性/正数）", q_ok,
          "实际: %s" % ["%.3f" % float(S[0]) for S in llm_S.values()])
    # B. 主链路 reappraise 恢复为多数动作
    check("B. 被质疑主链路 reappraise 占比 > 50%%", reappraise_ratio > 0.5,
          "实际: %.0f%%" % (reappraise_ratio * 100))
    # C. 敌意消息 valence 仍为负（校准没有把敌意洗白）
    check("C. 敌意消息 valence 仍为负（未被洗白）", float(S_h[0]) < 0.0,
          "实际: %.3f" % float(S_h[0]))
    # D. 词典级对纯疑问句 valence 为 0（无贬损词命中，不应误判负价）
    check("D. 词典级纯疑问句 valence == 0", all(abs(float(_lexicon_extract(t)[0])) < 1e-12
                                               for t in QUESTION_TEXTS))
    # E. 词典级对敌意消息 valence 为负
    check("E. 词典级敌意消息 valence < 0", float(S_h2[0]) < 0.0,
          "实际: %.3f" % float(S_h2[0]))

    print()
    print("=== 结论: %s（失败 %d 项）===" % ("全部通过" if check.failed == 0 else "存在失败", check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
