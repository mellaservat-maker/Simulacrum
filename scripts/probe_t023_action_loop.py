# ============================================================
# probe_t023_action_loop.py · 023 任务1 单场景 Action 闭环验证
# 病灶（清单问题1）：policy_minimal 算出 suppress/express 等 action，
#   但 action 只落 log 不进 Prompt，LLM 完全没听话。
# 本探针分两部分：
#   A. 确定性单测（不调 API）：假 LLM 返回 90 字长文，验证
#      suppress 暴走护栏（>30 字强行截断加……）真的生效；
#   B. 真链路四场景（调 DeepSeek）：被骂(suppress) / 被夸(task_first) /
#      被质疑(reappraise) / 暴怒(ask_clarify)，打印 action、输出字数、
#      输出全文与护栏触发情况，验证不同 action 下 LLM 输出真的变了。
# ============================================================
import os
import sys
import io
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pipeline
from pipeline import run_scenario, run_turn
from state import reset_state

EXPECTED_ACTION = {"被骂": "suppress", "被夸": "task_first",
                   "被质疑": "reappraise", "暴怒": "ask_clarify"}


class _FakeLLM:
    """确定性假 LLM：无视 Prompt，永远返回 90 字长文。"""

    def generate(self, prompt):
        return "这是一段故意超长的回复内容用于测试护栏截断逻辑是否生效" * 3


def part_a_guard_unit_test():
    print("=" * 70)
    print("A. suppress 护栏确定性单测（假 LLM 返回 90 字，不调 API）")
    print("=" * 70)
    orig = pipeline.get_llm
    pipeline.get_llm = lambda: _FakeLLM()
    try:
        st = reset_state(seed=42)
        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            log = run_turn(st, "被骂", feature_mode="fixture")
    finally:
        pipeline.get_llm = orig

    action = log["action"]
    out = log["llm_output"]
    ok_action = action == "suppress"
    ok_guard = len(out) == 32 and out.endswith("……")
    ok_warn = "护栏触发" in err_buf.getvalue()
    print(f"被骂 action={action}（预期 suppress）{'PASS' if ok_action else 'FAIL'}")
    print(f"输入 90 字 → 输出 {len(out)} 字，末位省略号：{'PASS' if ok_guard else 'FAIL'}")
    print(f"护栏告警已打印：{'PASS' if ok_warn else 'FAIL'}")
    print(f"截断结果：{out}")
    return ok_action and ok_guard and ok_warn


def part_b_real_llm():
    print()
    print("=" * 70)
    print("B. 真链路四场景（调 DeepSeek，观察不同 action 下输出差异）")
    print("=" * 70)
    results = {}
    for sc in ["被骂", "被夸", "被质疑", "暴怒"]:
        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            log = run_scenario(sc, feature_mode="fixture")
        action = log["action"]
        out = log["llm_output"]
        guard = "护栏触发" in err_buf.getvalue()
        exp = EXPECTED_ACTION[sc]
        match = "MATCH" if action == exp else f"MISMATCH(预期{exp})"
        results[sc] = (action, len(out), guard)
        print(f"--- {sc} ---")
        print(f"action={action} {match} | 输出 {len(out)} 字 | 护栏触发={guard}")
        print(f"输出全文：{out}")
        print()
    return results


if __name__ == "__main__":
    a_ok = part_a_guard_unit_test()
    print()
    results = part_b_real_llm()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"A 护栏单测：{'PASS' if a_ok else 'FAIL'}")
    for sc, (act, ln, guard) in results.items():
        print(f"  {sc:<4} action={act:<12} 输出长度={ln:<4} 护栏={guard}")
    print()
    print("判读要点：suppress（被骂）应显著短于 task_first（被夸）；")
    print("reappraise（被质疑）应含重新思考措辞；ask_clarify（暴怒）应含提问。")
