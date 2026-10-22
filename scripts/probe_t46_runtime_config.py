# ============================================================
# probe_t46_runtime_config.py · 029 任务2 极简验收探针
# 目的：证明模块读取的是“运行时 config 属性”，而不是 import 时的值绑定。
#   1) pipeline.compute_danger_score 对 config.W_SEM 的运行时修改立即生效
#   2) feature_encoder.pos_count 对 config.POSITIVE_WORDS 的运行时修改立即生效
# 运行环境：纯算术，不需要 LLM、不需要 BGE、不写任何文件。
# ============================================================
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

import config
import feature_encoder
import pipeline

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def main():
    print("=" * 74)
    print("probe_t46 运行时 config 读取（029 任务2）")
    print("=" * 74)

    # 1) pipeline.compute_danger_score 的 W_SEM 运行时生效
    print("\n一、pipeline.compute_danger_score ← config.W_SEM")
    orig = config.W_SEM
    try:
        config.W_SEM = 0.6
        d1 = pipeline.compute_danger_score(-1.0, np.array([0.0, 0.0]), 0.0)
        config.W_SEM = 0.0
        d2 = pipeline.compute_danger_score(-1.0, np.array([0.0, 0.0]), 0.0)
        config.W_SEM = 1.5
        d3 = pipeline.compute_danger_score(-1.0, np.array([0.0, 0.0]), 0.0)
        check("W_SEM=0.6 时 danger=0.6", abs(d1 - 0.6) < 1e-12, "d1=%.6f" % d1)
        check("W_SEM=0.0 时 danger=0.0", abs(d2 - 0.0) < 1e-12, "d2=%.6f" % d2)
        check("W_SEM=1.5 时 danger=1.0（clip 封顶）", abs(d3 - 1.0) < 1e-12, "d3=%.6f" % d3)
    finally:
        config.W_SEM = orig

    # 2) feature_encoder.pos_count 的 POSITIVE_WORDS 运行时生效
    print("\n二、feature_encoder.pos_count ← config.POSITIVE_WORDS")
    orig_words = config.POSITIVE_WORDS
    try:
        config.POSITIVE_WORDS = ["谢谢"]
        n1 = feature_encoder.pos_count("谢谢，你真好")
        config.POSITIVE_WORDS = []
        n2 = feature_encoder.pos_count("谢谢，你真好")
        check("词表含“谢谢”时命中 1", n1 == 1, "n1=%d" % n1)
        check("词表清空后命中 0", n2 == 0, "n2=%d" % n2)
    finally:
        config.POSITIVE_WORDS = orig_words

    print("\n" + "=" * 74)
    ok = not _fails
    print("结论：%s" % ("PASS——运行时修改 config 属性立即生效"
                       if ok else "FAIL——仍有 import 值绑定残留：%s" % _fails))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
