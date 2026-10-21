# -*- coding: utf-8 -*-
"""017.5 实现冒烟：编译检查 + 12 轮迷你 PUA（新读取侧/注入链是否跑通）"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import py_compile

for f in ("config.py", "schemas.py", "archiver.py", "decompressor.py",
          "intercept.py", "pipeline.py", "state.py", "embedder.py",
          "tests/fixtures.py"):
    py_compile.compile(os.path.join(ROOT, f), doraise=True)
    print("compile OK:", f)

import pipeline
from intercept import update_threat_mode, run_recall_intercept

# 迟滞机单元冒烟
a, raw = update_threat_mode(0.9, 0.0)
print("threat 0.9 -> alpha_new=%.3f raw=%.1f" % (a, raw))
a, raw = update_threat_mode(0.0, 1.0)
print("threat 0.0 -> alpha_new=%.3f raw=%.1f" % (a, raw))
a, raw = update_threat_mode(0.3, 1.0)
print("threat 0.3 (缓冲带) -> alpha_new=%.3f raw=%.1f" % (a, raw))
d, det = run_recall_intercept(0.8, -1.0, 0.16, 1.0, 0.5, None, 0.0)
print("PTSD ev0.8负 alpha1 -> delta=%.3f" % d)
d, det = run_recall_intercept(0.8, -1.0, 0.16, 0.0, 0.5, None, 0.0)
print("脱敏 ev0.8负 alpha0 H0.16 -> delta=%.3f (期望+0.3封顶)" % d)
d, det = run_recall_intercept(0.8, -1.0, 0.16, 0.0, 0.9, None, 0.0)
print("严重度门 severity0.9 -> delta=%.3f (期望0)" % d)

SCEN = ["被夸"] * 3 + ["被骂"] * 3 + ["被夸"] * 6
recs, st = pipeline.run_session(SCEN, feature_mode="fixture", seed=42)
print()
print("轮次 | 场景 | matched_sim | recall_src | ev | sign→极性 | threat | alpha | delta")
for i, r in enumerate(recs, 1):
    rm = r.get("recalled_memory") or {}
    sim = rm.get("matched_sim")
    src = rm.get("recall_source", "-")
    ev = rm.get("emotion_value")
    pol = rm.get("emotion_polarity", "-")
    print("%2d | %s | %s | %s | %s | %s | %.3f | %.3f | %+.3f"
          % (i, SCEN[i - 1],
             ("%.3f" % sim) if sim is not None else "-",
             src,
             ("%.2f" % ev) if ev is not None else "-",
             pol,
             r.get("threat_now", 0.0), r.get("alpha_mode", 0.0),
             r.get("recalled_delta_intended", 0.0)))
