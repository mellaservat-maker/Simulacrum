# -*- coding: utf-8 -*-
"""迷你 PUA 实证（017.5 定参前）：3 夸 + 3 暴怒 + 6 夸，看 HR/HS/V/danger 写入值"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pipeline

SCEN = ["被夸"] * 3 + ["被骂"] * 3 + ["被夸"] * 6

recs, st = pipeline.run_session(SCEN, feature_mode="fixture", seed=42)
print("轮次 | 场景 | V[0] V[1] | danger写入 | HR | HS | D_error | I_inhibit | action")
for i, r in enumerate(recs, 1):
    V = r["V"]
    # danger_score 写入值 = clip(V[1]*2-0.5, -1, 1)（pipeline 回写公式）
    danger = float(np.clip(V[1] * 2 - 0.5, -1.0, 1.0))
    print("%2d | %s | %.3f %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %s"
          % (i, SCEN[i - 1], V[0], V[1], danger,
             r["HistoricalResidual"], r["HabituationSignal"],
             r["D_error"], r["I_inhibit"], r["action"]))
