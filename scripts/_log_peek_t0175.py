# -*- coding: utf-8 -*-
"""查看被夸/被骂日志的 V 与 danger_score 写入值（017.5 定参前核实）"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("run_praise.jsonl", "run_insult.jsonl"):
    path = os.path.join(ROOT, "logs", name)
    if not os.path.exists(path):
        print(name, "不存在")
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    print("=== %s（%d 轮）===" % (name, len(rows)))
    for r in rows[-2:]:
        print("  V =", r.get("V"), "action =", r.get("action"),
              "D_slow =", r.get("D_slow"), "HR =", r.get("HistoricalResidual"),
              "HS =", r.get("HabituationSignal"))
