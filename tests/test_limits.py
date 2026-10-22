# ============================================================
# test_limits.py · V4.2
# 规格：第七部分 test_limits / 第九部分验收
# ============================================================

import math
import os
import json

import numpy as np

from config import c, W_MAX, W_DANGER, THETA_ERROR

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def all_run_logs():
    logs = []
    for name in ["run_insult.jsonl", "run_praise.jsonl", "run_doubt.jsonl"]:
        p = os.path.join(LOGS_DIR, name)
        if not os.path.exists(p):
            continue
        for rec in _read_jsonl(p):
            logs.append(rec)
    return logs


def assert_no_nan_inf(x):
    if isinstance(x, dict):
        for v in x.values():
            assert_no_nan_inf(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            assert_no_nan_inf(v)
    elif isinstance(x, (int, float)):
        assert not math.isnan(x)
        assert not math.isinf(x)


def test_clip_bounds():
    logs = all_run_logs()
    assert len(logs) >= 3, "应至少有 3 条 run 日志"
    for log in logs:
        for i in range(c):
            assert -1.0 <= log["E_raw"][i] <= 1.0
            assert -1.0 <= log["E_final"][i] <= 1.0
        assert 0.0 <= log["I_inhibit"] <= 1.0
        assert 0.0 <= log["release_rate"] <= 0.9
        assert 0.1 <= log["B_xg"] <= 1.0
        assert 0.0 <= log["S_E"] <= 1.0
        # [033] 当前公式下 D_error ∈ [0, (1-THETA_ERROR)+W_DANGER*(1-DANGER_NEUTRAL)]。
        # 但 logs/ 里存有 032 时期（尚无高门）的历史记录，可为负、上界也更高；
        # 这里按两代公式的并集校验：[-W_DANGER, (1-THETA_ERROR)+W_DANGER]。
        assert -W_DANGER <= log["D_error"] <= (1.0 - THETA_ERROR) + W_DANGER
        assert 0.0 <= log["w_PFC"] <= W_MAX
        assert_no_nan_inf(log)