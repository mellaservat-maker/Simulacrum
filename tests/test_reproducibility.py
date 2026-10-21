# ============================================================
# test_reproducibility.py · V4.2
# 规格：第七部分 test_reproducibility
# ============================================================

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_seed_frozen():
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    r1 = subprocess.check_output(
        [sys.executable, "-c", "from config import w_s; print(w_s.tolist())"],
        cwd=ROOT, env=env,
    ).decode().strip()
    r2 = subprocess.check_output(
        [sys.executable, "-c", "from config import w_s; print(w_s.tolist())"],
        cwd=ROOT, env=env,
    ).decode().strip()
    assert r1 == r2