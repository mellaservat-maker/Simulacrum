# ============================================================
# test_state_payload.py · 050
# 规格：A1 —— 未计算字段必须是 null，不能再用 0.0 冒充
#   T050-1 全新会话（无 log、无 last_metrics）：w_PFC / I_inhibit / e / e0 /
#          action 全为 None；
#   T050-2 有本轮 log：返回真实计算值；
#   T050-3 无 log 但有 last_metrics（GET /state 的典型场景）：回落上一轮值；
#   T050-4 last_metrics 能随会话落盘并恢复（旧存档无此字段 → None）。
# ============================================================

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from state import reset_state  # noqa: E402


def test_fresh_session_fields_are_none():
    import server_api

    st = reset_state(seed=42)
    p = server_api._state_payload(st)
    assert p["round"] == 0
    for k in ("w_PFC", "I_inhibit", "e", "e0", "action"):
        assert p[k] is None, "%s 应为 None（未计算）" % k


def test_payload_with_log_uses_real_values():
    import server_api

    log = {
        "w_PFC": 0.31, "I_inhibit": 0.42, "E_final": [-0.2, 0.4, -0.1, 0.3],
        "action": "reappraise", "danger_now": 0.5,
    }
    p = server_api._state_payload(reset_state(seed=42), log)
    assert p["w_PFC"] == 0.31
    assert p["I_inhibit"] == 0.42
    assert p["e0"] == -0.2
    assert p["e"] == [-0.2, 0.4, -0.1, 0.3]
    assert p["action"] == "reappraise"


def test_payload_falls_back_to_last_metrics():
    import server_api

    st = reset_state(seed=42)
    st["last_metrics"] = {
        "w_PFC": 0.12, "I_inhibit": 0.38, "e": [-0.32, 0.52, -0.11, 0.2],
        "action": "reappraise", "danger_now": 0.44,
    }
    p = server_api._state_payload(st)
    assert p["w_PFC"] == 0.12
    assert p["I_inhibit"] == 0.38
    assert p["e0"] == -0.32
    assert p["action"] == "reappraise"


def test_last_metrics_roundtrip(tmp_path):
    import session_store

    st = reset_state(seed=42)
    st["last_metrics"] = {
        "w_PFC": 0.12, "I_inhibit": 0.38, "e": [-0.32, 0.52, -0.11, 0.2],
        "action": "reappraise", "danger_now": 0.44,
    }
    path = session_store.save("t050_metrics", st, str(tmp_path))
    assert path
    back, _meta = session_store.load("t050_metrics", str(tmp_path))
    assert back is not None
    m = back.get("last_metrics")
    assert m and abs(m["I_inhibit"] - 0.38) < 1e-9
    assert m["e"][0] == -0.32

    # 旧存档无 last_metrics → None（前端显示 "-"）
    st2 = reset_state(seed=42)
    st2.get("last_metrics")
    assert session_store._dump_metrics(None) is None
