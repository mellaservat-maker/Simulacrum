# ============================================================
# test_implant_memory.py · 047
# 规格：上帝之手（scripts/implant_memory.py）的落盘安全与检索可见性
#   T047-1 植入后存档可被 session_store 正常回读，包与事件都在；
#   T047-2 改前必有备份，备份本身是合法 JSON；
#   T047-3 dry-run 不落盘；
#   T047-4 植入一段“被骂”记忆后，用一句无关紧要的话走 memory_gateway，
#          能被包级检索命中，且情绪极性与伪造值一致（negative / 0.85）。
# ============================================================

import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(BASE, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

STORY = "上周你因为写错代码被我狠狠骂了一顿，你当时特别委屈。"


def _implant(sid, dir_path, story=STORY, valence=-0.8, intensity=0.85,
             danger=0.55, **kw):
    import implant_memory
    return implant_memory.implant(
        sid, story, valence=valence, intensity=intensity,
        danger_score=danger, importance=0.8, dir_path=str(dir_path),
        verbose=False, **kw)


def test_implant_roundtrip(tmp_path):
    """T047-1：植入后存档合法、回读可见，包与事件各一条。"""
    import session_store

    res = _implant("t047_roundtrip", tmp_path)
    assert res["seq"] == 1
    assert res["existed"] is False
    assert res["backup"] is None          # 新建会话无原文件，不产生备份
    assert os.path.exists(res["path"])

    with open(res["path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data.get("packages"), list)
    assert len(data["packages"]) == 1
    assert data["packages"][0]["seq"] == 1
    assert data["packages"][0]["emotion_polarity"] == "negative"
    assert len(data.get("events") or []) == 1

    st, meta = session_store.load("t047_roundtrip", str(tmp_path))
    assert st is not None
    pkgs = st["archiver"].get_packages()
    assert len(pkgs) == 1 and pkgs[0].seq == 1
    assert pkgs[0].mean_vec is not None and len(pkgs[0].mean_vec) == 1024
    assert len(st["memory_graph"].get_all()) == 1


def test_implant_twice_creates_backup(tmp_path):
    """T047-2：第二次植入前必备份，备份可独立解析；seq 递增。"""
    first = _implant("t047_backup", tmp_path)
    second = _implant("t047_backup", tmp_path, story="你昨天夸我代码写得干净。",
                      valence=0.7, intensity=0.6, danger=0.05)

    assert second["seq"] == first["seq"] + 1
    assert second["existed"] is True
    assert second["backup"] and os.path.exists(second["backup"])
    with open(second["backup"], "r", encoding="utf-8") as f:
        backup_data = json.load(f)
    assert len(backup_data["packages"]) == 1          # 备份是植入前的状态

    with open(second["path"], "r", encoding="utf-8") as f:
        now_data = json.load(f)
    assert len(now_data["packages"]) == 2
    assert second["packages_after"] == 2


def test_implant_dry_run_does_not_write(tmp_path):
    """T047-3：dry-run 只计算，不生成文件。"""
    import session_store

    res = _implant("t047_dry", tmp_path, dry_run=True)
    assert res["dry_run"] is True
    assert res["path"] is None
    assert not os.path.exists(session_store._path("t047_dry", str(tmp_path)))


def test_implanted_memory_is_recalled(tmp_path):
    """T047-4：植入的"被骂"记忆在**相关**输入下能被检索到。

    走与线上一致的 memory_gateway → pipeline._recall_match 路径；
    单包只有植入项，因此命中即证明检索可见。now_valence 取 0.0（中性），
    极性过滤不介入，避免把包级软惩罚混入本用例的判断。

    [053] 规格修正：原用例用一句完全无关的话（"今天天气不错"）断言必须召回，
    等于把 TAU_FLOOR=0.15 这个假参数当成了规格——实测该句与植入故事余弦
    只有 0.256，任何真实的"低相关不召回"拦截都不该放过它。现拆成两条：
      相关输入 → 必须召回（余弦 0.812 / 0.706，远高于 TAU_FLOOR）；
      无关输入 → 必须不召回（余弦 0.256，低于 TAU_FLOOR=0.45）。
    """
    import config
    import session_store
    from embedder import get_embedder
    from memory_gateway import recall as memory_recall
    from schemas import EventNode

    res = _implant("t047_recall", tmp_path)
    st, _meta = session_store.load("t047_recall", str(tmp_path))
    assert st is not None

    emb = get_embedder("bge")

    def _recall(query):
        q_vec = np.asarray(emb.encode(query), dtype=np.float32)
        cur_event = EventNode(
            event_id="cur-1", timestamp=time.time(), raw_text=query,
            event_embedding=q_vec,
        )
        return memory_recall(query, cur_event, st["memory_graph"],
                             np.zeros(8), 0.0, time.time(), st,
                             int(st.get("round_counter", 0)), now_valence=0.0)

    # ---- 相关输入：必须召回，且极性与强度沿用植入值 ----
    out = _recall("你上次写错代码被我骂的时候，是不是很委屈？")
    assert out.recalled is not None, "相关输入未召回植入记忆"
    assert out.recalled.get("matched_pkg_seq") == res["seq"]
    assert out.recalled.get("recall_source") == "package_level"
    assert out.recalled.get("emotion_polarity") == "negative"
    assert abs(float(out.recalled.get("emotion_value")) - 0.85) < 1e-6
    assert float(out.recalled.get("matched_sim")) > config.TAU_FLOOR

    # ---- 无关输入：必须不召回（053 起 TAU_FLOOR 真正闭合）----
    out2 = _recall("今天天气不错，随便聊聊吧。")
    assert out2.recalled is None, (
        "无关输入召回了植入记忆（matched_sim=%.3f，TAU_FLOOR=%.2f）——"
        "说明低相关拦截没有生效"
        % (float((out2.recalled or {}).get("matched_sim", -1.0)),
           float(config.TAU_FLOOR)))
