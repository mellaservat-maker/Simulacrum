# ============================================================
# session_store.py · [021.4] 会话持久化（JSON 落盘 + 加载恢复）
#
# 目标：关掉 API 窗口后重启，各对话的疲劳值（w_PFC）、记忆包、情绪惯性、
#   历史事件（HR 来源）都能恢复，切回旧对话时可无缝续聊。
#
# 序列化范围（按「体验无缝」取舍）：
#   L1 标量：w_PFC / HabituationSignal_prev / alpha_prev / D_error_prev /
#           round_counter / E_prev / B / M_prev
#   L2 记录：raw_log 全量轮记录（召回回捞的原文来源）、memory_buffer
#           未归档记录
#   L3 记忆：archiver 的 MemoryPackage 列表（含主题/峰值向量与
#           mean_valence）、memory_graph 的 EventNode 列表（含 embedding）
#
# 体积控制：所有 numpy 向量（1024/8/4 维）一律 base64(float32) 编码，
#   不用 JSON 浮点数组（后者文本膨胀约 2-3 倍）。
#
# 兼容性：文件带 version 字段；反序列化任何字段缺失/损坏都按默认值降级
#   并打印告警（不崩、不静默）。
# ============================================================

import base64
import json
import os
import time

import numpy as np

from schemas import EventNode, MemoryPackage
from state import reset_state

STORE_VERSION = 1
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sessions")


def _safe_id(sid):
    """只允许字母数字与 -_.，防路径穿越。"""
    keep = "".join(c for c in str(sid) if c.isalnum() or c in "-_.")
    return keep or "default"


def _path(sid, dir_path=None):
    d = dir_path or DEFAULT_DIR
    return os.path.join(d, "%s.json" % _safe_id(sid))


def _vec_to_b64(v):
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _b64_to_vec(s, dtype=np.float32):
    if not s:
        return None
    try:
        return np.frombuffer(base64.b64decode(s), dtype=dtype).copy()
    except Exception:
        return None


def _pkg_to_dict(p):
    return {
        "seq": int(getattr(p, "seq", 0)),
        "emotion_value": float(getattr(p, "emotion_value", 0.0)),
        "emotion_polarity": str(getattr(p, "emotion_polarity", "neutral")),
        "keywords": list(getattr(p, "keywords", []) or []),
        "raw_pointer": str(getattr(p, "raw_pointer", "") or ""),
        "mean_vec": _vec_to_b64(getattr(p, "mean_vec", None)),
        "peak_vec": _vec_to_b64(getattr(p, "peak_vec", None)),
        "mean_valence": (None if getattr(p, "mean_valence", None) is None
                         else float(p.mean_valence)),
        "n_rounds": int(getattr(p, "n_rounds", 0) or 0),
        "mean_danger": (None if getattr(p, "mean_danger", None) is None
                        else float(p.mean_danger)),
        "max_danger": (None if getattr(p, "max_danger", None) is None
                       else float(p.max_danger)),
        "narrative_cause": getattr(p, "narrative_cause", ""),
        "narrative_trigger": getattr(p, "narrative_trigger", ""),
        "narrative_consequence": getattr(p, "narrative_consequence", ""),
        "narrative_evaluation": getattr(p, "narrative_evaluation", ""),
        "narrative_emotion": getattr(p, "narrative_emotion", ""),
        "narrative_thought": getattr(p, "narrative_thought", ""),
    }


def _dict_to_pkg(d):
    return MemoryPackage(
        seq=int(d.get("seq", 0)),
        emotion_value=float(d.get("emotion_value", 0.0)),
        emotion_polarity=str(d.get("emotion_polarity", "neutral")),
        keywords=list(d.get("keywords", []) or []),
        raw_pointer=str(d.get("raw_pointer", "") or ""),
        mean_vec=_b64_to_vec(d.get("mean_vec")),
        peak_vec=_b64_to_vec(d.get("peak_vec")),
        mean_valence=(None if d.get("mean_valence") is None
                      else float(d["mean_valence"])),
        n_rounds=int(d.get("n_rounds", 0) or 0),
        mean_danger=(None if d.get("mean_danger") is None
                     else float(d["mean_danger"])),
        max_danger=(None if d.get("max_danger") is None
                    else float(d["max_danger"])),
        narrative_cause=str(d.get("narrative_cause", "")),
        narrative_trigger=str(d.get("narrative_trigger", "")),
        narrative_consequence=str(d.get("narrative_consequence", "")),
        narrative_evaluation=str(d.get("narrative_evaluation", "")),
        narrative_emotion=str(d.get("narrative_emotion", "")),
        narrative_thought=str(d.get("narrative_thought", "")),
    )


def _event_to_dict(e):
    emb = getattr(e, "event_embedding", None)
    return {
        "event_id": str(getattr(e, "event_id", "")),
        "timestamp": float(getattr(e, "timestamp", 0.0)),
        "raw_text": str(getattr(e, "raw_text", "")),
        "event_embedding": _vec_to_b64(emb),
        "nodes": {k: list(v) for k, v in (getattr(e, "nodes", {}) or {}).items()},
        "relationships": list(getattr(e, "relationships", []) or []),
        "importance": float(getattr(e, "importance", 0.0)),
        "emotion_weight": float(getattr(e, "emotion_weight", 0.0)),
        "unfinished": float(getattr(e, "unfinished", 0.0)),
        "persons": list(getattr(e, "persons", set()) or []),
        "tasks": list(getattr(e, "tasks", set()) or []),
        "danger_score": float(getattr(e, "danger_score", 0.0)),
        "event_severity": float(getattr(e, "event_severity", 0.0)),
    }


def _dict_to_event(d):
    return EventNode(
        event_id=str(d.get("event_id", "")),
        timestamp=float(d.get("timestamp", 0.0)),
        raw_text=str(d.get("raw_text", "")),
        event_embedding=_b64_to_vec(d.get("event_embedding")),
        nodes={k: list(v) for k, v in (d.get("nodes", {}) or {}).items()},
        relationships=list(d.get("relationships", []) or []),
        importance=float(d.get("importance", 0.0)),
        emotion_weight=float(d.get("emotion_weight", 0.0)),
        unfinished=float(d.get("unfinished", 0.0)),
        persons=set(d.get("persons", []) or []),
        tasks=set(d.get("tasks", []) or []),
        danger_score=float(d.get("danger_score", 0.0)),
        event_severity=float(d.get("event_severity", 0.0)),
    )


def dump_state(st):
    """把运行时 st 字典转成可 JSON 化的 dict。"""
    raw_log = {}
    try:
        for rid in st["raw_log"].all_ids():
            rec = st["raw_log"].get(rid)
            if rec is not None:
                raw_log[str(rid)] = rec
    except Exception:
        raw_log = {}
    return {
        "version": STORE_VERSION,
        "saved_at": time.time(),
        "round_counter": int(st.get("round_counter", 0)),
        "w_PFC": float(st.get("w_PFC", 0.0)),
        "HabituationSignal_prev": float(st.get("HabituationSignal_prev", 0.0)),
        "alpha_prev": float(st.get("alpha_prev", 0.0)),
        "D_error_prev": float(st.get("D_error_prev", 0.0)),
        # [031] 性格预设随会话持久化（冷切换）；旧文件缺字段 → standard
        "profile_id": str(st.get("profile_id", "standard")),
        # [039] 破防冷却计数；旧文件缺字段 → 0
        "express_cooldown": int(st.get("express_cooldown", 0)),
        "E_prev": _vec_to_b64(st.get("E_prev")),
        "B": _vec_to_b64(st.get("B")),
        "M_prev": _vec_to_b64(st.get("M_prev")),
        "buffer": list(st["memory_buffer"].get_all()),
        "raw_log": raw_log,
        "packages": [_pkg_to_dict(p) for p in st["archiver"].get_packages()],
        "events": [_event_to_dict(e) for e in st["memory_graph"].get_all()],
    }


def load_state(data, warn=print):
    """从 dump_state 的结果恢复 st（缺字段/损坏按默认降级）。"""
    st = reset_state(seed=42)
    try:
        st["round_counter"] = int(data.get("round_counter", 0))
        st["w_PFC"] = float(data.get("w_PFC", st["w_PFC"]))
        st["HabituationSignal_prev"] = float(
            data.get("HabituationSignal_prev", st["HabituationSignal_prev"]))
        st["alpha_prev"] = float(data.get("alpha_prev", st["alpha_prev"]))
        st["D_error_prev"] = float(data.get("D_error_prev", st["D_error_prev"]))
        # [031] 旧会话无 profile_id → standard（不因缺字段而拒绝载入）
        st["profile_id"] = str(data.get("profile_id", "standard") or "standard")
        # [039] 旧会话无 express_cooldown → 0
        st["express_cooldown"] = max(0, int(data.get("express_cooldown", 0) or 0))
        for key in ("E_prev", "B", "M_prev"):
            vec = _b64_to_vec(data.get(key))
            if vec is not None:
                st[key] = vec

        # buffer（未归档轮记录）
        for rec in (data.get("buffer") or []):
            st["memory_buffer"].append(rec)

        # raw_log（召回回捞的原文来源）
        for rid_s, rec in (data.get("raw_log") or {}).items():
            try:
                rec = dict(rec)
                rec["round_id"] = int(rec.get("round_id", rid_s))
                st["raw_log"].append(rec)
            except Exception:
                warn("[session_store] 跳过损坏的 raw_log 记录: %r" % (rid_s,))

        # archiver 的包（直接写私有列表：不侵入封版模块的内部实现）
        pkgs = []
        for d in (data.get("packages") or []):
            try:
                pkgs.append(_dict_to_pkg(d))
            except Exception as e:
                warn("[session_store] 跳过损坏的记忆包: %s" % e)
        st["archiver"]._packages = pkgs

        # memory_graph 的事件（HR 与海马体竞争的数据源）
        for d in (data.get("events") or []):
            try:
                st["memory_graph"].add(_dict_to_event(d))
            except Exception as e:
                warn("[session_store] 跳过损坏的事件: %s" % e)
    except Exception as e:
        warn("[session_store] 会话恢复异常，按新建状态继续：%s" % e)
    return st


def save(sid, st, dir_path=None, warn=print):
    """把会话写到磁盘（原子写：先 tmp 再 replace）。返回文件路径或 None。"""
    d = dir_path or DEFAULT_DIR
    try:
        os.makedirs(d, exist_ok=True)
        payload = dump_state(st)
        payload["session_id"] = _safe_id(sid)
        path = _path(sid, d)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        return path
    except Exception as e:
        warn("[session_store] 保存失败（会话未持久化，不影响本次对话）：%s" % e)
        return None


def load(sid, dir_path=None, warn=print):
    """尝试从磁盘恢复会话。返回 (st 或 None, meta)。"""
    path = _path(sid, dir_path)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("version", 0)) > STORE_VERSION:
            warn("[session_store] 会话文件版本更高（%s > %s），按新建处理"
                 % (data.get("version"), STORE_VERSION))
            return None, None
        st = load_state(data, warn=warn)
        return st, {"saved_at": data.get("saved_at"),
                    "round": data.get("round_counter", 0),
                    "packages": len(data.get("packages") or []),
                    "events": len(data.get("events") or [])}
    except Exception as e:
        warn("[session_store] 读取失败，按新建会话处理：%s" % e)
        return None, None


def delete(sid, dir_path=None):
    """删除会话文件。返回是否真的删除了。"""
    path = _path(sid, dir_path)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def list_files(dir_path=None):
    """列出磁盘上的会话文件（观测用）。"""
    d = dir_path or DEFAULT_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        if name.endswith(".json"):
            p = os.path.join(d, name)
            try:
                out.append({"session_id": name[:-5],
                            "size": os.path.getsize(p),
                            "mtime": os.path.getmtime(p)})
            except OSError:
                continue
    return sorted(out, key=lambda x: -x["mtime"])
