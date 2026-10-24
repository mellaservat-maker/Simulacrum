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
import math
import os
import time

import numpy as np

import config
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
        # [056] 稳定实体 ID 列表；旧存档缺字段 → 空列表
        "entity_ids": list(getattr(e, "entity_ids", []) or []),
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
        entity_ids=list(d.get("entity_ids", []) or []),
        danger_score=float(d.get("danger_score", 0.0)),
        event_severity=float(d.get("event_severity", 0.0)),
    )


def _dump_relationships(rel):
    """[056] 关系账本白名单化：只保留 {entity_id: {trust,intimacy,interactions,broken}}。

    旧存档无此字段 → None（load 时按 reset_state 默认账本处理）。
    单条记录字段缺失/损坏 → 跳过该条，不让一个坏实体拖垮整个会话。
    """
    if not isinstance(rel, dict):
        return None
    out = {}
    for eid, v in rel.items():
        if not isinstance(eid, str) or not isinstance(v, dict):
            continue
        try:
            out[eid] = {
                "trust": float(v.get("trust", 0.5)),
                "intimacy": float(v.get("intimacy", 0.5)),
                "interactions": int(v.get("interactions", 0)),
                "broken": bool(v.get("broken", False)),
            }
        except (TypeError, ValueError):
            continue
    return out


def _load_relationships(data, st):
    """[056] 恢复关系账本：与 reset_state 默认账本深合并（缺字段补默认）。

    深合并而非整体替换：defaults 里的 user_001/ai_self 保证旧存档载入后
    账本结构完整；存档里多出来的实体（boss_01 等）原样并入。
    broken 不直接信任存档值——load 时按 trust 重算（见下），防旧版本
    写入的 broken 与 trust 不自洽；[058 任务2] 起重算改为施密特触发器
    口径：迟滞带 [BREAK, RELEASE) 内保持存档值（与 update_relationship
    同逻辑，带内不翻转）。
    """
    defaults = st.get("relationships") or {}
    merged = {eid: dict(v) for eid, v in defaults.items()}
    for eid, v in (data.get("relationships") or {}).items():
        if not isinstance(eid, str) or not isinstance(v, dict):
            continue
        cur = dict(merged.get(eid) or {"trust": 0.5, "intimacy": 0.5,
                                       "interactions": 0, "broken": False})
        try:
            cur["trust"] = float(v.get("trust", cur["trust"]))
            cur["intimacy"] = float(v.get("intimacy", cur["intimacy"]))
            cur["interactions"] = int(v.get("interactions", cur["interactions"]))
        except (TypeError, ValueError):
            continue
        cur["trust"] = min(1.0, max(0.0, cur["trust"]))
        cur["intimacy"] = min(1.0, max(0.0, cur["intimacy"]))
        cur["interactions"] = max(0, cur["interactions"])
        # [058 任务2] broken 按施密特触发器重算（非布尔存档值一律按
        #   False 处理，防 "false" 字符串被 bool() 吃成 True）：
        #   trust<BREAK → True；trust>=RELEASE → False；
        #   迟滞带 [BREAK, RELEASE) 内保持存档值——与轮末
        #   update_relationship 同口径，band 内 load 不会翻转状态。
        _saved_broken = v.get("broken", False)
        _saved_broken = (_saved_broken if isinstance(_saved_broken, bool)
                         else False)
        if cur["trust"] < config.REL_TRUST_BREAK:
            cur["broken"] = True
        elif cur["trust"] >= config.REL_TRUST_RELEASE:
            cur["broken"] = False
        else:
            cur["broken"] = _saved_broken
        merged[eid] = cur
    st["relationships"] = merged


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
        # [045] 聊天记录（用户输入 + AI 回复）随会话落盘，供跨浏览器/跨设备
        #   拉取同一份文字历史。旧文件无此字段 → 空列表。
        "messages": dump_messages(st.get("messages")),
        # [050 A1] 上一轮真实计算的面板数值；旧文件无此字段 → null，
        #   /state 据此区分「未计算」与「计算结果为 0」。
        "last_metrics": _dump_metrics(st.get("last_metrics")),
        # [056 任务1] 会话级关系账本；旧文件无此字段 → null
        "relationships": _dump_relationships(st.get("relationships")),
    }


def dump_messages(msgs):
    """只保留渲染必要字段，避免把临时对象写进存档。"""
    out = []
    for m in (msgs or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")) or "assistant"
        content = str(m.get("content", ""))
        try:
            ts = float(m.get("ts", 0.0))
        except Exception:
            ts = 0.0
        out.append({"role": role, "content": content, "ts": ts})
    return out


def _dump_metrics(m):
    """面板数值白名单化：只保留可 JSON 化的标量与四维情绪。"""
    if not isinstance(m, dict):
        return None
    out = {}
    for k in ("w_PFC", "I_inhibit", "danger_now", "threat_now",
              "alpha_mode", "turn_tokens"):
        try:
            out[k] = float(m[k]) if k != "turn_tokens" else int(m[k])
        except (KeyError, TypeError, ValueError):
            continue
    e = m.get("e")
    if isinstance(e, (list, tuple)) and len(e) == 4:
        try:
            out["e"] = [float(x) for x in e]
        except (TypeError, ValueError):
            pass
    action = m.get("action")
    if isinstance(action, str) and action:
        out["action"] = action
    # 没有核心三项就不算「已计算」，返回 None 让上层按未计算处理
    if not all(k in out for k in ("w_PFC", "I_inhibit", "e")):
        return None
    return out


def apply_gap_decay(st, gap_days):
    """[055 任务3 / 058 任务1] 跨天衰减四条通道（load 与常驻进程共用）。

    通道（常数见 config TAU_GAP_DAYS_*，全部 [未校准]）：
      E_prev                 ×= exp(-gap / TAU_GAP_DAYS_EMOTION)   tau=1 天
      w_PFC                  ×= exp(-gap / TAU_GAP_DAYS_FATIGUE)   tau=2 天
      HabituationSignal_prev ×= exp(-gap / TAU_GAP_DAYS_HABIT)     tau=3 天
      session_gap_days       += gap（包级召回折扣的全局乘数；同一会话内
                             多次空档累加——记忆只会比轮龄显示的更旧）

    gap_days <= 0 时任何值都不衰减（exp(0)=1 的短路，逐位不动）。
    返回结算明细 dict（before/after，供 log 与探针观测）。
    语义边界：只消退"没有轮次发生的时间"，不碰任何当轮计算；
    何时调用由调用方决定（load 时一次 / 常驻轮初超阈值时一次）。
    """
    gap = max(0.0, float(gap_days or 0.0))
    e_before = np.asarray(st.get("E_prev"), dtype=float).copy()
    w_before = float(st.get("w_PFC", 0.0))
    h_before = float(st.get("HabituationSignal_prev", 0.0))
    if gap > 0.0:
        _e_tau = float(getattr(config, "TAU_GAP_DAYS_EMOTION", 0.0) or 0.0)
        if _e_tau > 0.0:
            st["E_prev"] = e_before * math.exp(-gap / _e_tau)
        _f_tau = float(getattr(config, "TAU_GAP_DAYS_FATIGUE", 0.0) or 0.0)
        if _f_tau > 0.0:
            st["w_PFC"] = w_before * math.exp(-gap / _f_tau)
        _h_tau = float(getattr(config, "TAU_GAP_DAYS_HABIT", 0.0) or 0.0)
        if _h_tau > 0.0:
            st["HabituationSignal_prev"] = h_before * math.exp(-gap / _h_tau)
    st["session_gap_days"] = (float(st.get("session_gap_days", 0.0) or 0.0)
                              + gap)
    return {
        "gap_days": float(gap),
        "E_prev_before": [float(x) for x in e_before],
        "E_prev_after": [float(x) for x in np.asarray(st["E_prev"], dtype=float)],
        "w_PFC_before": w_before,
        "w_PFC_after": float(st.get("w_PFC", 0.0)),
        "H_prev_before": h_before,
        "H_prev_after": float(st.get("HabituationSignal_prev", 0.0)),
        "session_gap_days_after": float(st.get("session_gap_days", 0.0)),
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
        # [045] 旧会话无 messages → 空列表（不因缺字段而拒绝载入）
        st["messages"] = dump_messages(data.get("messages"))
        # [050 A1] 旧会话无 last_metrics → None（/state 返回 null 而非 0.0）
        st["last_metrics"] = _dump_metrics(data.get("last_metrics"))
        # [056 任务1] 旧会话无 relationships → 保持 reset_state 默认账本
        _load_relationships(data, st)
        for key in ("E_prev", "B", "M_prev"):
            vec = _b64_to_vec(data.get(key))
            if vec is not None:
                st[key] = vec

        # [055 任务3] 跨天时间感：用 saved_at 算空档，让"没有轮次发生"的时间
        #   也产生作用。这是「隔 3 天」与「隔 30 轮」的本质区别——
        #   30 轮里有轮次推进（E 每轮 ×0.85、w_PFC 每轮累积），
        #   3 天里一轮都没有，靠的是这里的空档回落。
        #   三项各自独立衰减，常数见 config 的 TAU_GAP_DAYS_*。
        #   注意：只在这里做一次（每次 load 从 saved_at 重算），不会重复衰减。
        #   [058 任务1] 衰减逻辑提纯为 apply_gap_decay——常驻进程（LRU 命中
        #   不走 load）由 pipeline 轮初按 last_active 复用同一函数，两条路径
        #   四条通道逐位同口径。
        saved_at = data.get("saved_at")
        gap_days = 0.0
        if saved_at is not None:
            try:
                gap_days = max(0.0, (time.time() - float(saved_at)) / 86400.0)
            except (TypeError, ValueError):
                gap_days = 0.0
        apply_gap_decay(st, gap_days)
        # [058 任务1] 常驻空档基准对齐到加载时刻：saved_at 那段空档已在
        #   上面结算，此后由 pipeline 每轮结束推进 last_active，轮初按
        #   now - last_active 超阈值再结算——防常驻进程重复结算同一段。
        #   不落盘：落盘值会与 saved_at 构成双重结算（重启路径以 saved_at 为准）。
        st["last_active"] = time.time()

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

        # archiver 的包（走公开接口，不直接写私有列表）
        pkgs = []
        for d in (data.get("packages") or []):
            try:
                pkgs.append(_dict_to_pkg(d))
            except Exception as e:
                warn("[session_store] 跳过损坏的记忆包: %s" % e)
        st["archiver"].replace_packages(pkgs)

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
