# ============================================================
# server_api.py · [021.2] 前端桥接层（FastAPI）
#
# 用途：把前端设计 AI 交付的 index.html 接到真实 V6.4 链路。
#   前端契约（index.html normalizeResponse，字段宽松兼容）：
#     POST /chat  {message, session_id, deep_think, web_search}
#       -> {reply, w_PFC, I_inhibit, e0, recall:{text,meta}, usage:{total_tokens}, ...}
#   本层直接复用 scripts/chat_cli_v6.py 的 chat_turn（fixture 注入 +
#   pipeline.run_turn encoder 模式），因此 compute_danger_score、极性
#   一致性过滤、threat_now、alpha 迟滞机全部是主树代码本体，零重写。
#
# 附加接口：
#   GET  /state?session_id=...   完整内部状态（调试面板/轮询）
#   POST /reset  {session_id}    重置会话（清 st 与 Token 累计）
#   GET  /health                 存活探针
#
# 启动：D:\aiqg\venv\Scripts\python.exe -m uvicorn server_api:app
#        --host 0.0.0.0 --port 8000
#        （0.0.0.0 允许同一局域网的手机访问；仅本机用 127.0.0.1 亦可）
# ============================================================

import os
import sys
import threading
import time

try:
    import requests as _requests
except ImportError:            # venv 未装 requests 时 /api/models 不可用
    _requests = None

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

import config                                          # noqa: E402
import llm_client                                      # noqa: E402

from state import reset_state                          # noqa: E402
from chat_cli_v6 import chat_turn                      # noqa: E402
from llm_client import (token_stats, LLMUnavailableError)   # noqa: E402
import session_store                                   # noqa: E402

app = FastAPI(title="emotion_agent bridge", version="6.4")

# 前端可能是 file:// 打开（Origin: null）或任意 localhost 端口，放行本机
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话表：session_id -> st。chat_turn 内部会临时 patch
# pipeline.get_fixture（全局），故用一把锁串行化，避免并发互相覆盖。
# [021.3] 多对话隔离：一个浏览器可开多个对话，每个对话持有独立
#   session_id，状态各自保留（切回旧对话继续累积）。用 OrderedDict
#   实现 LRU：超过上限淘汰最久未用的会话，防内存无限增长。
from collections import OrderedDict

_LOCK = threading.Lock()
_SESSIONS = OrderedDict()
_MAX_SESSIONS = 50
# [021.4] 持久化开关：True 时每轮落盘 + 未命中时从磁盘恢复。
#   环境变量 EMOTION_NO_PERSIST=1 可关闭（测试/排查用）。
_PERSIST = os.environ.get("EMOTION_NO_PERSIST", "") != "1"


def _get_session(sid: str):
    if sid in _SESSIONS:
        _SESSIONS.move_to_end(sid)      # LRU：刷新活跃度
        return _SESSIONS[sid]
    # [021.4] 内存未命中 → 先尝试从磁盘恢复（关掉 API 再打开可续聊）
    st = None
    if _PERSIST:
        st, meta = session_store.load(sid)
        if st is not None and meta:
            print("[server_api] 会话 %s 已从磁盘恢复：round=%s "
                  "packages=%s events=%s"
                  % (sid, meta.get("round"), meta.get("packages"),
                     meta.get("events")), file=sys.stderr)
    if st is None:
        st = reset_state(seed=42)
    _SESSIONS[sid] = st
    if len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)   # 淘汰最久未用的（磁盘文件保留）
    return st


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"
    deep_think: bool = False     # 前端 UI 开关；当前链路无此能力，忽略
    web_search: bool = False     # 同上
    # [041] 性格预设：standard / sensitive / gentle（也接受中文别名）。
    #   留空表示沿用会话当前性格；传入时在开轮前写入会话（仍是冷切换）。
    profile_id: str = ""


class ResetIn(BaseModel):
    session_id: str = "default"


def _recall_text(rm, st):
    """把召回包转成前端可显示的一句话（text）与标签（meta）。"""
    if not isinstance(rm, dict):
        return None
    pol = rm.get("emotion_polarity") or "neutral"
    pol_cn = {"positive": "被夸奖", "negative": "被责怪", "neutral": "中性互动"}.get(pol, "互动")
    rounds = rm.get("recalled_rounds") or []
    quote = ""
    rid = None
    for r in rounds:
        if isinstance(r, dict) and r.get("round_id") is not None:
            rid = r.get("round_id")
            break
    if rid is not None:
        rec = None
        try:
            rec = st["raw_log"].get(int(rid))
        except Exception:
            rec = None
        if rec:
            quote = str(rec.get("user_input", ""))[:40]
    text = ("「%s」" % quote) if quote else ("一段%s的记忆" % pol_cn)
    meta = "%s · 相似度 %.2f · %s" % (
        pol_cn, float(rm.get("matched_sim", 0.0)),
        "轮级回捞" if rm.get("recall_source") == "round_level" else "包级")
    age = rm.get("pkg_age_rounds")
    if age is not None:
        meta += " · %d 轮前" % int(age)
    return {"text": text, "meta": meta}


def _state_payload(st, log=None):
    """完整状态（供 /state 与 /chat 的 state 字段）。"""
    base = {
        "round": st.get("round_counter", 0),
        "buffer": len(st["memory_buffer"]),
        "packages": len(st["archiver"].get_packages()),
        "tokens": token_stats(),
        # [041] 性格、破防冷却与阈值，前端直接展示
        "profile_id": str(st.get("profile_id", config.PERSONALITY_DEFAULT)),
        "profile_label": config.PERSONALITY_PROFILES.get(
            st.get("profile_id", config.PERSONALITY_DEFAULT), {}).get("label", ""),
        "express_cooldown": int(st.get("express_cooldown", 0)),
        "thresholds": {
            "inhibit_high": float(config.POLICY_THRESHOLD_INHIBIT_HIGH),
            "inhibit_mid": float(config.POLICY_THRESHOLD_INHIBIT_MID),
            "express_valence": float(config.POLICY_EXPRESS_VALENCE),
            "express_inhibit": float(config.POLICY_EXPRESS_INHIBIT),
        },
        "e": [float(x) for x in st.get("E_prev", [0.0, 0.0, 0.0, 0.0])],
        # [042] 无轮次（log is None）时也必须给出这四项，否则前端只能显示
        #   占位值，误以为初始状态是 0.18/0.92/+0.16。
        #   口径（042 裁决）：I_inhibit 首轮未计算，返回 0.0 = 无抑制状态；
        #   action 取 policy 默认值与首轮默认动作一致。
        "w_PFC": float(st.get("w_PFC", 0.0)),
        "I_inhibit": 0.0,
        "e0": float(st.get("E_prev", [0.0])[0]),
        "action": "task_first",
    }
    if log is not None:
        base.update({
            "w_PFC": float(log["w_PFC"]),
            "I_inhibit": float(log["I_inhibit"]),
            "threat_now": float(log.get("threat_now", 0.0)),
            "alpha_mode": float(log.get("alpha_mode", 0.0)),
            "e0": float(log["E_final"][0]),
            "e": [float(x) for x in log["E_final"]],
            "action": log.get("action"),
            "danger_now": float(log.get("danger_now", 0.0)),
            "express_cooldown": int(st.get("express_cooldown", 0)),
            "turn_tokens": int(log.get("turn_tokens", 0)),
            "S": [float(x) for x in log.get("S", [])],
            "S_source": log.get("S_source"),
            "recalled": bool(st.get("recalled_memory")),
        })
    return base


@app.get("/health")
def health():
    return {"ok": True, "sessions": len(_SESSIONS), "tokens": token_stats()}


# [022] 前後端同源托管：FastAPI 直接提供 web/index.html，
#   访问 http://127.0.0.1:8000/ 即可用（不再依赖 file:// 打开，
#   也就没有 CORS / 本地文件权限问题）。
#   注意：mount 必须放在所有 API 路由之后，避免抢路由。
_WEB_DIR = os.path.join(BASE, "web")
if os.path.isdir(_WEB_DIR):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    def web_index():
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")


@app.on_event("startup")
def _maybe_open_browser():
    """EMOTION_OPEN_BROWSER=1 时启动后自动打开浏览器（供 启动网页版.bat 用）。"""
    if os.environ.get("EMOTION_OPEN_BROWSER") != "1":
        return
    import threading
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open(
        "http://127.0.0.1:8000/")).start()


@app.get("/api/sessions")
def api_list_sessions():
    """[044] 历史会话列表：扫描 sessions/ 目录，供前端下拉恢复任意存档。

    返回 default（config.DEFAULT_WEB_SESSION，前端首次打开默认用它）
    与 sessions 列表（session_id / size / mtime，按修改时间倒序）。
    """
    items = []
    if _PERSIST:
        for f in session_store.list_files():
            items.append({
                "session_id": f["session_id"],
                "size": int(f["size"]),
                "mtime": float(f["mtime"]),
            })
    return {"default": config.DEFAULT_WEB_SESSION, "sessions": items}


@app.get("/api/history")
def api_history(session_id: str = "default"):
    """[045] 返回该会话的完整聊天记录（用户 + AI 成对追加）。

    内存未命中时会先按 session_id 从 sessions/ 恢复，因此换浏览器/
    换设备首次打开也能拿到同一份文字历史。旧存档无 messages 字段 → 空列表。
    """
    with _LOCK:
        st = _get_session(session_id)
        return {
            "session_id": session_id,
            "round": int(st.get("round_counter", 0)),
            "messages": session_store._dump_messages(st.get("messages")),
        }


@app.get("/sessions")
def list_sessions():
    """[021.3/021.4] 内存中的会话 + 磁盘上的会话文件。"""
    with _LOCK:
        return {
            "count": len(_SESSIONS),
            "max": _MAX_SESSIONS,
            "persist": _PERSIST,
            "items": [
                {"session_id": sid,
                 "round": st.get("round_counter", 0),
                 "packages": len(st["archiver"].get_packages()),
                 "buffer": len(st["memory_buffer"])}
                for sid, st in _SESSIONS.items()
            ],
            "disk": session_store.list_files() if _PERSIST else [],
        }


@app.post("/chat")
def chat(body: ChatIn):
    text = (body.message or "").strip()
    if not text:
        return {"reply": "", "error": "empty message"}
    with _LOCK:
        st = _get_session(body.session_id)
        # [041] 性格冷切换：请求里带 profile_id 时写入本会话，下一轮起生效。
        #   非法值直接 400，不静默回落，避免前端以为切了其实没切。
        if body.profile_id:
            try:
                st["profile_id"] = config.load_personality(body.profile_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        try:
            log, S, src = chat_turn(st, text)
        except LLMUnavailableError as e:
            # 上游通道异常：不 500，返回降级提示（前端可正常显示）
            return {
                "reply": "（上游模型通道暂时不可用，本轮已降级跳过。可以稍后再试。）",
                "error": "llm_unavailable",
                "detail": str(e)[:200],
                "w_PFC": float(st.get("w_PFC", 0.0)),
                "I_inhibit": 0.0, "e0": float(st["E_prev"][0]),
                "recall": None,
                "usage": {"total_tokens": 0},
                "state": _state_payload(st),
            }
        except Exception as e:
            return {
                "reply": "（本轮处理出错：%s: %s）" % (type(e).__name__, e),
                "error": "internal",
                "w_PFC": float(st.get("w_PFC", 0.0)),
                "I_inhibit": 0.0, "e0": float(st["E_prev"][0]),
                "recall": None,
                "usage": {"total_tokens": 0},
                "state": _state_payload(st),
            }
        rm = st.get("recalled_memory")
        # [045] 聊天记录随状态一并落盘（跨浏览器/跨设备拉取同一份文字历史）。
        #   只记录成功轮次；失败轮次的兜底提示不入库，避免污染历史。
        reply_text = str(log.get("llm_output", "")).strip()
        msgs = st.setdefault("messages", [])
        msgs.append({"role": "user", "content": text, "ts": time.time()})
        msgs.append({"role": "assistant", "content": reply_text, "ts": time.time()})
        # [021.4] 每轮成功后落盘（原子写）。失败只告警，不影响本轮响应。
        if _PERSIST:
            session_store.save(body.session_id, st)
        return {
            "reply": reply_text,
            "w_PFC": float(log["w_PFC"]),
            "I_inhibit": float(log["I_inhibit"]),
            "e0": float(log["E_final"][0]),
            # [041] 四维情绪 + 破防冷却 + 危险度 + 阈值
            "e": [float(x) for x in log["E_final"]],
            "action": log.get("action"),
            "express_cooldown": int(st.get("express_cooldown", 0)),
            "danger_now": float(log.get("danger_now", 0.0)),
            "profile_id": str(st.get("profile_id", config.PERSONALITY_DEFAULT)),
            "profile_label": config.PERSONALITY_PROFILES.get(
                st.get("profile_id", config.PERSONALITY_DEFAULT), {}).get("label", ""),
            "thresholds": {
                "inhibit_high": float(config.POLICY_THRESHOLD_INHIBIT_HIGH),
                "inhibit_mid": float(config.POLICY_THRESHOLD_INHIBIT_MID),
                "express_valence": float(config.POLICY_EXPRESS_VALENCE),
                "express_inhibit": float(config.POLICY_EXPRESS_INHIBIT),
            },
            "alpha_mode": float(log.get("alpha_mode", 0.0)),
            "threat_now": float(log.get("threat_now", 0.0)),
            "recall": _recall_text(rm, st),
            "usage": {"total_tokens": int(log.get("turn_tokens", 0))},
            "tokens": {"turn": int(log.get("turn_tokens", 0)),
                       "total": int(token_stats()["total_tokens"])},
            "state": _state_payload(st, log),
        }


@app.get("/state")
def state(session_id: str = "default"):
    with _LOCK:
        st = _get_session(session_id)
        return _state_payload(st)


@app.post("/reset")
def reset(body: ResetIn):
    """销毁一个会话（聊天历史里的删除对话 / 新建对话旧版逻辑调用）。

    [021.3] 不再清空全局 Token 计数：Token 是真实花费统计，多会话并存时
    删掉一个对话不应抹掉其他对话的计费。Token 只在进程重启时归零。
    """
    with _LOCK:
        existed = _SESSIONS.pop(body.session_id, None) is not None
        # [021.4] 同时删除磁盘归档（否则下次访问会"复活"旧状态）
        removed_disk = session_store.delete(body.session_id) if _PERSIST else False
        return {"ok": True, "reset": body.session_id, "existed": existed,
                "disk_removed": removed_disk, "sessions": len(_SESSIONS)}


# ============================================================
# [021.6] API Key 网页配置
# 安全设计：
#   1. 永不读取、永不修改 secrets_local.py（原始密钥文件的铁律）。
#      网页写入的是独立的 secrets_override.py，llm_client 加载时用它
#      覆盖同名变量。
#   2. 接口只回显「是否已配置」（布尔）与非敏感项（base_url/model），
#      任何情况下不返回 key 本身。
#   3. 写接口仅接受本机回环请求（uvicorn 已绑 127.0.0.1，双保险）。
# ============================================================

OVERRIDE_PATH = llm_client._OVERRIDE_PATH


def _override_vars():
    """读取 override 文件里的变量（本模块自己写的文件，可安全读取）。"""
    if not os.path.exists(OVERRIDE_PATH):
        return {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ov_probe", OVERRIDE_PATH)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return {k: v for k, v in vars(m).items() if not k.startswith("__")}
    except Exception as e:
        print("[server_api] override 读取失败（按空处理）：%s" % e,
              file=sys.stderr)
        return {}


def _write_override(data):
    """原子写 override 文件，并清 import 缓存使下一次加载生效。"""
    lines = [
        "# 由网页端保存（021.6）。只覆盖同名变量，不修改 secrets_local.py。",
        "# 本文件含明文密钥：请勿提交到任何仓库、勿分享、勿打包分发。",
    ]
    for k in sorted(data):
        lines.append("%s = %r" % (k, str(data[k])))
    tmp = OVERRIDE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, OVERRIDE_PATH)
    sys.modules.pop("secrets_override", None)   # 让 _load_secrets 读到新内容


class KeyIn(BaseModel):
    provider: str = "official"      # official | third
    api_key: str = ""               # 空字符串 = 清除该 provider 的覆盖密钥
    base_url: str = ""
    model: str = ""


@app.get("/config")
def get_config():
    """返回 LLM 配置状态（绝不返回密钥值）。"""
    sec = llm_client._load_secrets()

    def has(name):
        v = getattr(sec, name, "") if sec is not None else ""
        return bool(str(v or "").strip())

    def val(name):
        v = getattr(sec, name, "") if sec is not None else ""
        return str(v or "")

    rt = llm_client.runtime_llm_config()
    return {
        "backend": config.LLM_BACKEND,
        "official_key_set": has("DEEPSEEK_OFFICIAL_API_KEY"),
        "third_key_set": has("DEEPSEEK_THIRD_API_KEY"),
        "official_base_url": val("DEEPSEEK_OFFICIAL_BASE_URL"),
        "third_base_url": val("DEEPSEEK_THIRD_BASE_URL"),
        "model": rt["model"] or val("DEEPSEEK_MODEL"),
        "override_exists": os.path.exists(OVERRIDE_PATH),
        "runtime": rt,
        "note": "密钥只显示是否配置，不回显内容",
    }


# ============================================================
# [022] 动态模型配置：运行时注入 + 模型列表拉取
# ============================================================

class ConfigIn(BaseModel):
    base_url: str = ""
    api_key: str = ""          # 留空 = 沿用当前生效的 key
    model: str = ""


class ModelsIn(BaseModel):
    base_url: str = ""
    api_key: str = ""          # 留空 = 用当前生效的 key（便于首次配置时直接拉取）


def _assert_local(request: Request):
    host = (request.client.host if request.client else "") or ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="仅允许本机操作")


def _current_key():
    """服务端内部的当前生效 key（运行时注入 > override > secrets_local）。

    只在进程内使用，绝不返回给客户端、绝不打印。
    """
    rt = llm_client.runtime_llm_config()
    if rt["has_key"]:
        # runtime_llm_config 不暴露 key 值，这里直接读内部状态
        return llm_client._RUNTIME_LLM["api_key"] or ""
    sec = llm_client._load_secrets()
    if sec is None:
        return ""
    return str(getattr(sec, "DEEPSEEK_OFFICIAL_API_KEY", "") or "")


def _current_base():
    rt = llm_client.runtime_llm_config()
    if rt["base_url"]:
        return rt["base_url"]
    sec = llm_client._load_secrets()
    if sec is None:
        return ""
    return str(getattr(sec, "DEEPSEEK_OFFICIAL_BASE_URL", "") or "")


@app.post("/api/config")
def api_set_config(body: ConfigIn, request: Request):
    """保存动态模型配置：运行时注入 + 落盘 override（重启后仍生效）。

    优先级：运行时注入 > secrets_override.py > secrets_local.py。
    本接口永不修改 secrets_local.py，也不在任何日志中输出 key。
    """
    _assert_local(request)
    base = (body.base_url or "").strip().rstrip("/")
    model = (body.model or "").strip()
    key = (body.api_key or "").strip()
    if not base:
        raise HTTPException(status_code=400, detail="Base URL 不能为空")
    if key and (len(key) < 10 or any(c in key for c in "\r\n\"'")):
        raise HTTPException(status_code=400, detail="API Key 格式不正确（至少 10 字符、无引号换行）")
    if not key:
        key = _current_key()
        if not key:
            raise HTTPException(status_code=400, detail="API Key 不能为空")
    if not model:
        model = "deepseek-chat"

    # 1) 运行时注入（立即生效）
    llm_client.set_runtime_llm_config(base, key, model)
    # 2) 落盘（进程重启后由 override 恢复；前端 localStorage 也会再同步一次）
    data = _override_vars()
    data["DEEPSEEK_OFFICIAL_API_KEY"] = key
    data["DEEPSEEK_OFFICIAL_BASE_URL"] = base
    data["DEEPSEEK_MODEL"] = model
    _write_override(data)

    try:
        kind = type(llm_client.get_llm()).__name__
    except Exception as e:
        kind = "error: %s" % type(e).__name__
    print("[server_api] 动态配置已更新（model=%s, base=%s, llm=%s, key=%s）"
          % (model, base, kind, "set" if key else "empty"), file=sys.stderr)
    return {"ok": True, "base_url": base, "model": model, "llm": kind,
            "runtime": llm_client.runtime_llm_config(),
            "note": "已立即生效并保存；原 secrets_local.py 未修改"}


@app.post("/api/config/reset")
def api_reset_config(request: Request):
    """清除运行时注入 + 删除 override 文件，回落到 secrets_local.py。"""
    _assert_local(request)
    llm_client.clear_runtime_llm_config()
    if os.path.exists(OVERRIDE_PATH):
        try:
            os.remove(OVERRIDE_PATH)
        except OSError as e:
            print("[server_api] 删除 override 失败：%s" % e, file=sys.stderr)
    sys.modules.pop("secrets_override", None)
    print("[server_api] 动态配置已清除，回落 secrets_local（模型=%s）"
          % llm_client.runtime_llm_config().get("model", ""), file=sys.stderr)
    return {"ok": True, "runtime": llm_client.runtime_llm_config(),
            "note": "已清除动态配置，回落到 secrets_local.py"}


@app.post("/api/models")
def api_models(body: ModelsIn, request: Request):
    """按 {base_url, api_key} 拉取可用模型列表（OpenAI 兼容 /models）。

    探测顺序：{base}/v1/models → {base}/models。api_key 留空时用当前
    生效的 key（服务端内部读取，不回传、不打印）。
    """
    _assert_local(request)
    if _requests is None:
        return {"ok": False, "models": [], "error": "服务端缺少 requests 库"}
    base = (body.base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="Base URL 不能为空")
    key = (body.api_key or "").strip() or _current_key()
    if not key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")

    urls = []
    for u in (base + "/v1/models", base + "/models"):
        if u not in urls:
            urls.append(u)
    last = ""
    for url in urls:
        try:
            r = _requests.get(url, headers={"Authorization": "Bearer " + key},
                              timeout=20)
            if r.status_code != 200:
                last = "HTTP %d" % r.status_code
                continue
            data = r.json()
            items = data.get("data") or data.get("models") or []
            ids = []
            for m in items:
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("name") or m.get("model")
                else:
                    mid = m
                if mid:
                    ids.append(str(mid))
            ids = sorted(set(ids))
            if ids:
                print("[server_api] 模型列表拉取成功（%d 个，来源 %s）"
                      % (len(ids), url), file=sys.stderr)
                return {"ok": True, "models": ids, "source": url}
        except Exception as e:
            last = type(e).__name__
    return {"ok": False, "models": [],
            "error": "获取失败（%s）——请检查 Base URL 与 API Key" % (last or "无响应")}


@app.post("/config/key")
def set_key(body: KeyIn, request: Request):
    """保存（或清除）API Key 到 secrets_override.py，立即生效不重启。

    api_key 传空字符串 = 清除该 provider 的覆盖值（回落到 secrets_local
    里原有的配置；若 override 变空则删除该文件）。
    """
    host = (request.client.host if request.client else "") or ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="仅允许本机操作")
    provider = (body.provider or "").strip().lower()
    if provider not in ("official", "third"):
        raise HTTPException(status_code=400, detail="provider 只能是 official 或 third")
    key = (body.api_key or "").strip()

    data = _override_vars()
    key_name = ("DEEPSEEK_OFFICIAL_API_KEY" if provider == "official"
                else "DEEPSEEK_THIRD_API_KEY")
    url_name = ("DEEPSEEK_OFFICIAL_BASE_URL" if provider == "official"
                else "DEEPSEEK_THIRD_BASE_URL")

    if not key:
        # ---- 清除模式 ----
        data.pop(key_name, None)
        data.pop(url_name, None)
        removed = True
    else:
        if len(key) < 10:
            raise HTTPException(status_code=400, detail="密钥过短（至少 10 个字符）")
        if any(c in key for c in "\r\n\"'"):
            raise HTTPException(status_code=400, detail="密钥含非法字符")
        data[key_name] = key
        if (body.base_url or "").strip():
            data[url_name] = body.base_url.strip()
        removed = False
    if (body.model or "").strip():
        data["DEEPSEEK_MODEL"] = body.model.strip()

    if data:
        _write_override(data)
    else:
        # override 已空 → 删除文件并清缓存
        try:
            if os.path.exists(OVERRIDE_PATH):
                os.remove(OVERRIDE_PATH)
        except OSError as e:
            print("[server_api] 删除 override 失败：%s" % e, file=sys.stderr)
        sys.modules.pop("secrets_override", None)

    # 立即验证（不打印密钥）
    try:
        kind = type(llm_client.get_llm()).__name__
    except Exception as e:
        kind = "error: %s" % type(e).__name__
    print("[server_api] API Key 更新（provider=%s, action=%s, llm=%s）"
          % (provider, "clear" if removed else "set", kind), file=sys.stderr)
    return {"ok": True, "provider": provider,
            "action": "clear" if removed else "set",
            "llm": kind, "file": "secrets_override.py",
            "note": "已保存并立即生效；原 secrets_local.py 未被修改"}
