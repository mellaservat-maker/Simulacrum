# ============================================================
# llm_client.py · 模块12 心流
# 支持多后端：mock / deepseek_official / deepseek_third
# 默认 mock，真实密钥由 secrets_local.py 提供
# 013：网络层瞬时故障有界重试（见 _post_json_with_retry），
#       重试耗尽抛 LLMUnavailableError，不静默降级
# 015：generate 支持 temperature 覆写（温度分层）；新增同输入 TTL 缓存
#       cached_generate（evaluator=0.0 测量 / archiver=0.3 创作，见 config）
# ============================================================

import json
import os
import sys
import time
import urllib.error
import urllib.request

import config

# [021.6] 网页端保存的密钥覆盖文件（与 secrets_local.py 同目录）。
# 该文件由 server_api 的 /config/key 写入，本模块只读取；原始密钥文件
# secrets_local.py 永远不被网页端修改。
_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "secrets_override.py")

try:
    import requests
except ImportError:
    requests = None  # venv 未装 requests 时走 urllib 兜底（两者行为等价）

# [023.5] 强制禁用代理（仅作用于本模块的 LLM API 外呼）：
# api.deepseek.com 为境内 API，直连即可达。但 requests/urllib 默认继承
# 系统代理（环境变量 HTTP(S)_PROXY 与 Windows WinINET 注册表都会被读取），
# 梯子软件关闭后残留的死代理（典型如 127.0.0.1:7890）会被自动套到 LLM
# 请求上，表现为连接被拒 / SSL EOF / 代理 502——即 023.5 排查的"网络不通"。
# 本模块外呼只有 LLM API 一处（BGE 模型走 HF 镜像且已本地缓存，不经此路），
# 故在网络层统一禁用代理，与系统代理配置解耦。若确需走代理，删除本常量及
# 下方两处 _NO_PROXY / _NO_PROXY_OPENER 用法即可恢复默认继承行为。
_NO_PROXY = {"http": None, "https": None}
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---- 瞬时故障重试策略（013 号工作新增）----
# 代理隧道 502 / 上游 5xx / 429：有限次退避重试，不静默、不吞异常。
# 4xx（鉴权/参数错误）为确定性失败，不重试，直接抛 RuntimeError。
# 重试耗尽抛 LLMUnavailableError（RuntimeError 子类），调用方与观测日志
# 均可明确识别"真 LLM 不可用"，与"返回了垃圾内容"严格区分。
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (2.0, 5.0)
_TRANSIENT_HTTP_CODES = (429, 500, 502, 503, 504)


class LLMUnavailableError(RuntimeError):
    """LLM 通道在有限次重试后仍不可用（网络层故障）。"""


# ---- [021] Token 消耗统计（全局累加器）----
# 设计：DeepSeek 每次响应携带 usage（prompt_tokens / completion_tokens /
#   total_tokens）。三处调用点（evaluator 情绪提取、主模型生成、archiver
#   归档）都经过 DeepSeekLLM.generate，故在 generate 内统一记录即可全量
#   覆盖，无需各处埋点。
#   cached_generate 命中缓存时不发请求 → 不产生 token（正确语义）；
#   需要「本轮消耗」时用 token_snapshot() 前后取差值。
_TOKEN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,          # 真实 API 调用次数（缓存命中不计）
}


def _record_usage(usage):
    """记录一次 API 调用的 usage。缺字段/类型异常一律静默记 0（不抛）。"""
    if not isinstance(usage, dict):
        return
    try:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt_raw = usage.get("total_tokens", None)
        tt = int(tt_raw) if tt_raw is not None else (pt + ct)
    except (TypeError, ValueError):
        return
    _TOKEN_STATS["prompt_tokens"] += pt
    _TOKEN_STATS["completion_tokens"] += ct
    _TOKEN_STATS["total_tokens"] += tt
    _TOKEN_STATS["calls"] += 1


def token_stats():
    """返回累计 Token 统计副本（观测用）。"""
    return dict(_TOKEN_STATS)


def token_snapshot():
    """当前累计 total_tokens（用于「本轮消耗 = 轮末 - 轮初」）。"""
    return int(_TOKEN_STATS["total_tokens"])


def token_reset():
    """清零累计统计（重置会话/测试用）。"""
    for _k in _TOKEN_STATS:
        _TOKEN_STATS[_k] = 0


def _post_json_with_retry(url, headers, payload):
    """POST 到 LLM API 并返回解析后的 JSON dict。

    瞬时故障（OSError/超时/URLError/5xx/429）最多 _MAX_ATTEMPTS 次尝试，
    每次间隔 _BACKOFF_SECONDS；4xx 不重试。

    [020.2] 响应体防空/防垃圾：上游偶发返回 HTTP 200 但响应体为空或非
    JSON（余额不足的网关变体、代理错误页、限流页）。此前 json.loads 的
    ValueError 不在任何 except 分支内，会直接冒泡成
    「JSONDecodeError: Expecting value: line 1 column 1 (char 0)」打断
    整轮对话（020.1 只兜住了 content 为空的情况，没兜响应体本身）。
    现在统一转成 URLError 走重试通道；耗尽后 raise
    LLMUnavailableError（带响应体前 120 字诊断，不含密钥），由上层
    evaluator 降级链 / archiver 归档降级接住，对话不中断。
    """
    data_bytes = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            if requests is not None:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=60, proxies=_NO_PROXY)
                if resp.status_code in _TRANSIENT_HTTP_CODES:
                    raise urllib.error.URLError(f"上游返回瞬时错误 HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM 请求失败: HTTP {resp.status_code} {resp.reason}")
                body = resp.text
                if not (body or "").strip():
                    raise urllib.error.URLError(
                        "上游返回 HTTP 200 但响应体为空（疑似网关/余额/限流变体）")
                try:
                    return json.loads(body)
                except ValueError:
                    raise urllib.error.URLError(
                        f"上游响应体非 JSON（前 120 字：{body[:120]!r}）")
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with _NO_PROXY_OPENER.open(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if not body.strip():
                    raise urllib.error.URLError(
                        "上游返回 HTTP 200 但响应体为空（疑似网关/余额/限流变体）")
                try:
                    return json.loads(body)
                except ValueError:
                    raise urllib.error.URLError(
                        f"上游响应体非 JSON（前 120 字：{body[:120]!r}）")
        except urllib.error.HTTPError as e:
            if e.code in _TRANSIENT_HTTP_CODES:
                if attempt < _MAX_ATTEMPTS:
                    last_err = e
                    print(f"[llm_client] 上游瞬时错误 HTTP {e.code}，"
                          f"第 {attempt}/{_MAX_ATTEMPTS} 次尝试，退避后重试", file=sys.stderr)
                else:
                    raise LLMUnavailableError(
                        f"LLM 通道在 {_MAX_ATTEMPTS} 次尝试后仍不可用"
                        f"（上游持续返回瞬时错误 HTTP {e.code} {e.reason}）") from e
            else:
                # 4xx 为确定性失败（鉴权/参数错误），不重试
                raise RuntimeError(f"LLM 请求失败: {e.code} {e.reason}") from e
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            # 代理隧道 502 等网络层故障在 urllib 下表现为 URLError/OSError；
            # requests 的 ConnectionError/Timeout 亦为 OSError 子类，同一通道处理。
            if attempt < _MAX_ATTEMPTS:
                last_err = e
                print(f"[llm_client] 网络层故障 {type(e).__name__}，"
                      f"第 {attempt}/{_MAX_ATTEMPTS} 次尝试，退避后重试", file=sys.stderr)
            else:
                raise LLMUnavailableError(
                    f"LLM 通道在 {_MAX_ATTEMPTS} 次尝试后仍不可用"
                    f"（网络层故障，末次 {type(e).__name__}: {e}）") from e
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)])
    raise LLMUnavailableError(f"LLM 通道重试耗尽（末次错误 {last_err!r}）")


# ---- 密钥加载 ----

# ---- [022] 运行时 LLM 配置注入（网页设置面板）----
# 优先级：运行时注入 > secrets_override.py > secrets_local.py
#   · set_runtime_llm_config() 由 server_api 的 /api/config 调用，立即生效；
#   · 进程重启后运行时注入丢失，前端会用 localStorage 里的配置重新同步一次；
#   · 本模块任何情况下不打印 api_key（日志只记是否已设置）。
_RUNTIME_LLM = {"base_url": None, "api_key": None, "model": None}


def set_runtime_llm_config(base_url=None, api_key=None, model=None):
    """注入 LLM 配置（None 表示该项不变，空串表示清除该项）。"""
    if base_url is not None:
        _RUNTIME_LLM["base_url"] = (base_url or "").strip() or None
    if api_key is not None:
        _RUNTIME_LLM["api_key"] = (api_key or "").strip() or None
    if model is not None:
        _RUNTIME_LLM["model"] = (model or "").strip() or None
    # 配置变更后清空同输入缓存：缓存 key 不含模型名，避免换模型后
    # 仍返回旧模型的结果。
    _CACHE.clear()


def clear_runtime_llm_config():
    for _k in _RUNTIME_LLM:
        _RUNTIME_LLM[_k] = None
    _CACHE.clear()


def runtime_llm_config():
    """运行时注入状态（只暴露是否已配置与地址/模型，绝不返回 key）。"""
    return {
        "has_key": bool(_RUNTIME_LLM["api_key"]),
        "base_url": _RUNTIME_LLM["base_url"] or "",
        "model": _RUNTIME_LLM["model"] or "",
        "active": bool(_RUNTIME_LLM["api_key"] and _RUNTIME_LLM["base_url"]),
    }


def _load_secrets():
    """加载密钥配置。

    [021.6] 支持网页端保存：secrets_override.py（网页写入）覆盖
    secrets_local.py 的同名变量。合并策略：
      底座 = secrets_local 的全部公开属性（缺失则为空）
      覆盖 = secrets_override 的同名属性（网页端写入的 key / base_url / model）
      两者都缺失 → None（上层降级 MockLLM）
    网页端永远只写 secrets_override.py，不触碰 secrets_local.py。
    """
    import types
    # 文件被删（网页端"清除"）时清掉 import 缓存，避免旧模块残留
    if not os.path.exists(_OVERRIDE_PATH):
        sys.modules.pop("secrets_override", None)
    base = None
    try:
        import secrets_local as _base_mod
        base = {k: v for k, v in vars(_base_mod).items()
                if not k.startswith("__")}
    except ImportError:
        base = None
    try:
        import secrets_override as _ov_mod      # noqa: F401
        ovd = {k: v for k, v in vars(_ov_mod).items()
               if not k.startswith("__")}
    except ImportError:
        ovd = {}
    if base is None and not ovd:
        print("[llm_client] 警告：未找到 secrets_local.py，降级为 MockLLM")
        return None
    ns = types.SimpleNamespace()
    ns.__dict__.update(base or {})
    ns.__dict__.update(ovd)
    return ns


# ---- MockLLM ----

class MockLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return f"[MOCK] 收到 prompt，长度 {len(prompt)}"


# ---- DeepSeekLLM ----

class DeepSeekLLM:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt, temperature=None, messages=None):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # [055] messages 优先：调用方传多轮对话时原样使用；
        #   未传则退回单条 user（旧调用点与探针行为不变）。
        if messages:
            _msgs = list(messages)
        else:
            _msgs = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": _msgs,
            "temperature": (config.LLM_TEMPERATURE if temperature is None
                            else float(temperature)),
            "max_tokens": config.LLM_MAX_TOKENS,
        }

        # [020.1] 空响应防空：上游偶发返回空 content（HTTP 200 但内容为空，
        #   非网络错误），若原样返回会让上层 json.loads("") 抛
        #   JSONDecodeError（Expecting value: line 1 column 1），且空串会被
        #   cached_generate 缓存 300s 污染后续同 prompt 调用（失败不缓存的
        #   逻辑只覆盖异常，不覆盖空串）。处理：重试 1 次；仍空则 raise
        #   LLMUnavailableError——上层降级链接管，异常不入缓存。
        for _try in range(2):
            data = _post_json_with_retry(url, headers, payload)
            # [021] 先记 usage（无论 content 是否可解析，本次调用都已计费）
            _record_usage(data.get("usage") if isinstance(data, dict) else None)
            # [020.2] 200 变体的业务错误：某些网关/上游在余额不足、内容
            #   审核拦截时返回 HTTP 200 + {"error": {...}}（无 choices）。
            #   直接转 LLMUnavailableError 并带上 error 文本（不含密钥），
            #   避免 KeyError: 'choices' 这种无信息量的崩溃。
            if isinstance(data, dict) and data.get("error"):
                raise LLMUnavailableError(
                    f"上游错误响应：{str(data.get('error'))[:300]}")
            content = data["choices"][0]["message"]["content"]
            if content is not None and str(content).strip():
                return content
            if _try == 0:
                print("[llm_client] 上游返回空内容（HTTP 200），重试 1 次",
                      file=sys.stderr)
                time.sleep(1.0)
        raise LLMUnavailableError("LLM 返回空内容（重试 1 次后仍为空）")


# ---- 同输入缓存（015 温度分层配套）----
# key=(temperature, prompt)，TTL 默认取 config.LLM_CACHE_TTL_SECONDS（300s）。
# 计时用 time.monotonic（不受墙钟调整影响）。缓存只省重复调用、降抖动，
# 不改变任何语义；LLMUnavailableError 等异常不入缓存（失败不缓存，下次重试）。
_CACHE = {}
_CACHE_STATS = {"hit": 0, "miss": 0}


def cached_generate(llm, prompt, temperature, ttl=None):
    """带 TTL 的同输入缓存生成。返回 LLM 文本输出。"""
    if ttl is None:
        ttl = getattr(config, "LLM_CACHE_TTL_SECONDS", 300)
    key = (float(temperature), prompt)
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and hit[0] > now:
        _CACHE_STATS["hit"] += 1
        return hit[1]
    _CACHE_STATS["miss"] += 1
    content = llm.generate(prompt, temperature=temperature)
    _CACHE[key] = (now + float(ttl), content)
    return content


def cache_stats():
    """缓存命中统计（观测用）。"""
    return dict(_CACHE_STATS)


def cache_clear():
    """清空缓存（测试/诊断用）。"""
    _CACHE.clear()


# ---- 工厂函数 ----

def get_llm():
    # [022] 运行时注入优先（网页设置面板写入，立即生效）
    if _RUNTIME_LLM["api_key"] and _RUNTIME_LLM["base_url"]:
        return DeepSeekLLM(_RUNTIME_LLM["api_key"], _RUNTIME_LLM["base_url"],
                           _RUNTIME_LLM["model"] or "deepseek-chat")

    backend = config.LLM_BACKEND

    if backend == "mock":
        return MockLLM()

    if backend in ("deepseek_official", "deepseek_third"):
        secrets = _load_secrets()
        if secrets is None:
            return MockLLM()

        if backend == "deepseek_official":
            api_key = getattr(secrets, "DEEPSEEK_OFFICIAL_API_KEY", "")
            base_url = getattr(secrets, "DEEPSEEK_OFFICIAL_BASE_URL", "")
        else:
            api_key = getattr(secrets, "DEEPSEEK_THIRD_API_KEY", "")
            base_url = getattr(secrets, "DEEPSEEK_THIRD_BASE_URL", "")

        model = getattr(secrets, "DEEPSEEK_MODEL", "deepseek-chat")

        if not api_key or not base_url:
            print(f"[llm_client] 警告：{backend} 的密钥/URL 为空，降级为 MockLLM")
            return MockLLM()

        return DeepSeekLLM(api_key, base_url, model)

    raise ValueError(f"未知 LLM_BACKEND: {backend}")
