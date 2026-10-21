# ============================================================
# probe_llm_retry.py · 013 探针：llm_client 重试与失败语义自检
# 不碰真网络：monkeypatch urllib.request.urlopen 模拟三种剧本
#   剧本1「抖两次后成功」：前 2 次 OSError，第 3 次成功 → 应成功返回
#   剧本2「永远失败」：恒 OSError → 3 次尝试后抛 LLMUnavailableError
#   剧本3「HTTP 502」：HTTPError(502) 属瞬时 → 重试后抛 LLMUnavailableError
#   剧本4「HTTP 401」：确定性失败 → 不重试，直接 RuntimeError
# 另验证 requests=None 时（本 venv 实况）urllib 兜底路径可跑通。
# 断言：尝试次数、异常类型、返回内容。跑完即删（临时探针）。
# ============================================================

import io
import json
import os
import sys
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import llm_client
from llm_client import LLMUnavailableError, _post_json_with_retry

URL = "https://example.invalid/chat/completions"
HEADERS = {"Content-Type": "application/json"}
PAYLOAD = {"model": "x", "messages": []}
GOOD = {"choices": [{"message": {"content": "Pong!"}}]}

all_ok = True


def check(name, cond, detail=""):
    global all_ok
    all_ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class FakeResp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_urlopen(script):
    """script: 每次调用返回 ('ok', resp) 或 ('err', exc)；按序取，取尽重复末项。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=60):
        i = calls["n"]
        calls["n"] += 1
        kind, val = script[min(i, len(script) - 1)]
        if kind == "ok":
            return val
        raise val

    return fake_urlopen, calls


print("=== 013 探针：llm_client 重试与失败语义 ===")
print(f"requests 可用性：{'已安装' if llm_client.requests is not None else '未安装（走 urllib 兜底）'}")
print(f"退避序列 {llm_client._BACKOFF_SECONDS}s，最大尝试 {llm_client._MAX_ATTEMPTS} 次")
print()

# 剧本1：抖两次后成功（把退避压到 0.01 避免真睡）
print("剧本1：前 2 次 OSError，第 3 次成功")
orig_backoff = llm_client._BACKOFF_SECONDS
llm_client._BACKOFF_SECONDS = (0.01, 0.01)
fake, calls = make_urlopen([
    ("err", OSError("Tunnel connection failed: 502 Bad Gateway")),
    ("err", OSError("Tunnel connection failed: 502 Bad Gateway")),
    ("ok", FakeResp(GOOD)),
])
import urllib.request
real_urlopen = urllib.request.urlopen
urllib.request.urlopen = fake
try:
    out = _post_json_with_retry(URL, HEADERS, PAYLOAD)
    check("重试后成功返回", out == GOOD, f"content={out['choices'][0]['message']['content']}")
    check("恰好 3 次尝试", calls["n"] == 3, f"实际 {calls['n']} 次")
finally:
    urllib.request.urlopen = real_urlopen
    llm_client._BACKOFF_SECONDS = orig_backoff
print()

# 剧本2：永远 OSError → LLMUnavailableError
print("剧本2：恒 OSError → 重试耗尽")
llm_client._BACKOFF_SECONDS = (0.01, 0.01)
fake, calls = make_urlopen([("err", OSError("Tunnel connection failed: 502 Bad Gateway"))])
urllib.request.urlopen = fake
try:
    try:
        _post_json_with_retry(URL, HEADERS, PAYLOAD)
        check("应抛 LLMUnavailableError", False, "未抛异常")
    except LLMUnavailableError as e:
        check("抛 LLMUnavailableError", True, f"消息含尝试次数：{'3 次尝试' in str(e)}")
        check("恰好 3 次尝试后放弃", calls["n"] == 3, f"实际 {calls['n']} 次")
    except Exception as e:
        check("抛 LLMUnavailableError", False, f"抛了 {type(e).__name__}: {e}")
finally:
    urllib.request.urlopen = real_urlopen
    llm_client._BACKOFF_SECONDS = orig_backoff
print()

# 剧本3：HTTP 502 属瞬时 → 同样重试耗尽
print("剧本3：恒 HTTPError(502)")
llm_client._BACKOFF_SECONDS = (0.01, 0.01)
fake, calls = make_urlopen([("err", urllib.error.HTTPError(URL, 502, "Bad Gateway", None, None))])
urllib.request.urlopen = fake
try:
    try:
        _post_json_with_retry(URL, HEADERS, PAYLOAD)
        check("应抛 LLMUnavailableError", False, "未抛异常")
    except LLMUnavailableError:
        check("502 重试耗尽抛 LLMUnavailableError", True)
        check("恰好 3 次尝试", calls["n"] == 3, f"实际 {calls['n']} 次")
    except Exception as e:
        check("应抛 LLMUnavailableError", False, f"抛了 {type(e).__name__}: {e}")
finally:
    urllib.request.urlopen = real_urlopen
    llm_client._BACKOFF_SECONDS = orig_backoff
print()

# 剧本4：HTTP 401 确定性失败 → 不重试
print("剧本4：HTTPError(401) 确定性失败")
fake, calls = make_urlopen([("err", urllib.error.HTTPError(URL, 401, "Unauthorized", None, None))])
urllib.request.urlopen = fake
try:
    try:
        _post_json_with_retry(URL, HEADERS, PAYLOAD)
        check("应抛 RuntimeError", False, "未抛异常")
    except LLMUnavailableError:
        check("401 不应走重试耗尽通道", False, "抛了 LLMUnavailableError")
    except RuntimeError as e:
        check("401 直接抛 RuntimeError 不重试", True, f"消息={e}")
        check("只尝试 1 次", calls["n"] == 1, f"实际 {calls['n']} 次")
    except Exception as e:
        check("应抛 RuntimeError", False, f"抛了 {type(e).__name__}: {e}")
finally:
    urllib.request.urlopen = real_urlopen
print()

# 回归：MockLLM 与工厂不受影响
from llm_client import get_llm, MockLLM
check("MockLLM 仍可构造", isinstance(MockLLM(), MockLLM))
print()
print("探针结论：%s" % ("全部 PASS" if all_ok else "存在 FAIL，需检查"))
sys.exit(0 if all_ok else 1)
