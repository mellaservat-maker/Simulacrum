# ============================================================
# probe_proxy_check.py · 023.5 代理排查探针
# 目的：定位"网络不通"是否由系统残留死代理（如 127.0.0.1:7890）造成。
#   1. 打印当前 Python 环境读到的代理环境变量（大小写两种都查）；
#   2. 打印 urllib.request.getproxies() 的系统代理视图
#      （Windows 上含 WinINET 注册表代理，与纯环境变量对照可识别注册表残留）；
#   3. 直连测试：对 https://api.deepseek.com/v1/models 各请求一次——
#      A 跟随系统代理（默认行为），B 强制禁用代理
#      （proxies={"http": None, "https": None} / ProxyHandler({})）。
# 判读：拿到任意 HTTP 状态码（含 401/403）= 连通；连接/代理/SSL/超时错误 = 不通。
# 不需要 API 密钥：401 恰好证明 TCP+TLS+HTTP 全通。
# ============================================================
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TEST_URL = "https://api.deepseek.com/v1/models"
TIMEOUT = 10


def section(title):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)


def main():
    print("=" * 72)
    print("代理排查探针（023.5）")
    print("=" * 72)
    print(f"Python: {sys.executable}")
    print(f"版本: {sys.version.split()[0]}  平台: {sys.platform}")

    try:
        import requests
        print(f"requests: 已安装 {requests.__version__}（将用 requests 分支测试）")
    except ImportError:
        requests = None
        print("requests: 未安装（将用 urllib 兜底分支测试）")

    # ---- 1. 环境变量 ----
    section("1. 环境变量中的代理设置")
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy"]
    found = False
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            print(f"  {k} = {v}")
            found = True
    if not found:
        print("  （无标准代理环境变量）")
    extra = [f"{k}={os.environ[k]}" for k in sorted(os.environ)
             if "PROXY" in k.upper() and k not in keys]
    for line in extra:
        print(f"  [其他] {line}")

    # ---- 2. urllib 系统代理视图（Windows 含注册表） ----
    section("2. urllib 系统代理视图")
    import urllib.request
    env_view = urllib.request.getproxies_environment()
    sys_view = urllib.request.getproxies()
    print(f"  getproxies_environment(): {env_view if env_view else '（空）'}")
    print(f"  getproxies()          : {sys_view if sys_view else '（空）'}")
    reg_only = {k: v for k, v in sys_view.items()
                if env_view.get(k) != v}
    if reg_only:
        print(f"  注册表/WinINET 独有条目: {reg_view_diff}")
        print("  （环境变量里没有、但 getproxies 里有的代理 = 系统/WinINET 残留，")
        print("   清环境变量治不了它，必须用强制禁用代理或改系统设置）")
    elif sys_view:
        print("  （系统代理全部来自环境变量，清环境变量即可生效）")

    # ---- 3. 直连测试 ----
    section(f"3. 连通性测试 {TEST_URL}")
    print("  A. 跟随系统代理（默认行为，复现老板遇到的故障形态）")
    _run_one("A", honor_proxy=True, requests_mod=requests)
    print("  B. 强制禁用代理（proxies 全 None / ProxyHandler({})）")
    _run_one("B", honor_proxy=False, requests_mod=requests)

    section("结论提示")
    print("  A 不通 + B 通 = 系统残留死代理是祸根，按报告建议处理；")
    print("  A 通 = 代理没问题，去别处找原因；")
    print("  A/B 都不通 = 本机到 api.deepseek.com 的直连也被断（沙箱/网络层封锁）。")


def _report_result(tag, honor_proxy, ok, detail):
    if ok:
        print(f"    [{tag}] 通：{detail}")
    else:
        print(f"    [{tag}] 不通：{detail}")


def _run_one(tag, honor_proxy, requests_mod):
    if requests_mod is not None:
        try:
            if honor_proxy:
                resp = requests_mod.get(TEST_URL, timeout=TIMEOUT)
            else:
                resp = requests_mod.get(
                    TEST_URL, timeout=TIMEOUT,
                    proxies={"http": None, "https": None})
            _report_result(tag, honor_proxy, True,
                           f"HTTP {resp.status_code} {resp.reason}"
                           f"（401/403 属预期：可达但未带密钥）")
        except Exception as e:
            _report_result(tag, honor_proxy, False,
                           f"{type(e).__name__}: {str(e)[:200]}")
        return
    # urllib 兜底
    import urllib.request
    import urllib.error
    try:
        if honor_proxy:
            opener = urllib.request.build_opener()
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(TEST_URL, timeout=TIMEOUT) as resp:
            _report_result(tag, honor_proxy, True,
                           f"HTTP {resp.status} {resp.reason}")
    except Exception as e:
        _report_result(tag, honor_proxy, False,
                       f"{type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
