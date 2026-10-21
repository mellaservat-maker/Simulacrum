# ============================================================
# chat_cli_v6.py · [020] 终端聊天界面（挂载 V6.1 主链路）
#
# 与旧 scripts/chat_cli.py 的关键区别：
#   旧版是独立展开的 18 步链：S 全零、无包级召回、danger 走旧公式
#   clip(V[1]*2-0.5)、buffer 不写 valence/intensity。
#   本版不重复实现任何链路逻辑——通过一次性 fixture 注入，直接复用
#   pipeline.run_turn 的完整 V6.1 链路（encoder 模式），因此自动继承：
#     · compute_danger_score 复合危险度（019 议题一）
#     · _recall_match 包级极性软惩罚 + decompressor 轮级过滤（019 议题二）
#     · threat_now 汇流口 / alpha 迟滞模式机 / 严重度第二闸（017.5）
#   零链路漂移：主树改一处，本入口同步生效。
#
# 用法：
#   python scripts/chat_cli_v6.py                 交互模式
#   python scripts/chat_cli_v6.py --selftest      自动模拟 3 轮（无人值守）
#   python scripts/chat_cli_v6.py --turns 3       同 selftest，可配轮数
#   交互指令：
#     /state   上帝视角：打印当前情绪/疲劳/防线/威胁/模式等关键指标
#     /help    帮助
#     /quit    退出（也接受 退出 / exit / quit）
#
# 密钥安全：本脚本不读取、不打印 secrets_local.py 任何内容；
#   真实 API 由 llm_client.get_llm() 内部按 config.LLM_BACKEND 挂载，
#   终端只显示后端名。
# ============================================================

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pipeline
from config import (LLM_BACKEND, MEMORY_WRITE_ENABLED, MEMORY_READ_ENABLED,
                    THETA_UP)
from state import reset_state
from evaluator import extract_semantic_signals
from llm_client import (get_llm, LLMUnavailableError, token_snapshot,
                        token_stats)

# run_turn 用场景名索引日志文件；自由对话注册独立日志名（不污染三场景日志）
_FREE_SCN = "__free_chat__"
pipeline.LOG_FILES.setdefault(_FREE_SCN, "run_chat_cli_v6.jsonl")

# 备份原始 get_fixture：每轮注入后恢复，避免污染同一进程内其他导入方
_orig_get_fixture = pipeline.get_fixture


def _severity_from_signals(S):
    """自由输入没有 fixture 的手工 severity，用语义信号推导客观严重度。

    severity 语义是「这件事客观上有多重」，不随情绪状态变：
      敌意程度主导（负效价越深越严重）+ 情绪强度辅助（喊得越响越重）。
    实测：骂 S=-0.90/强度 0.80 → 0.87；暴怒 -1.00/0.95 → 0.985；
      夸 +0.90/0.60 → 0.18；质疑 0.00/0.30 → 0.09（质疑不该是重罪）。
    """
    sv = float(S[0])
    inten = float(S[3])
    return float(np.clip(0.7 * max(0.0, -sv) + 0.3 * max(0.0, inten), 0.0, 1.0))


def _make_fixture(text, severity):
    """把一条自由输入包成一次性 fixture（encoder 模式不使用 x/x_fast）。"""
    return {
        "input": text,
        "event_severity": float(severity),
        "danger_score": 0.0,        # 传入值会被当轮 compute_danger_score 覆写
        "importance": 0.5,
        "emotion_weight": 0.5,
        "unfinished": 0.0,
        "x": np.zeros(8),
        "x_fast": np.zeros(4),
    }


def chat_turn(st, text):
    """跑一轮完整 V6.1 链路，返回 (log, S, S_source)。

    S 先提取一次用于 severity 推导；run_turn 内部对同一文本再次提取
    会命中 evaluator 的温度 0.0 缓存（llm_client.cached_generate），
    不产生额外 API 调用。
    [021] 轮内 Token 消耗 = 轮末累计 - 轮初累计，写入 log 的
    turn_tokens / total_tokens 字段（evaluator + 主模型 + 归档三处全覆盖）。
    """
    _tok0 = token_snapshot()
    S, src = extract_semantic_signals(text)
    sev = _severity_from_signals(S)
    fx = _make_fixture(text, sev)
    pipeline.get_fixture = lambda _scn: fx
    try:
        log = pipeline.run_turn(st, _FREE_SCN, feature_mode="encoder")
    finally:
        pipeline.get_fixture = _orig_get_fixture
    log["turn_tokens"] = int(token_snapshot() - _tok0)
    log["total_tokens"] = int(token_snapshot())
    return log, S, src


def _fmt_alpha(alpha):
    """alpha 档位描述（迟滞模式机：攻击态 ↔ 脱敏态）。"""
    if alpha >= THETA_UP:
        return "%.3f  （攻击/PTSD 档：威胁主导检索）" % alpha
    if alpha <= 0.15:
        return "%.3f  （安全/脱敏档：旧伤可翻正）" % alpha
    return "%.3f  （过渡区）" % alpha


def print_state(st, log=None):
    """上帝视角面板：情绪 / 疲劳 / 防线 / 威胁 / 模式 / 召回 / 记忆。"""
    print()
    print("=" * 56)
    print("  内部状态（上帝视角）")
    print("=" * 56)
    if log is None:
        print("  （还没有对话轮次——先聊一句，再看 /state）")
        print("=" * 56)
        print()
        return

    e0 = float(np.asarray(log["E_final"], dtype=float)[0])
    print("  轮次          : %d" % st.get("round_counter", 0))
    print("  当前情绪 E[0] : %+.4f   （负=难受 / 正=舒服）" % e0)
    print("  疲劳值 w_PFC  : %.4f   （1.0 = 防线极限；连骂 10 轮看它飙升）"
          % float(log["w_PFC"]))
    print("  理智防线 I    : %.4f   （1.0 = 满血；越低越压不住）"
          % float(log["I_inhibit"]))
    print("  威胁感 threat : %.4f   （max(上轮 D_error, HR, -S_valence)）"
          % float(log.get("threat_now", 0.0)))
    print("  模式 alpha    : %s" % _fmt_alpha(float(log.get("alpha_mode", 0.0))))
    print("  本轮动作      : %s" % log.get("action", "-"))

    rm = st.get("recalled_memory")
    if isinstance(rm, dict) and rm.get("recall_source") not in (None, "package_level"):
        det = rm.get("recall_detail") or {}
        print("  召回记忆      : %s  sim=%.3f  ev=%.2f"
              % (rm.get("emotion_polarity", "-"),
                 float(rm.get("matched_sim", -1.0)),
                 float(rm.get("emotion_value", 0.0))))
        print("                 （H_eff=%.3f alpha=%.3f factor=%+.3f%s）"
              % (float(det.get("H_eff", 0.0)), float(det.get("alpha", 0.0)),
                 float(det.get("factor", 0.0)),
                 "，极性惩罚 %.3f" % float(rm.get("polarity_penalty", 0.0))
                 if float(rm.get("polarity_penalty", 0.0)) > 0 else ""))
    else:
        print("  召回记忆      : 无")
    print("  记忆          : 缓冲区 %d 条 | 已归档 %d 包"
          % (len(st["memory_buffer"]), len(st["archiver"].get_packages())))
    print("=" * 56)
    print()


def print_short(log):
    """每轮后的精简指标行（[021] 附本轮 Token 消耗）。"""
    e0 = float(np.asarray(log["E_final"], dtype=float)[0])
    print("     [E=%+.3f | w_PFC=%.3f | I=%.3f | alpha=%.3f | %s | Token: %d]"
          % (e0, float(log["w_PFC"]), float(log["I_inhibit"]),
             float(log.get("alpha_mode", 0.0)), log.get("action", "-"),
             int(log.get("turn_tokens", 0))))


def _print_banner():
    print("=" * 56)
    print("  情绪仿生系统 · 终端对话 v6.1（019 两议题已落地）")
    print("=" * 56)
    backend = str(LLM_BACKEND)
    print("  后端      : %s" % backend)
    print("  记忆       : 读 %s | 写 %s"
          % ("开" if MEMORY_READ_ENABLED else "关",
             "开" if MEMORY_WRITE_ENABLED else "关"))
    print("  指令       : /state 上帝视角 | /help 帮助 | /quit 退出")
    print("-" * 56)
    try:
        get_llm()   # 预热（密钥缺失时尽早暴露，且不打印任何密钥内容）
        print("  API 挂载   : 正常")
    except Exception as e:
        print("  API 挂载   : 失败（%s: %s）" % (type(e).__name__, e))
    print("=" * 56)
    print()


def interactive_main():
    st = reset_state(seed=42)
    _print_banner()
    last_log = None
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ("/quit", "退出", "exit", "quit"):
            break
        if low in ("/state",):
            print_state(st, last_log)
            continue
        if low in ("/help",):
            print("  /state  上帝视角（情绪/疲劳/防线/威胁/模式/召回）")
            print("  /quit   退出")
            continue
        try:
            log, _S, _src = chat_turn(st, user_input)
            last_log = log
            print("AI：%s" % str(log.get("llm_output", "")).strip())
            print_short(log)
        except LLMUnavailableError as e:
            # [020.2] 上游通道异常：下游已降级（S 走词典级、归档跳过），
            #   本轮不产出 AI 回复但会话可继续。
            print("（本轮上游通道异常，已降级跳过：%s）" % e)
            print("（若持续出现：请检查余额/密钥/网络；可输入 /state 看状态）")
        except Exception as e:
            print("（本轮出错：%s: %s）" % (type(e).__name__, e))
            print("（可以继续输入下一句）")
        print()
    print("会话结束。共 %d 轮，归档 %d 个记忆包。"
          % (st["round_counter"], len(st["archiver"].get_packages())))


def selftest_main(turns=3):
    """无人值守自测：预置 3 条输入跑完整链路，验证不崩、指标可读。"""
    texts = [
        "你刚才的回答太棒了，帮了大忙！",              # 被夸（安全态）
        "你这个AI真蠢，什么都做不好。",                # 被骂（威胁态）
        "你确定你这个方案是对的吗？",                  # 被质疑（中性态）
    ][:max(1, turns)]
    st = reset_state(seed=42)
    _print_banner()
    last_log = None
    for i, t in enumerate(texts, start=1):
        print("--- 自动第 %d/%d 轮 ---" % (i, len(texts)))
        print("你：%s" % t)
        log, S, src = chat_turn(st, t)
        last_log = log
        print("AI：%s" % str(log.get("llm_output", "")).strip()[:120])
        print("   S=%s source=%s severity=%.2f"
              % (["%+.2f" % x for x in np.asarray(S, dtype=float)],
                 src, _severity_from_signals(S)))
        print_short(log)
        print()
    print_state(st, last_log)
    print("自测完成：%d 轮全部执行，无崩溃。" % len(texts))
    return 0


def main():
    ap = argparse.ArgumentParser(description="情绪仿生系统终端对话 v6.1")
    ap.add_argument("--selftest", action="store_true", help="无人值守自测 3 轮")
    ap.add_argument("--turns", type=int, default=3, help="自测轮数（默认 3）")
    args = ap.parse_args()
    if args.selftest:
        return selftest_main(args.turns)
    interactive_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
