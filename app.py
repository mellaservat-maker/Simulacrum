# ============================================================
# app.py · [021] Streamlit 可视化终端（挂载 V6.2 主链路）
#
# 与 chat_cli_v6.py 共用同一套链路调用：直接 import 其 chat_turn
#   （fixture 注入 + pipeline.run_turn encoder 模式），底层公式零重写、
#   零漂移——compute_danger_score、极性一致性过滤、threat_now 汇流口、
#   alpha 迟滞机全部是主树代码本体在执行。
#
# 状态保持：Streamlit 每次交互重跑脚本，跨轮状态（疲劳 w_PFC、记忆包、
#   E_prev 等）全部保存在 st.session_state["st_core"]，刷新不丢。
#
# 启动：
#   双击 启动网页版.bat
#   或命令行：D:\aiqg\venv\Scripts\streamlit run app.py
# ============================================================

import os
import sys
import time

import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

from state import reset_state                      # noqa: E402
from llm_client import (token_stats, token_reset,   # noqa: E402
                        LLMUnavailableError)
from chat_cli_v6 import chat_turn                   # noqa: E402

st.set_page_config(page_title="情绪仿生系统 v6.2", page_icon="🧠", layout="wide")


# ---------------- 状态初始化 ----------------

def _init_session():
    """首次进入时建会话状态；跨交互复用（Streamlit rerun 不清 session_state）。"""
    if "st_core" not in st.session_state:
        st.session_state.st_core = reset_state(seed=42)
        st.session_state.messages = []      # [{role, content, meta}]
        st.session_state.last_log = None
        st.session_state.turn_tokens = []   # 每轮 token
        token_reset()


_init_session()


def _reset_all():
    st.session_state.clear()
    token_reset()
    _init_session()


# ---------------- 侧边栏：情绪雷达 + 计费表 ----------------

def _alpha_label(a):
    if a >= 0.60:
        return "攻击/PTSD 档"
    if a <= 0.15:
        return "安全/脱敏档"
    return "过渡区"


def _render_sidebar():
    with st.sidebar:
        st.title("🧠 情绪雷达")
        log = st.session_state.last_log
        if log is None:
            st.info("还没有对话。发一句话，这里会实时显示内部状态。")
        else:
            e0 = float(log["E_final"][0]) if hasattr(log["E_final"], "__len__") else float(log["E_final"])
            wp = float(log["w_PFC"])
            ii = float(log["I_inhibit"])
            th = float(log.get("threat_now", 0.0))
            al = float(log.get("alpha_mode", 0.0))

            c1, c2 = st.columns(2)
            c1.metric("当前情绪 E[0]", "%+.3f" % e0, help="负=难受 / 正=舒服")
            c2.metric("本轮动作", str(log.get("action", "-")))

            st.caption("疲劳值 w_PFC（1.0=防线极限）")
            st.progress(min(max(wp, 0.0), 1.0), text="%.3f" % wp)
            st.caption("理智防线 I_inhibit（1.0=满血）")
            st.progress(min(max(ii, 0.0), 1.0), text="%.3f" % ii)
            st.caption("威胁感 threat_now")
            st.progress(min(max(th, 0.0), 1.0), text="%.3f" % th)
            st.caption("模式 alpha（%s）" % _alpha_label(al))
            st.progress(min(max(al, 0.0), 1.0), text="%.3f" % al)

            rm = st.session_state.st_core.get("recalled_memory")
            if isinstance(rm, dict) and rm.get("recall_source") not in (None, "package_level"):
                st.caption("召回：%s  sim=%.3f  ev=%.2f"
                           % (rm.get("emotion_polarity", "-"),
                              float(rm.get("matched_sim", -1.0)),
                              float(rm.get("emotion_value", 0.0))))
            else:
                st.caption("召回：无")
            st.caption("记忆：缓冲 %d 条 | 归档 %d 包"
                       % (len(st.session_state.st_core["memory_buffer"]),
                          len(st.session_state.st_core["archiver"].get_packages())))

        st.divider()

        # ---- 计费表 ----
        st.title("💰 Token")
        stats = token_stats()
        turn = st.session_state.last_log.get("turn_tokens", 0) if st.session_state.last_log else 0
        c1, c2 = st.columns(2)
        c1.metric("本轮消耗", "%d" % int(turn))
        c2.metric("会话累计", "%d" % int(stats["total_tokens"]))
        st.caption("拆分：prompt %d / completion %d / 调用 %d 次"
                   % (stats["prompt_tokens"], stats["completion_tokens"],
                      stats["calls"]))
        if st.session_state.turn_tokens:
            st.caption("近 10 轮：" + " → ".join(str(t) for t in st.session_state.turn_tokens[-10:]))

        st.divider()
        if st.button("🔄 重置会话", use_container_width=True):
            _reset_all()
            st.rerun()

        st.caption("输入 /state 在聊天里查看详细数据")


# ---------------- /state 详细面板 ----------------

def _state_markdown():
    st_core = st.session_state.st_core
    log = st.session_state.last_log
    if log is None:
        return "**还没有对话轮次**——先聊一句，再看 /state。"
    e0 = float(log["E_final"][0]) if hasattr(log["E_final"], "__len__") else float(log["E_final"])
    det = (st_core.get("recalled_memory") or {}).get("recall_detail") or {}
    stats = token_stats()
    lines = [
        "**内部状态（上帝视角）**",
        "",
        "| 指标 | 值 | 判读 |",
        "|---|---|---|",
        "| 轮次 | %d | — |" % st_core.get("round_counter", 0),
        "| 当前情绪 E[0] | %+.4f | 负=难受 / 正=舒服 |" % e0,
        "| 疲劳值 w_PFC | %.4f | 1.0=防线极限 |" % float(log["w_PFC"]),
        "| 理智防线 I | %.4f | 1.0=满血 |" % float(log["I_inhibit"]),
        "| 威胁感 threat | %.4f | max(上轮D_error, HR, -S_val) |" % float(log.get("threat_now", 0.0)),
        "| 模式 alpha | %.4f | %s |" % (float(log.get("alpha_mode", 0.0)),
                                        _alpha_label(float(log.get("alpha_mode", 0.0)))),
        "| 本轮动作 | %s | — |" % log.get("action", "-"),
        "| 本轮 Tokens | %d | 本会话累计 %d |"
        % (int(log.get("turn_tokens", 0)), int(stats["total_tokens"])),
    ]
    if det:
        lines.append("| 召回中间量 | H_eff=%.3f alpha=%.3f factor=%+.3f | — |"
                     % (float(det.get("H_eff", 0)), float(det.get("alpha", 0)),
                        float(det.get("factor", 0))))
    lines += ["", "记忆：缓冲 %d 条 | 归档 %d 包"
              % (len(st_core["memory_buffer"]),
                 len(st_core["archiver"].get_packages()))]
    return "\n".join(lines)


# ---------------- 打字机效果 ----------------

def _typewriter(text, delay=0.012):
    """纯前端打字机（不改变链路：底层仍是一次性同步生成）。"""
    step = 2
    for i in range(0, len(text), step):
        yield text[i:i + step]
        time.sleep(delay)


# ---------------- 主区域 ----------------

st.title("🧠 情绪仿生系统 · 对话")
st.caption("V6.2 主链路 · 复合危险度 + 极性一致性过滤 · 输入 /state 看上帝视角")

_render_sidebar()

# 历史消息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 新输入
prompt = st.chat_input("说点什么…（/state 看详细状态）")
if prompt:
    text = prompt.strip()
    if not text:
        st.stop()

    # /state 指令：不发 API，直接在聊天流里出面板
    if text.lower().startswith("/state"):
        st.session_state.messages.append(
            {"role": "user", "content": "/state"})
        st.session_state.messages.append(
            {"role": "assistant", "content": _state_markdown(), "meta": "state"})
        st.rerun()

    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant"):
        try:
            with st.spinner("思考中..."):
                log, S, src = chat_turn(st.session_state.st_core, text)
            reply = str(log.get("llm_output", "")).strip()
            st.write_stream(_typewriter(reply))
            st.caption("E=%+.3f | w_PFC=%.3f | I=%.3f | alpha=%.3f | %s | Token: %d"
                       % (float(log["E_final"][0]), float(log["w_PFC"]),
                          float(log["I_inhibit"]), float(log.get("alpha_mode", 0.0)),
                          log.get("action", "-"), int(log.get("turn_tokens", 0))))
            st.session_state.last_log = log
            st.session_state.turn_tokens.append(int(log.get("turn_tokens", 0)))
            st.session_state.messages.append(
                {"role": "user", "content": text})
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "meta": "chat"})
        except LLMUnavailableError as e:
            st.warning("本轮上游通道异常，已降级跳过：%s" % e)
            st.caption("若持续出现：请检查余额/密钥/网络。")
        except Exception as e:
            st.error("本轮出错：%s: %s" % (type(e).__name__, e))
            st.caption("可以继续输入下一句。")
