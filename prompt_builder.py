# ============================================================
# prompt_builder.py · V4.2 · 模块11b Prompt 组装
# 把 state_translator 的心理独白 + 用户输入拼成最终 prompt
# [023] Action 闭环（清单问题1）：行动指令紧贴用户输入放置
#   （recency 位置，LLM 遵守率最高），与 state_translator 内嵌的
#   规则第 6 条形成双重保险。
# [055] 对话历史：build_prompt 仍产出单条 prompt 字符串（兼容旧调用与探针），
#   新增 build_messages 把历史拼成多轮 messages 数组；
#   collect_history 负责从 raw_log + memory_buffer 取最近 N 轮并按 token 截断。
# ============================================================

import config

from state_translator import action_directive


def build_prompt(internal_monologue, user_input, action="task_first"):
    directive = action_directive(action)
    if directive:
        return (f"{internal_monologue}\n\n"
                f"【行动指令（最高优先级，必须执行）】{directive}\n\n"
                f"用户输入：{user_input}")
    return f"{internal_monologue}\n\n用户输入：{user_input}"


# ---- [055 任务1] 对话历史 ----

def _is_cjk(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)


def estimate_tokens(text):
    """粗略 token 估算：CJK 字 ≈ 1 token，其余 ≈ 0.3 token。

    宁可高估（少带历史）也不低估（撑爆上下文）。真实 token 数以
    API 返回的 usage 为准，这里只用于预算裁剪。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(cjk * config.HISTORY_CJK_TOKEN_RATIO
               + other * config.HISTORY_ASCII_TOKEN_RATIO) + 1


def collect_history(state, max_turns=None, max_tokens=None):
    """取最近 N 轮问答，按 token 预算从新到旧截断。

    数据来源（两处必须都取，单取任一处都会漏）：
      raw_log       已归档轮（归档时 flushed 写入）
      memory_buffer 未归档轮（含最近 2 轮，flush_keep_recent 留下的）
    按 round_id 合并去重后升序排列，取末尾 max_turns 轮。

    返回: [{"role": "user"/"assistant", "content": str}, ...]（时间正序）
    降级：state 结构异常、无历史 → 返回 []（调用方按无历史处理，不崩）。
    """
    max_turns = config.HISTORY_MAX_TURNS if max_turns is None else int(max_turns)
    max_tokens = config.HISTORY_MAX_TOKENS if max_tokens is None else int(max_tokens)
    if max_turns <= 0 or max_tokens <= 0:
        return []

    recs = {}
    try:
        raw = state.get("raw_log")
        if raw is not None:
            for rid in raw.all_ids():
                r = raw.get(rid)
                if isinstance(r, dict):
                    recs[int(rid)] = r
    except Exception:
        pass
    try:
        buf = state.get("memory_buffer")
        if buf is not None:
            for r in buf.get_all():
                if isinstance(r, dict) and r.get("round_id") is not None:
                    recs[int(r["round_id"])] = r
    except Exception:
        pass
    if not recs:
        return []

    ordered = [recs[k] for k in sorted(recs)][-max_turns:]

    # 从最近往前累加 token，超预算就停（保证最近的那几轮一定在里面）
    picked = []
    used = 0
    for r in reversed(ordered):
        u = str(r.get("user_input", "") or "")
        a = str(r.get("ai_output", "") or "")
        cost = estimate_tokens(u) + estimate_tokens(a)
        if picked and used + cost > max_tokens:
            break
        picked.append((u, a))
        used += cost
    picked.reverse()

    msgs = []
    for u, a in picked:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs


def build_messages(internal_monologue, user_input, action="task_first",
                   history=None):
    """把历史 + 本轮 prompt 拼成多轮 messages 数组。

    结构：
      [ 历史 user / assistant ... ] + [ 本轮 user = 独白 + 行动指令 + 当前输入 ]
    独白与指令仍放在**最后一条 user 消息**里，所以状态注入方式与 023 一致，
    只是模型现在能看到前面聊了什么。
    """
    msgs = []
    for h in (history or []):
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": str(content)})
    msgs.append({"role": "user",
                 "content": build_prompt(internal_monologue, user_input, action)})
    return msgs
