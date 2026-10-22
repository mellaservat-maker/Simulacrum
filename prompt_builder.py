# ============================================================
# prompt_builder.py · V4.2 · 模块11b Prompt 组装
# 把 state_translator 的心理独白 + 用户输入拼成最终 prompt
# [023] Action 闭环（清单问题1）：行动指令紧贴用户输入放置
#   （recency 位置，LLM 遵守率最高），与 state_translator 内嵌的
#   规则第 6 条形成双重保险。
# ============================================================

from state_translator import action_directive


def build_prompt(internal_monologue, user_input, action="task_first"):
    directive = action_directive(action)
    if directive:
        return (f"{internal_monologue}\n\n"
                f"【行动指令（最高优先级，必须执行）】{directive}\n\n"
                f"用户输入：{user_input}")
    return f"{internal_monologue}\n\n用户输入：{user_input}"