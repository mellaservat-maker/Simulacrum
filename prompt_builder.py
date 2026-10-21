# ============================================================
# prompt_builder.py · V4.2 · 模块11b Prompt 组装
# 把 state_translator 的心理独白 + 用户输入拼成最终 prompt
# ============================================================

def build_prompt(internal_monologue, user_input):
    return f"{internal_monologue}\n\n用户输入：{user_input}"