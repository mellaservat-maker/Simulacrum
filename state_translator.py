# ============================================================
# state_translator.py · V4.2 · 模块11 状态翻译
# 规格：5.5.13
# ============================================================

from config import (
    PROMPT_VALENCE_THRESHOLD,
    PROMPT_POSITIVE_THRESHOLD,
    PROMPT_INHIBIT_THRESHOLD,
    PROMPT_HABITUATION_THRESHOLD,
)


def state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal):
    # 感受（自然语言，不带数字）
    if E_final[0] < PROMPT_VALENCE_THRESHOLD:
        feeling_text = "有点被冒犯，心里不太舒服"
    elif E_final[0] > PROMPT_POSITIVE_THRESHOLD:
        feeling_text = "心情不错，挺开心的"
    else:
        feeling_text = "没什么特别的波动"

    # 自控（自然语言，不带数字）
    if I_inhibit > PROMPT_INHIBIT_THRESHOLD:
        control_text = "你在努力克制，不想让情绪跑出来"
    else:
        control_text = "你没怎么在压制自己"

    # 熟悉度（自然语言，不带数字）
    if HabituationSignal > PROMPT_HABITUATION_THRESHOLD:
        familiarity_text = "这种情况你碰到过几次了，有点麻木"
    else:
        familiarity_text = "这件事对你来说还比较新鲜"

    # 隐约的回忆：只有真的包含"记得"才算有记忆
    if M_text and "记得" in M_text:
        memory_text = "你隐约想起之前也遇到过类似的事"
    else:
        memory_text = "脑子里没什么特别的念头"

    return f"""【你此刻的状态（仅供你参考，不要说出口）】
- 感受：{feeling_text}
- 自控：{control_text}
- 熟悉度：{familiarity_text}
- 隐约的回忆：{memory_text}

【输出规则（必须严格遵守）】
1. 直接回应用户，用第一人称说话。
2. 禁止描述自己的内心状态。禁止使用括号、旁白、内心独白。
3. 禁止提及任何数字、分数、参数名。
4. 你的状态应当自然体现在语气、长短、用词上，而不是被说出来。
5. 若"自控"为克制状态，回应要短、要收着；若"感受"为积极状态，回应可以稍放开。"""
