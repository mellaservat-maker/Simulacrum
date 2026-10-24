# ============================================================
# state_translator.py · V4.2 · 模块11 状态翻译
# 规格：5.5.13
# [023] Action 闭环（清单问题1）：policy_minimal 算出的 action 必须
#   进入 Prompt——旧版 action 只落在 log 里，LLM 完全没听话。
# [056 任务3] 关系状态注入：按 trust 四档（破裂/淡漠/熟人/普通）
#   追加一行关系独白。relationship=None 时行为与旧版逐字一致
#   （旧调用点与黄金轨迹不受影响）。
# [058 任务2] 亲密度注入：按 intimacy 两档（>0.7 认识很久 /
#   <0.2 还很防备）追加一行独白。relationship=None 或中间档时
#   不出现该行；与关系行皆空时 prompt 与 056 版逐字一致。
# ============================================================

import config
from relationship import relationship_text, intimacy_text

# [023] 行动策略强指令表（与 policy.py 的优先级对应：
#   ask_clarify > suppress > reappraise > express > task_first）。
#   task_first 无附加指令——默认行为不变，旧探针/CLI 输出不受影响。
ACTION_DIRECTIVES = {
    "suppress": "强制要求：保持极简回复，字数不超过 20 字，禁止解释和反问。",
    "express": "允许展现第一人称情绪，允许句子碎片化。",
    "reappraise": "回复中必须包含重新思考、确认含义的措辞。",
    "ask_clarify": "必须提出一个问题。",
    "task_first": None,
}


def action_directive(action):
    """取 action 对应的强指令文本；task_first/未知 action 返回 None。"""
    return ACTION_DIRECTIVES.get(action)


def state_to_prompt(E_final, I_inhibit, M_text, HistoricalResidual, HabituationSignal,
                    action="task_first", relationship=None):
    # 感受（自然语言，不带数字）
    if E_final[0] < config.PROMPT_VALENCE_THRESHOLD:
        feeling_text = "有点被冒犯，心里不太舒服"
    elif E_final[0] > config.PROMPT_POSITIVE_THRESHOLD:
        feeling_text = "心情不错，挺开心的"
    else:
        feeling_text = "没什么特别的波动"

    # 自控（自然语言，不带数字）
    if I_inhibit > config.PROMPT_INHIBIT_THRESHOLD:
        control_text = "你在努力克制，不想让情绪跑出来"
    else:
        control_text = "你没怎么在压制自己"

    # 熟悉度（自然语言，不带数字）
    if HabituationSignal > config.PROMPT_HABITUATION_THRESHOLD:
        familiarity_text = "这种情况你碰到过几次了，有点麻木"
    else:
        familiarity_text = "这件事对你来说还比较新鲜"

    # 隐约的回忆：只有真的包含"记得"才算有记忆
    if M_text and "记得" in M_text:
        memory_text = "你隐约想起之前也遇到过类似的事"
    else:
        memory_text = "脑子里没什么特别的念头"

    # [056 任务3] 关系（自然语言，不带数字）：trust 四档文案由
    #   relationship.relationship_text 给出（破裂/淡漠/熟人/普通）。
    #   relationship=None（旧调用点/黄金轨迹）→ 整行不出现，
    #   prompt 与旧版逐字一致（下面的 \n 拼接保证空行不丢）。
    _rel_text = relationship_text(relationship)
    _rel_line = f"- 关系：{_rel_text}\n" if _rel_text else ""

    # [058 任务2] 亲密度（自然语言，不带数字）：intimacy 两档文案由
    #   relationship.intimacy_text 给出（认识很久/还很防备）。
    #   None（旧调用点/中间档）→ 整行不出现；{_rel_line}{_intim_line}
    #   拼接在两者皆空时与 056 版逐字一致。
    _intim_text = intimacy_text(relationship)
    _intim_line = f"- 亲密度：{_intim_text}\n" if _intim_text else ""

    # [023] 行动策略指令：嵌入输出规则（软引导，与 prompt_builder 的
    #   硬指令位形成双重保险——清单问题1 的病灶是 LLM 无视单点指令）
    _rules = """【输出规则（必须严格遵守）】
1. 直接回应用户，用第一人称说话。
2. 禁止描述自己的内心状态。禁止使用括号、旁白、内心独白。
3. 禁止提及任何数字、分数、参数名。
4. 你的状态应当自然体现在语气、长短、用词上，而不是被说出来。
5. 若"自控"为克制状态，回应要短、要收着；若"感受"为积极状态，回应可以稍放开。"""
    _directive = action_directive(action)
    if _directive:
        _rules += f"\n6. 行动策略[{action}]：{_directive}"

    return f"""【你此刻的状态（仅供你参考，不要说出口）】
- 感受：{feeling_text}
- 自控：{control_text}
- 熟悉度：{familiarity_text}
- 隐约的回忆：{memory_text}
{_rel_line}{_intim_line}
{_rules}"""
