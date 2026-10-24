# ============================================================
# relationship.py · [056 任务2] 关系状态动力学
#
# 定位：给无状态的情绪系统外挂一层"关系账本"——trust/intimacy
#   跨轮累积，让"被同一个人骂"与"被陌生人骂"在数值上分道扬镳。
#
# 与既有模块的分工（边界写死，防职责漂移）：
#   event_parser  词面 → 稳定实体 ID（entity_ids），只映射不动力学；
#   relationship  本模块——账本的读写与效应换算，不碰任何情绪/记忆
#                 数值，只输出增益系数（gain）给 pipeline 应用；
#   pipeline      唯一接线点：读 gain → 应用到 H/D_slow；
#                 轮末调 update_relationship 写账本。
#
# 顺序约束（防循环依赖）：
#   熟人效应读取的是"上一轮结转"的 trust（轮末才更新），因此
#   本轮攻击要先作用到情绪上，下一轮才体现为"更麻木/想更多"。
#   这不是缺陷而是设计：人对关系的感知本就滞后于事件。
#
# M1 明确边界（所有者拍板"先支持当前用户和他人"）：
#   · 动力学只挂 user_001（当前用户）；ai_self / boss_01 等只计数；
#   · intimacy 只升不降（攻击不降亲密度）；
#   · 无共指消解（"他"永远是 other_01），无跨会话关系迁移；
#   · 全部参数 [未校准]。
# ============================================================

import config


def _default_entry():
    """新实体条目。trust 用 config.REL_TRUST_INIT（与 user_001 同起点）。"""
    return {
        "trust": float(config.REL_TRUST_INIT),
        "intimacy": 0.5,
        "interactions": 0,
        "broken": False,
    }


def _clip01(x):
    return min(1.0, max(0.0, float(x)))


def ensure_entity(rels, entity_id):
    """账本里确保有该实体（幂等）。返回该实体条目（dict 引用）。"""
    if entity_id not in rels or not isinstance(rels[entity_id], dict):
        rels[entity_id] = _default_entry()
    return rels[entity_id]


def update_relationship(st, entity_ids, danger_now, S_valence):
    """轮末调用：按本轮事件更新关系账本（就地修改 st["relationships"]）。

    两步，顺序固定：
      1) 互动计数——entity_ids 里每个实体 interactions += 1
         （含 ai_self 与话题里提到的他人，计数不影响任何数值）；
      2) 动力学——只对 user_001：
           攻击（danger_now > 0）：trust -= 0.02 × danger_now
             乘 danger 而非固定量：恶意辱骂（≈0.96）一轮掉 0.019，
             轻度质疑（≈0.12）只掉 0.0024——"骂得多重"直接决定
             "伤得多深"，这与 compute_danger_score 的复合口径一致；
           安抚（S_valence > 0）：trust += 0.01
             修复速率取攻击的一半量级：信任毁起来快、建起来慢；
           intimacy：每轮 +0.01（只升不降，M1 规格）；
           broken：[058 任务2] 施密特触发器——跌破 0.2 置 True，
             涨回 0.3 才解除 False，迟滞带 [0.2, 0.3) 内保持原状
             （防 trust 在 0.2 附近横跳时 broken 抖动）。

    降级：entity_ids 为 None/损坏 → 只更新 user_001 计数，不崩。
    """
    rels = st.setdefault("relationships", {})
    user = ensure_entity(rels, "user_001")

    # 1) 互动计数（所有被提及实体；他人条目只计数不动力学）
    for eid in (entity_ids or []):
        if not isinstance(eid, str) or not eid:
            continue
        ensure_entity(rels, eid)["interactions"] += 1
    user["interactions"] += 1  # 用户本人在场，即使没提"我"也算一次互动

    # 2) 动力学（仅 user_001）
    try:
        danger = max(0.0, float(danger_now))
        valence = float(S_valence)
    except (TypeError, ValueError):
        danger, valence = 0.0, 0.0

    trust = _clip01(user["trust"]
                    - float(config.REL_TRUST_ATTACK_COEF) * danger
                    + (float(config.REL_TRUST_PRAISE_GAIN) if valence > 0.0
                       else 0.0))
    user["trust"] = trust
    user["intimacy"] = _clip01(user["intimacy"]
                               + float(config.REL_INTIMACY_GAIN))
    # [058 任务2] broken 施密特触发器：跌破 REL_TRUST_BREAK 置 True，
    #   涨回 REL_TRUST_RELEASE 才解除 False；迟滞带
    #   [REL_TRUST_BREAK, REL_TRUST_RELEASE) 内保持上一轮的值。
    #   读上一轮的 broken 实现迟滞：band 内 trust 一升一降两个来回，
    #   broken 也保持原状，不会每轮翻转。
    if trust < float(config.REL_TRUST_BREAK):
        user["broken"] = True
    elif trust >= float(config.REL_TRUST_RELEASE):
        user["broken"] = False
    else:
        user["broken"] = bool(user.get("broken", False))
    return user


def acquaintance_effects(trust):
    """熟人效应换算：trust → (d_slow_gain, h_gain)。

      trust > 0.7（熟人）  → (1.5, 1.0)  慢通路介入放大：
                              "熟人面前想得多"——信任高时每句话都值得斟酌；
      trust < 0.3（淡漠）  → (1.0, 1.5)  H 当轮放大：
                              "麻了"——信任见底时同类攻击更快脱敏；
      中间区               → (1.0, 1.0)  不干预（普通熟人/普通陌生人）。
    """
    try:
        t = float(trust)
    except (TypeError, ValueError):
        t = float(config.REL_TRUST_INIT)
    if t > float(config.REL_ACQUAINT_HI):
        return float(config.REL_D_SLOW_GAIN), 1.0
    if t < float(config.REL_ACQUAINT_LO):
        return 1.0, float(config.REL_H_GAIN)
    return 1.0, 1.0


def user_trust(st):
    """取当前用户的 trust（账本缺失/损坏 → 默认 0.5，不崩）。"""
    rels = st.get("relationships") or {}
    u = rels.get("user_001")
    if not isinstance(u, dict):
        return float(config.REL_TRUST_INIT)
    try:
        return _clip01(u.get("trust", config.REL_TRUST_INIT))
    except (TypeError, ValueError):
        return float(config.REL_TRUST_INIT)


def relationship_text(rel):
    """[056 任务3] 供 state_translator 的自然语言独白（不带数字）。

    rel 为 user_001 条目（dict）或 None。四档：
      破裂（trust<0.2） → 伤口措辞（不想再经营这段关系）；
      淡漠（trust<0.3） → 麻木措辞（他说什么随他便）；
      熟人（trust>0.7） → 信任措辞（可以坦诚直接）；
      中间              → 普通措辞（客气、有分寸）。
    """
    if not isinstance(rel, dict):
        return None
    try:
        trust = float(rel.get("trust", 0.5))
    except (TypeError, ValueError):
        return None
    if trust < float(config.REL_TRUST_BREAK):
        return "你和这个人之间的关系已经碎了，不太想再经营，回应变得敷衍"
    if trust < float(config.REL_ACQUAINT_LO):
        return "你对这个人已经无所谓了，他说什么都很难再让你放在心上"
    if trust > float(config.REL_ACQUAINT_HI):
        return "你们之间已经有信任，面对他你可以更坦诚、更直接"
    return "你们还算客气，保持着普通的分寸"


def intimacy_text(rel):
    """[058 任务2] 供 state_translator 的亲密度独白（不带数字）。

    rel 为 user_001 条目（dict）或 None。两档：
      intimacy > REL_INTIMACY_HI（0.7） → 认识很久措辞；
      intimacy < REL_INTIMACY_LO（0.2） → 还很防备措辞；
      中间                                → None（独白不出现该行）。

    与 relationship_text 正交：trust 说"关系质量"（破裂/淡漠/熟人/
    普通），intimacy 说"认识多久、多深"——同一个人可以既值得信任
    （trust 高）又认识很久（intimacy 高），两条线各说各话。

    注意：intimacy 起点 0.5 且只升不降，LO 档正常动力学不可达
    （仅旧存档/构造态可触发）；HI 档约 20 轮互动可达。
    """
    if not isinstance(rel, dict):
        return None
    try:
        intimacy = float(rel.get("intimacy", 0.5))
    except (TypeError, ValueError):
        return None
    if intimacy > float(config.REL_INTIMACY_HI):
        return "你们已经认识很久了，你对他很熟悉。"
    if intimacy < float(config.REL_INTIMACY_LO):
        return "你对这个人还很防备。"
    return None
