# ============================================================
# test_relationship.py · 056 号工作 · 任务 4（058 任务2 扩充）
#
# 覆盖关系状态（Relationship State）：
#   T056-A 实体映射：词面 → 稳定实体 ID（event_parser）
#   T056-B trust / intimacy / interactions 动力学（relationship）
#   T056-C 熟人效应换算与独白文案
#   T056-D Prompt 注入（state_translator，含"None 时与旧版逐字一致"）
#   T056-E 会话持久化往返与旧存档兼容（session_store）
#   T056-F 端到端：pipeline 一轮被骂后的账本与 log 观测字段
#   T056-G 熟人效应对数值链的作用（H 放大只当轮、不写回）
#   T058-H broken 施密特触发器（迟滞带不翻转）
#   T058-I intimacy 独白文案与 Prompt 注入
#   T058-J load 迟滞带保持存档值（带外仍重算）
#
# 端到端用例全部断网：假 LLM + 固定 S 表 + LLM_BACKEND="mock"
#   （与 test_golden_trajectory / test_dialogue_history 同规格）。
# ============================================================

import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config          # noqa: E402
import pipeline        # noqa: E402
import decompressor    # noqa: E402
import event_parser    # noqa: E402
import relationship    # noqa: E402
import session_store   # noqa: E402
from state import reset_state       # noqa: E402
from state_translator import state_to_prompt   # noqa: E402

pipeline.LOG_FILES.setdefault("__t056__", "run_t056_relationship.jsonl")

INSULT = "你这个AI真蠢，什么都做不好。"
INSULT_S = [-0.88, 0.72, 0.62, 0.82]


# ---- T056-A：实体映射 ----

def test_entity_map_self_and_others():
    """"我"→user_001；"老板"→boss_01；"他/她/它"→other_01；"大家"→others_group。"""
    assert "user_001" in event_parser._extract_entities("我昨天被骂了")
    assert "boss_01" in event_parser._extract_entities("老板又在骂人")
    assert "teacher_01" in event_parser._extract_entities("老师夸了我")
    for w in ("他", "她", "它"):
        assert event_parser._extract_entities("昨天%s来过" % w) == ["other_01"]
    assert event_parser._extract_entities("大家怎么看") == ["others_group"]
    # "你"/"AI" → ai_self（自己与自己无信任可言，只计数）
    assert event_parser._extract_entities("你真蠢") == ["ai_self"]
    assert event_parser._extract_entities("你这个AI") == ["ai_self"]


def test_entity_map_no_hit_and_stable():
    """无关句 → 空；同词多现 → 单 ID（去重）。"""
    assert event_parser._extract_entities("今天天气不错") == []
    assert event_parser._extract_entities("我我我说了三次") == ["user_001"]


def test_parse_writes_entity_ids():
    """parse 落值：entity_ids 与 persons 双口径并存（053 口径不动）。"""
    ev = event_parser.parse("老板说我是个废物")
    assert "boss_01" in ev.entity_ids and "user_001" in ev.entity_ids
    # persons 仍是词面集合（喂 feature_encoder 的 x1，056 不动它）
    assert "老板" in ev.persons and "我" in ev.persons


# ---- T056-B：动力学 ----

def _ledger():
    st = reset_state(seed=42)
    return st


def test_trust_attack_decay_scales_with_danger():
    """攻击：trust -= 0.02×max(0,danger)，乘危险度而非固定值。"""
    st = _ledger()
    rel = relationship.update_relationship(st, ["user_001"], 0.9, -0.9)
    assert rel["trust"] == pytest.approx(0.5 - 0.02 * 0.9)
    rel2 = relationship.update_relationship(st, ["user_001"], 0.1, -0.1)
    # 轻度事件只掉 0.002——"骂得多重"决定"伤得多深"
    assert rel2["trust"] == pytest.approx(0.5 - 0.02 * 0.9 - 0.02 * 0.1)


def test_trust_praise_gain_and_no_negative_danger():
    """安抚 +0.01；danger<0（被夸）不吃攻击系数（max(0,danger)）。"""
    st = _ledger()
    rel = relationship.update_relationship(st, ["user_001"], -0.65, 0.86)
    assert rel["trust"] == pytest.approx(0.5 + 0.01)
    # 中性（valence<=0, danger==0）：trust 不动
    rel2 = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel2["trust"] == pytest.approx(0.51)


def test_trust_clip_and_break_recovery():
    """clip [0,1]；broken 低于 0.2 置位、涨回 0.3 以上解除（058 施密特）。"""
    st = _ledger()
    for _ in range(60):
        rel = relationship.update_relationship(st, ["user_001"], 1.0, -1.0)
    assert rel["trust"] == 0.0 and rel["broken"] is True
    for _ in range(40):
        rel = relationship.update_relationship(st, ["user_001"], -0.5, 0.9)
    assert rel["trust"] > 0.2 and rel["broken"] is False
    assert rel["trust"] <= 1.0
    # 连夸上天也不破 1
    for _ in range(200):
        rel = relationship.update_relationship(st, ["user_001"], -0.5, 0.9)
    assert rel["trust"] == 1.0


# ---- T058-H：broken 施密特触发器（迟滞带）----

def test_broken_schmitt_hysteresis_band():
    """[058 任务2] 迟滞带 [0.2, 0.3)：band 内 trust 一升一降，
    broken 保持原值不翻转；带外才切换。

    用 (danger=0, valence=0) 的中性轮固定 trust，逐一验证四象限：
      带外下方（trust=0.19）→ True（触发）
      带内 True（trust=0.25）→ 保持 True（旧版会在此翻转成 False）
      带外上方（trust=0.30）→ False（解除）
      带内 False（trust=0.25）→ 保持 False（旧版会在此翻转成 True）
    """
    st = _ledger()
    u = st["relationships"]["user_001"]

    u["trust"], u["broken"] = 0.19, False
    rel = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel["trust"] == 0.19 and rel["broken"] is True

    u["trust"], u["broken"] = 0.25, True
    rel = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel["trust"] == 0.25 and rel["broken"] is True

    u["trust"], u["broken"] = 0.30, True
    rel = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel["trust"] == 0.30 and rel["broken"] is False

    u["trust"], u["broken"] = 0.25, False
    rel = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel["trust"] == 0.25 and rel["broken"] is False

    # 边界值：恰好等于 BREAK（0.2）不触发（严格小于才触发）
    u["trust"], u["broken"] = 0.2, False
    rel = relationship.update_relationship(st, ["user_001"], 0.0, 0.0)
    assert rel["broken"] is False


def test_broken_no_flap_over_dynamics():
    """[058 任务2] 动力学路径防抖：trust 在 band 内来回两轮，broken 不翻转。

    从 0.0（broken=True）出发：+0.01×20 到 0.20（band），
    再 -0.02×5 回 0.10（带外下方，仍 True），再 +0.01×20 回 0.30
    （带外上方，才解除）——全程 broken 只在最后翻一次。
    """
    st = _ledger()
    u = st["relationships"]["user_001"]
    u["trust"], u["broken"] = 0.0, True
    flips = 0
    prev = True
    seq = ([("p", 1)] * 20      # 0.00 → 0.20（band，保持 True）
           + [("a", 1)] * 5     # 0.20 → 0.10（带外下，仍 True）
           + [("p", 1)] * 20)   # 0.10 → 0.30（带外上，翻 False）
    for kind, n in seq:
        for _ in range(n):
            if kind == "p":
                rel = relationship.update_relationship(
                    st, ["user_001"], -0.5, 0.9)
            else:
                rel = relationship.update_relationship(
                    st, ["user_001"], 1.0, -1.0)
            if rel["broken"] != prev:
                flips += 1
                prev = rel["broken"]
    assert rel["broken"] is False and flips == 1


def test_intimacy_monotonic_accumulates():
    """intimacy 每轮 +0.01，只升不降（M1 规格：攻击不降亲密度）。"""
    st = _ledger()
    for _ in range(5):
        rel = relationship.update_relationship(st, ["user_001"], 0.95, -0.95)
    assert rel["intimacy"] == pytest.approx(0.55)


def test_interactions_count_all_entities():
    """entity_ids 每个实体 +1；user_001 恒 +1（即使文本没提"我"）。"""
    st = _ledger()
    relationship.update_relationship(st, ["ai_self", "boss_01"], 0.0, 0.0)
    rels = st["relationships"]
    assert rels["user_001"]["interactions"] == 1   # 用户本人在场
    assert rels["ai_self"]["interactions"] == 1
    assert rels["boss_01"]["interactions"] == 1    # 他人只计数不动力学
    assert rels["boss_01"]["trust"] == 0.5          # 他人 trust 不动


def test_others_get_no_dynamics():
    """M1 边界：骂"老板"不改变 boss_01 的 trust，只改 user_001。"""
    st = _ledger()
    for _ in range(10):
        relationship.update_relationship(st, ["boss_01"], 0.95, -0.9)
    rels = st["relationships"]
    assert rels["boss_01"]["trust"] == 0.5
    assert rels["boss_01"]["interactions"] == 10
    assert rels["user_001"]["trust"] < 0.4          # 眼前这个人挨了骂


# ---- T056-C：熟人效应与文案 ----

def test_acquaintance_effects_thresholds():
    """trust>0.7 → D_slow×1.5；trust<0.3 → H×1.5；中间区恒 1。"""
    assert relationship.acquaintance_effects(0.9) == (1.5, 1.0)
    assert relationship.acquaintance_effects(0.71) == (1.5, 1.0)
    assert relationship.acquaintance_effects(0.7) == (1.0, 1.0)   # 边界不含
    assert relationship.acquaintance_effects(0.1) == (1.0, 1.5)
    assert relationship.acquaintance_effects(0.29) == (1.0, 1.5)
    assert relationship.acquaintance_effects(0.3) == (1.0, 1.0)   # 边界不含
    assert relationship.acquaintance_effects(0.5) == (1.0, 1.0)
    # 损坏输入 → 默认 0.5 → 不干预
    assert relationship.acquaintance_effects(None) == (1.0, 1.0)


def test_relationship_text_four_tiers():
    """独白四档：破裂 / 淡漠 / 熟人 / 普通；None → 无文案。"""
    assert "碎" in relationship.relationship_text({"trust": 0.1})
    assert "无所谓" in relationship.relationship_text({"trust": 0.25})
    assert "信任" in relationship.relationship_text({"trust": 0.9})
    assert "客气" in relationship.relationship_text({"trust": 0.5})
    assert relationship.relationship_text(None) is None
    assert relationship.relationship_text({"trust": "坏值"}) is None


# ---- T058-I：intimacy 独白 ----

def test_intimacy_text_two_tiers():
    """[058 任务2] 亲密度两档：>0.7 认识很久 / <0.2 还很防备；
    中间档与 None/坏值 → None（独白不出现该行）。"""
    assert "认识很久" in relationship.intimacy_text({"intimacy": 0.71})
    assert "熟悉" in relationship.intimacy_text({"intimacy": 0.9})
    assert "防备" in relationship.intimacy_text({"intimacy": 0.19})
    assert "防备" in relationship.intimacy_text({"intimacy": 0.0})
    # 边界不含：恰好 0.7 / 0.2 属中间档
    assert relationship.intimacy_text({"intimacy": 0.7}) is None
    assert relationship.intimacy_text({"intimacy": 0.2}) is None
    assert relationship.intimacy_text({"intimacy": 0.5}) is None
    # 缺字段按默认 0.5 → 中间档
    assert relationship.intimacy_text({}) is None
    assert relationship.intimacy_text(None) is None
    assert relationship.intimacy_text({"intimacy": "坏值"}) is None


def test_intimacy_text_orthogonal_to_trust():
    """[058 任务2] 两线正交：trust 高 ≠ intimacy 必有文案，
    同一条目可同时命中信任档与亲密度档。"""
    rel = {"trust": 0.9, "intimacy": 0.8}
    assert relationship.relationship_text(rel) is not None
    assert relationship.intimacy_text(rel) is not None
    rel2 = {"trust": 0.9, "intimacy": 0.5}   # 信任但还不太熟
    assert relationship.relationship_text(rel2) is not None
    assert relationship.intimacy_text(rel2) is None


# ---- T056-D：Prompt 注入 ----

def test_state_to_prompt_relationship_none_is_legacy_identical():
    """relationship=None → 与 056 之前逐字一致（空行结构不丢）。"""
    p = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first")
    assert "- 关系：" not in p
    assert "- 亲密度：" not in p
    assert "- 隐约的回忆：脑子里没什么特别的念头\n\n【输出规则" in p


def test_state_to_prompt_with_relationship_inserts_line():
    """带 relationship → 出现关系行，且不破坏其余结构。"""
    p = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                        relationship={"trust": 0.9})
    assert "- 关系：你们之间已经有信任" in p
    assert p.count("【输出规则（必须严格遵守）】") == 1
    # 关系行插在回忆与规则之间，空行结构保持一致
    assert "- 隐约的回忆：脑子里没什么特别的念头\n- 关系：" in p


def test_state_to_prompt_intimacy_inserts_line():
    """[058 任务2] intimacy>0.7 → 亲密度行出现；中间档 → 不出现。"""
    p_hi = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                           relationship={"trust": 0.5, "intimacy": 0.9})
    assert "- 亲密度：你们已经认识很久了，你对他很熟悉。" in p_hi
    assert p_hi.count("【输出规则（必须严格遵守）】") == 1
    # 中间档（trust 普通 + intimacy 0.5）：两行都无，结构与 None 一致
    p_mid = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                            relationship={"trust": 0.5, "intimacy": 0.5})
    assert "- 亲密度：" not in p_mid
    assert "- 关系：你们还算客气" in p_mid
    # LO 档文案
    p_lo = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                           relationship={"trust": 0.5, "intimacy": 0.1})
    assert "- 亲密度：你对这个人还很防备。" in p_lo


def test_state_to_prompt_both_lines_order_and_structure():
    """[058 任务2] 两行齐备时顺序为关系→亲密度→规则，空行结构不破。"""
    p = state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                        relationship={"trust": 0.9, "intimacy": 0.9})
    assert "- 关系：" in p and "- 亲密度：" in p
    assert p.index("- 关系：") < p.index("- 亲密度：")
    assert ("- 隐约的回忆：脑子里没什么特别的念头\n- 关系：" in p)
    assert ("\n- 亲密度：你们已经认识很久了，你对他很熟悉。\n\n"
            "【输出规则" in p)


# ---- T056-E：持久化 ----

def test_session_roundtrip_relationships():
    """dump→load 往返：user_001 三字段一致，他人实体随账本持久化。"""
    st = reset_state(seed=42)
    relationship.update_relationship(st, ["user_001", "boss_01"], 0.9, -0.8)
    payload = session_store.dump_state(st)
    st2 = session_store.load_state(payload, warn=lambda m: None)
    u, u2 = st["relationships"]["user_001"], st2["relationships"]["user_001"]
    assert u2["trust"] == u["trust"]
    assert u2["intimacy"] == u["intimacy"]
    assert u2["interactions"] == u["interactions"]
    assert "boss_01" in st2["relationships"]


def test_session_old_file_without_relationships():
    """旧存档无 relationships 字段 → 回落 reset_state 默认账本（不崩）。"""
    st = reset_state(seed=42)
    payload = session_store.dump_state(st)
    payload.pop("relationships")
    st2 = session_store.load_state(payload, warn=lambda m: None)
    assert st2["relationships"]["user_001"]["trust"] == 0.5
    assert st2["relationships"]["user_001"]["interactions"] == 0


def test_session_corrupt_entry_skipped():
    """单条损坏（trust 非数字）→ 跳过该条，其余实体照常恢复。"""
    st = reset_state(seed=42)
    relationship.update_relationship(st, ["user_001"], 0.5, -0.5)
    payload = session_store.dump_state(st)
    payload["relationships"]["user_001"]["trust"] = "坏值"
    payload["relationships"]["ai_self"] = "整条坏掉"
    st2 = session_store.load_state(payload, warn=lambda m: None)
    # user_001 因 trust 损坏被跳过 → 保持默认 0.5；ai_self 整条坏 → 默认
    assert st2["relationships"]["user_001"]["trust"] == 0.5
    assert st2["relationships"]["ai_self"]["interactions"] == 0


def test_session_broken_recomputed_from_trust():
    """broken 不信任存档布尔值直接生效，载入时按 trust 重算（防不自洽）；
    [058 任务2] 起重算为施密特口径——带外（<0.2 / >=0.3）强制重算，
    带内（[0.2,0.3)）保持存档值（见下方迟滞带专项用例）。"""
    st = reset_state(seed=42)
    relationship.update_relationship(st, ["user_001"], 0.9, -0.9)
    payload = session_store.dump_state(st)
    payload["relationships"]["user_001"]["broken"] = True   # 与 trust 矛盾
    payload["relationships"]["user_001"]["trust"] = 0.5      # 但 trust 明明安全
    st2 = session_store.load_state(payload, warn=lambda m: None)
    assert st2["relationships"]["user_001"]["broken"] is False


def test_session_broken_load_band_keeps_saved():
    """[058 任务2] load 迟滞带：trust 落在 [0.2, 0.3) 时 broken 保持
    存档值（True 存 True 回、False 存 False 回），不被重算翻转；
    带外仍强制重算（trust<0.2→True、trust>=0.3→False）。"""
    st = reset_state(seed=42)
    relationship.update_relationship(st, ["user_001"], 0.9, -0.9)
    base = session_store.dump_state(st)

    # 带内：存档 True → 回 True
    p1 = dict(base)
    p1["relationships"] = {"user_001": dict(base["relationships"]["user_001"],
                                            trust=0.25, broken=True)}
    st1 = session_store.load_state(p1, warn=lambda m: None)
    assert st1["relationships"]["user_001"]["broken"] is True

    # 带内：存档 False → 回 False
    p2 = dict(base)
    p2["relationships"] = {"user_001": dict(base["relationships"]["user_001"],
                                            trust=0.25, broken=False)}
    st2 = session_store.load_state(p2, warn=lambda m: None)
    assert st2["relationships"]["user_001"]["broken"] is False

    # 带内：非布尔存档值 → 按 False 处理（防 "false" 字符串被吃成 True）
    p3 = dict(base)
    p3["relationships"] = {"user_001": dict(base["relationships"]["user_001"],
                                            trust=0.25, broken="false")}
    st3 = session_store.load_state(p3, warn=lambda m: None)
    assert st3["relationships"]["user_001"]["broken"] is False

    # 带外下：存档 False 也强制 True
    p4 = dict(base)
    p4["relationships"] = {"user_001": dict(base["relationships"]["user_001"],
                                            trust=0.1, broken=False)}
    st4 = session_store.load_state(p4, warn=lambda m: None)
    assert st4["relationships"]["user_001"]["broken"] is True

    # 带外上：存档 True 也强制 False
    p5 = dict(base)
    p5["relationships"] = {"user_001": dict(base["relationships"]["user_001"],
                                            trust=0.5, broken=True)}
    st5 = session_store.load_state(p5, warn=lambda m: None)
    assert st5["relationships"]["user_001"]["broken"] is False


# ---- T056-F：端到端（pipeline 一轮）----

class _FixedLLM:
    def generate(self, prompt, temperature=None, messages=None):
        return "（测试固定回复）"


def _run_one_insult_turn(monkeypatch, tmp_path, trust_before=None):
    """跑一轮被骂，返回 (st, log)。trust_before 非 None 时预置账本。"""
    def _fake_signals(_text, llm=None):
        return np.asarray(INSULT_S, dtype=float), "test"

    def _fake_fixture(_scn):
        return {"input": INSULT, "event_severity": 0.86, "danger_score": 0.0,
                "importance": 0.5, "emotion_weight": 0.5, "unfinished": 0.0,
                "x": np.zeros(8), "x_fast": np.zeros(4)}

    monkeypatch.setattr(pipeline, "get_llm", lambda: _FixedLLM())
    monkeypatch.setattr(pipeline, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(decompressor, "extract_semantic_signals", _fake_signals)
    monkeypatch.setattr(pipeline, "get_fixture", _fake_fixture)
    monkeypatch.setattr(pipeline, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")

    st = reset_state(seed=42)
    if trust_before is not None:
        st["relationships"]["user_001"]["trust"] = float(trust_before)
    log = pipeline.run_turn(st, "__t056__", feature_mode="encoder")
    return st, log


def test_pipeline_updates_ledger_and_log_fields(monkeypatch, tmp_path):
    """一轮被骂：trust 下降；log 带全部 056 观测字段。"""
    st, log = _run_one_insult_turn(monkeypatch, tmp_path)
    u = st["relationships"]["user_001"]
    assert u["trust"] < 0.5
    assert u["interactions"] == 1
    assert "ai_self" in st["relationships"]
    # log 观测字段齐全
    assert log["relationships"]["user_001"]["trust"] == u["trust"]
    assert log["rel_trust_used"] == 0.5            # 熟人效应读的是轮前旧值
    assert log["rel_d_slow_gain"] == 1.0
    assert log["rel_h_gain"] == 1.0
    assert log["HabituationSignal_raw"] == log["HabituationSignal"]


def test_pipeline_h_gain_active_but_writeback_unamplified(monkeypatch, tmp_path):
    """淡漠档（trust<0.3）：当轮 H_eff = clip(H_raw×1.5)，
    但写回 HabituationSignal_prev 用未放大值（防逐轮自叠加）。"""
    st, log = _run_one_insult_turn(monkeypatch, tmp_path, trust_before=0.1)
    assert log["rel_h_gain"] == 1.5
    assert log["rel_trust_used"] == 0.1
    # 当轮生效值 = 放大后（上限 1.0）
    assert log["HabituationSignal"] == pytest.approx(
        min(1.0, log["HabituationSignal_raw"] * 1.5))
    # 写回的是原值——下一轮不会在放大的基础上继续放大
    assert st["HabituationSignal_prev"] == pytest.approx(
        log["HabituationSignal_raw"])


def test_pipeline_d_slow_gain_active(monkeypatch, tmp_path):
    """熟人档（trust>0.7）：D_slow 被 ×1.5（相对未放大值的差额可验证）。"""
    st_on, log_on = _run_one_insult_turn(monkeypatch, tmp_path, trust_before=0.9)
    assert log_on["rel_d_slow_gain"] == 1.5
    # 与 trust=0.5（无增益）的同输入轮对比：D_slow 应显著更高
    st_off, log_off = _run_one_insult_turn(monkeypatch, tmp_path, trust_before=0.5)
    assert log_on["D_slow"] > log_off["D_slow"] * 1.2


def test_reset_state_has_relationships_key():
    """T33 断言覆盖 relationships：reset_state 必含该键。"""
    st = reset_state(seed=42)
    assert "relationships" in st
    assert set(st["relationships"]) >= {"user_001", "ai_self"}
