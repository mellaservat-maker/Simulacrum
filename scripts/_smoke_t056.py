# -*- coding: utf-8 -*-
"""056 任务 1/2 冒烟：实体提取 + 关系动力学 + 熟人效应 + Prompt 注入。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import relationship
import state as state_mod
import event_parser
import pipeline
import state_translator
import session_store
import config

print("== 1) 实体提取 ==")
for text in ["你这个AI真蠢，什么都做不好。",
             "老板昨天夸了我，他说这个方案很好",
             "大家怎么看？",
             "今天天气不错"]:
    ev = event_parser.parse(text)
    print("  %-20s -> %s" % (text[:18], ev.entity_ids))

print("== 2) 关系账本初始化 ==")
st = state_mod.reset_state()
print("  keys:", list(st["relationships"].keys()))

print("== 3) 动力学：连骂 6 轮（danger≈0.96） ==")
for i in range(6):
    rel = relationship.update_relationship(st, ["user_001", "ai_self"], 0.96, -0.9)
    print("  第%d轮 trust=%.4f intimacy=%.4f inter=%d broken=%s"
          % (i + 1, rel["trust"], rel["intimacy"], rel["interactions"], rel["broken"]))

print("== 4) 动力学：连夸 6 轮（danger=-0.65, valence>0） ==")
st2 = state_mod.reset_state()
for i in range(6):
    rel = relationship.update_relationship(st2, ["user_001"], -0.65, 0.86)
    print("  第%d轮 trust=%.4f intimacy=%.4f inter=%d broken=%s"
          % (i + 1, rel["trust"], rel["intimacy"], rel["interactions"], rel["broken"]))

print("== 5) 熟人效应换算 ==")
for t in (0.1, 0.25, 0.5, 0.75, 0.95):
    print("  trust=%.2f -> gains=%s" % (t, relationship.acquaintance_effects(t)))

print("== 6) 独白文案四档 ==")
for t in (0.1, 0.25, 0.5, 0.9):
    print("  trust=%.2f -> %s" % (t, relationship.relationship_text({"trust": t})))

print("== 7) state_to_prompt 关系注入 ==")
import numpy as np
p_old = state_translator.state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first")
p_new = state_translator.state_to_prompt(np.zeros(4), 0.1, "", 0.0, 0.0, "task_first",
                                         relationship={"trust": 0.85})
assert "- 关系：" not in p_old, "relationship=None 时不应出现关系行"
assert "- 关系：" in p_new, "传入 relationship 时应出现关系行"
assert p_old.count("\n\n") == 1 and p_new.count("\n\n") == 1, "空行结构应保持一致"
print("  relationship=None 与旧版一致: OK")
print("  relationship 注入片段:", [l for l in p_new.splitlines() if "关系" in l])

print("== 8) session_store 持久化往返 ==")
st3 = state_mod.reset_state()
relationship.update_relationship(st3, ["user_001", "boss_01"], 0.9, -0.8)
payload = session_store.dump_state(st3)
st4 = session_store.load_state(payload, warn=lambda m: print("  warn:", m))
u3 = st3["relationships"]["user_001"]
u4 = st4["relationships"]["user_001"]
assert u4["trust"] == u3["trust"] and u4["intimacy"] == u3["intimacy"], (u3, u4)
assert u4["interactions"] == u3["interactions"], (u3, u4)
assert "boss_01" in st4["relationships"], "他人实体应随账本持久化"
print("  user_001 往返一致: OK  trust=%.4f intimacy=%.4f inter=%d"
      % (u4["trust"], u4["intimacy"], u4["interactions"]))
print("  boss_01 往返存在: OK")

print("== 9) 旧存档兼容（无 relationships 字段） ==")
payload_old = dict(payload)
payload_old.pop("relationships")
st5 = session_store.load_state(payload_old, warn=lambda m: print("  warn:", m))
assert st5["relationships"]["user_001"]["trust"] == 0.5
print("  旧存档载入后回落默认账本: OK")

print("ALL SMOKE OK")
