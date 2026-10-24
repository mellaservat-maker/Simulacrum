# ============================================================
# event_parser.py · V4.5 · 模块1 事件拆解
# 规格：5.5.1 / 5.7
# 输出：EventNode + event_embedding
# V4.5（T3-B）：嵌入改用embedder.get_embedder("bge")，1024 维语义嵌入
# M1：极简解析，只提取 person/task
# ============================================================

import hashlib
import time

import numpy as np

from schemas import EventNode
from embedder import get_embedder


def hash_to_8d(raw_text):
    """【已弃用】8 维 MD5 hash 嵌入，无语义。

    仅保留供 scripts/probe_embedder.py 向后兼容验证使用（它比对
    HashEmbedder 与本函数输出是否逐元素一致），主链路不再调用。
    """
    h = hashlib.md5(raw_text.encode()).digest()
    vec = np.array([h[i] / 255.0 for i in range(8)])
    return vec / (np.linalg.norm(vec) + 1e-8)


# 模块级单例：避免每次 parse 都重建 embedder（BGE 为懒加载，构造本身轻量）
_embedder = get_embedder("bge")


_PERSON_HINTS = ("你", "我", "他", "她", "它", "大家", "老板", "用户", "AI", "老师")
_TASK_HINTS = ("方案", "代码", "问题", "任务", "答案", "回复", "报告", "文档", "项目")

# [056 任务1] 人称/身份词 → 稳定实体 ID 映射表。
#   M1 范围（所有者拍板"先支持识别当前用户和他人"）：
#     user_001   当前用户（说话人"我"）——关系账本的主对象，
#                trust/intimacy 动力学全部挂在它身上；
#     ai_self    AI 自己（对话中的"你"）——自己与自己没有信任
#                可言，只进 entity_ids 计数，不进动力学；
#     boss_01 / teacher_01 / other_01 / others_group
#                话题中提到的他人——只计数。M1 不给他们建信任：
#                他人不在对话现场，"骂老板"挨骂的是眼前这个人，
#                动力学必须挂在 user_001 上（否则 trust 会被话题
#                里的第三方稀释，这是 M1 明确的边界）。
#   为什么"他/她/它"合并为 other_01：第三人称指代对象不固定
#   （"他"这次是老板、下次是同事），M1 无共指消解能力，合并成
#   泛指"他人"是保守选择——宁可少分，不可错分。
_PERSON_ENTITY_MAP = {
    "我": "user_001",
    "用户": "user_001",
    "你": "ai_self",
    "AI": "ai_self",
    "老板": "boss_01",
    "老师": "teacher_01",
    "他": "other_01",
    "她": "other_01",
    "它": "other_01",
    "大家": "others_group",
}


def _extract_persons(raw_text):
    return [p for p in _PERSON_HINTS if p in raw_text]


def _extract_entities(raw_text):
    """[056] 词面命中 → 稳定实体 ID（去重，按 _PERSON_HINTS 顺序稳定输出）。

    与 _extract_persons 的关系：同一个 _PERSON_HINTS 词表，两种口径。
      _extract_persons 产出的词集合喂 feature_encoder 的 x1（053 口径，
        不动）；本函数产出的 ID 集合是会话级关系账本的钥匙。
    """
    out = []
    for w in _PERSON_HINTS:
        eid = _PERSON_ENTITY_MAP.get(w)
        if eid and w in raw_text and eid not in out:
            out.append(eid)
    return out


def _extract_tasks(raw_text):
    return [t for t in _TASK_HINTS if t in raw_text]


def parse(raw_text, event_id=None, timestamp=None,
          event_severity=0.0, importance=0.5,
          emotion_weight=0.5, unfinished=0.0,
          danger_score=0.0):
    if event_id is None:
        event_id = "E" + str(int(time.time() * 1000) % 1000000)
    if timestamp is None:
        timestamp = time.time()

    # [053] persons / tasks 双写：此前只填 nodes["person"] / nodes["task"]，
    #   EventNode.persons / .tasks 恒为空集 → xinguang.competition_score 里
    #   person_match_term / task_match_term 永远算 0（L18 的真因，不是提取
    #   能力缺失）。feature_encoder 读 nodes（所以 x1 有值），海马体读
    #   .persons/.tasks（所以恒 0），同一份信息两个口径，必须同时写。
    #   注意：_PERSON_HINTS 含"你/我/他"等高频词，绝大多数语句都会命中，
    #   因此 person_match 更接近"恒 1 的常数项"而非区分度来源；
    #   真正的人物区分要等 T20（实体抽取）。
    persons = _extract_persons(raw_text)
    tasks = _extract_tasks(raw_text)
    entity_ids = _extract_entities(raw_text)

    return EventNode(
        event_id=event_id,
        timestamp=timestamp,
        raw_text=raw_text,
        event_embedding=_embedder.encode(raw_text),
        nodes={
            "person": persons,
            "task": tasks,
            "topic": [],
            "state": [],
            "relation": [],
        },
        relationships=[],
        persons=set(persons),
        tasks=set(tasks),
        entity_ids=entity_ids,
        importance=importance,
        emotion_weight=emotion_weight,
        unfinished=unfinished,
        danger_score=danger_score,
        event_severity=event_severity,
    )