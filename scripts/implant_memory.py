# ============================================================
# implant_memory.py · 047 · 上帝之手（记忆植入工具）
#
# 用途：所有者手动运行，把一段编造的故事以「合法 MemoryPackage +
#   EventNode」的形式直接写入指定会话存档（默认 sessions/owner_main.json），
#   使 Vesper 后续对话能检索到这段记忆并受其情绪影响。
#
# 设计要点（不修改任何既有模块，只复用其数据契约）：
#   1. 向量：调用现有 BGE embedder（config.PACKAGE_EMBEDDER，1024 维），
#      同一向量同时作为 mean_vec 与 peak_vec（与 archiver 单轮包口径一致）。
#   2. 情绪：valence（[-1,1]）、intensity（[0,1]）、danger_score（[0,1]）
#      全部由外部传入，不做任何再推导。
#   3. raw_pointer 写成 "implant-<seq>"（非数字）：decompressor 解析失败
#      → 不走轮级回捞 → 读取侧回落到 package_level，使用我们伪造的
#      emotion_value / emotion_polarity，不会被 evaluator 重提覆盖。
#   4. 落盘安全：改前备份为 <会话>.json.bak-<时间戳>，改用
#      session_store.save（先写 tmp 再 os.replace，原子替换），
#      写后立刻 session_store.load 回读校验，失败则保留原文件。
#
# 重要：后端进程把会话缓存在内存（server_api._SESSIONS）。植入是对磁盘
#   操作，后端正在运行时不会自动感知——植入后必须重启后端，或换一个
#   尚未加载过的 session_id，新的记忆才会生效。
#
# 用法一（推荐）：直接改下面的 MEMORY_TO_IMPLANT，然后运行
#   d:\aiqg\venv\Scripts\python.exe scripts\implant_memory.py
# 用法二（命令行）：
#   ... scripts\implant_memory.py --text "上周你因为写错代码被我狠狠骂了一顿" \
#       --valence -0.8 --intensity 0.85 --danger 0.55
# 用法三（纯交互）：MEMORY_TO_IMPLANT["story"] 留空直接运行，按提示输入。
# 附加：--dry-run 只计算不写盘；--show 打印目标会话当前记忆包清单。
# ============================================================

import argparse
import os
import shutil
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config                                   # noqa: E402
import session_store                            # noqa: E402
from embedder import get_embedder               # noqa: E402
from schemas import EventNode, MemoryPackage    # noqa: E402
from state import reset_state                   # noqa: E402


# ------------------------------------------------------------
# 所有者可直接修改这里，然后运行脚本即可
# ------------------------------------------------------------
MEMORY_TO_IMPLANT = {
    "session_id": "owner_main",
    "story": "上周你因为写错代码被我狠狠骂了一顿，你当时特别委屈。",
    "valence": -0.8,          # [-1, 1] 负 = 伤痛/被责怪，正 = 愉悦/被夸奖
    "intensity": 0.85,        # [0, 1] 情绪强度（读取侧 ev）
    "danger_score": 0.55,     # [0, 1] 危险度（事件级 HR 与包级 danger 聚合）
    "importance": 0.80,       # [0, 1] 事件重要度（竞争分用）
    "keywords": [],           # 留空则自动从故事文本截取
    "narrative_cause": "",    # 六个叙事字段留空则用 story 兜底
    "narrative_trigger": "",
    "narrative_consequence": "",
    "narrative_evaluation": "",
    "narrative_emotion": "",
    "narrative_thought": "",
}


def _clip01(x):
    return float(max(0.0, min(1.0, float(x))))


def _polarity_of(valence):
    if valence > 0:
        return "positive"
    if valence < 0:
        return "negative"
    return "neutral"


def _auto_keywords(story, limit=3, width=6):
    """无关键词时从故事里截取若干片段（与 archiver 桩口径同样式，够用即可）。"""
    text = "".join(ch for ch in str(story) if not ch.isspace())
    if not text:
        return []
    return [text[i * width:(i + 1) * width] for i in range(limit)
            if text[i * width:(i + 1) * width]]


def _embed(text):
    """用系统现有 BGE 编码并归一化；同一向量同时作 mean_vec / peak_vec。"""
    name = getattr(config, "PACKAGE_EMBEDDER", "bge")
    vec = get_embedder(name).encode(text)
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return arr


def _load_or_init(session_id, dir_path=None):
    """读目标会话；不存在则按 reset_state 初始化（不写盘）。"""
    st, _meta = session_store.load(session_id, dir_path)
    if st is None:
        st = reset_state(seed=42)
        return st, False
    return st, True


def build_memory_package(story, valence, intensity, danger_score, seq,
                         keywords=None, narratives=None):
    """构造合法的 MemoryPackage（字段口径对齐 archiver 单轮包）。"""
    vec = _embed(story)
    narratives = narratives or {}
    val = float(valence)
    inten = _clip01(intensity)
    danger = _clip01(danger_score)
    return MemoryPackage(
        seq=int(seq),
        emotion_value=inten,
        emotion_polarity=_polarity_of(val),
        keywords=list(keywords) if keywords else _auto_keywords(story),
        # 非数字指针：decompressor 解析失败 → package_level 回落，
        # 保证读取侧用的是我们伪造的情绪值，而不是 evaluator 重提的结果。
        raw_pointer="implant-%d" % int(seq),
        mean_vec=vec,
        peak_vec=vec.copy(),
        mean_valence=val,
        n_rounds=1,
        mean_danger=danger,
        max_danger=danger,
        narrative_cause=str(narratives.get("narrative_cause") or story),
        narrative_trigger=str(narratives.get("narrative_trigger") or "所有者植入"),
        narrative_consequence=str(narratives.get("narrative_consequence") or story),
        narrative_evaluation=str(narratives.get("narrative_evaluation") or ""),
        narrative_emotion=str(narratives.get("narrative_emotion") or ""),
        narrative_thought=str(narratives.get("narrative_thought") or ""),
    )


def build_event_node(story, pkg, importance=0.8, danger_score=0.55):
    """构造对应的 EventNode（事件级：HR / 习惯化 / 海马体竞争的数据源）。"""
    vec = pkg.mean_vec
    return EventNode(
        event_id="implant-%d" % int(pkg.seq),
        timestamp=time.time(),
        raw_text=story,
        event_embedding=np.asarray(vec, dtype=np.float32).copy(),
        nodes={},
        relationships=[],
        importance=_clip01(importance),
        emotion_weight=abs(float(pkg.mean_valence or 0.0)),
        unfinished=0.0,
        persons=set(),
        tasks=set(),
        danger_score=_clip01(danger_score),
        event_severity=_clip01(danger_score),
    )


def implant(session_id="owner_main", story="", valence=-0.5, intensity=0.7,
            danger_score=0.4, importance=0.7, keywords=None,
            narratives=None, dir_path=None, dry_run=False, verbose=True):
    """执行植入。返回 dict（含 backup / path / seq / 校验结果）。

    失败时抛异常，且不覆盖原文件（备份先做、原子替换、回读校验）。
    """
    story = str(story or "").strip()
    if not story:
        raise ValueError("故事文本为空，拒绝植入")

    st, existed = _load_or_init(session_id, dir_path)
    packages = st["archiver"].get_packages()
    seq = max([int(getattr(p, "seq", 0)) for p in packages] or [0]) + 1

    pkg = build_memory_package(story, valence, intensity, danger_score, seq,
                               keywords=keywords, narratives=narratives)
    event = build_event_node(story, pkg, importance=importance,
                             danger_score=danger_score)

    result = {
        "session_id": session_id,
        "seq": seq,
        "event_id": event.event_id,
        "polarity": pkg.emotion_polarity,
        "emotion_value": pkg.emotion_value,
        "mean_valence": pkg.mean_valence,
        "danger_score": event.danger_score,
        "existed": existed,
        "dry_run": bool(dry_run),
        "backup": None,
        "path": None,
        "packages_after": None,
        "events_after": None,
    }

    if dry_run:
        if verbose:
            print("[implant] dry-run：未写盘。目标 seq=%d，极性=%s，强度=%.2f"
                  % (seq, pkg.emotion_polarity, pkg.emotion_value))
        return result

    path = session_store._path(session_id, dir_path)
    if existed and os.path.exists(path):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = "%s.bak-%s" % (path, stamp)
        shutil.copy2(path, backup)
        result["backup"] = backup
        if verbose:
            print("[implant] 已备份原存档：%s" % backup)

    # 直接追加到运行时状态，再走 session_store.save（原子写）
    st["archiver"].append_package(pkg)
    st["memory_graph"].add(event)
    saved = session_store.save(session_id, st, dir_path)
    if not saved:
        raise RuntimeError("写入会话存档失败（原文件未被替换）：%s" % path)
    result["path"] = saved

    # 回读校验：JSON 可解析 + 包/事件都能被 load_state 还原
    check, meta = session_store.load(session_id, dir_path)
    if check is None:
        raise RuntimeError("回读校验失败：存档无法加载，请用备份还原：%s"
                           % result["backup"])
    got_pkgs = check["archiver"].get_packages()
    if not any(int(getattr(p, "seq", 0)) == seq for p in got_pkgs):
        raise RuntimeError("回读校验失败：未找到刚植入的 seq=%d，请用备份还原：%s"
                           % (seq, result["backup"]))
    result["packages_after"] = len(got_pkgs)
    result["events_after"] = len(check["memory_graph"].get_all())

    if verbose:
        print("[implant] 植入完成")
        print("  会话      : %s" % session_id)
        print("  存档      : %s" % saved)
        print("  记忆包 seq: %d（共 %d 个包）" % (seq, result["packages_after"]))
        print("  事件节点  : %s（共 %d 个事件）"
              % (event.event_id, result["events_after"]))
        print("  极性与强度: %s / %.2f" % (pkg.emotion_polarity, pkg.emotion_value))
        print("  效价/危险 : %.2f / %.2f" % (pkg.mean_valence, event.danger_score))
        print("  故事      : %s" % story)
        print("  提示      : 后端若正在运行，请重启后再对话，新记忆才会被加载。")
    return result


def show_session(session_id="owner_main", dir_path=None):
    """打印目标会话现有的记忆包与事件（只读）。"""
    st, meta = session_store.load(session_id, dir_path)
    if st is None:
        print("[implant] 会话 %s 不存在或无法加载" % session_id)
        return
    pkgs = st["archiver"].get_packages()
    events = st["memory_graph"].get_all()
    print("[implant] 会话 %s：round=%s，记忆包 %d 个，事件 %d 个"
          % (session_id, st.get("round_counter", 0), len(pkgs), len(events)))
    for p in pkgs:
        print("  包 seq=%-3d 极性=%-8s 强度=%.2f 效价=%s 指针=%s 关键词=%s"
              % (p.seq, p.emotion_polarity, p.emotion_value,
                 ("%.2f" % p.mean_valence) if p.mean_valence is not None else "None",
                 p.raw_pointer, "、".join(p.keywords)))
    for e in events:
        print("  事件 %s 危险=%.2f 重要=%.2f 原文=%s"
              % (e.event_id, e.danger_score, e.importance, e.raw_text[:30]))


def simulate_retrieval(session_id="owner_main", query="今天天气不错", dir_path=None):
    """植入后自检：按 memory_gateway 的路径确认能否召回植入包（只读）。

    与线上链路同参：包级 _recall_match 需要 now_valence；这里传 0.0（中性）。
    """
    from pipeline import _recall_match

    st, _ = session_store.load(session_id, dir_path)
    if st is None:
        print("[implant] 会话 %s 无法加载" % session_id)
        return None
    packages = st["archiver"].get_packages()
    recalled = None
    if packages:
        recalled = _recall_match(st, packages, query,
                                 int(st.get("round_counter", 0)), now_valence=0.0)
    print("[implant] 自检查询：%s" % query)
    if recalled is None:
        print("  结果：未召回（TAU_FLOOR=%.2f 未过，或极性过滤耗尽）"
              % config.TAU_FLOOR)
    else:
        print("  结果：召回 seq=%s，极性=%s，强度=%.2f，来源=%s，相似度=%.3f"
              % (recalled.get("matched_pkg_seq"),
                 recalled.get("emotion_polarity"),
                 float(recalled.get("emotion_value", 0.0)),
                 recalled.get("recall_source"),
                 float(recalled.get("matched_sim", 0.0))))
    return recalled


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Vesper 记忆植入工具（上帝之手，047）")
    ap.add_argument("--session", default=MEMORY_TO_IMPLANT["session_id"],
                    help="目标会话 id（默认 owner_main）")
    ap.add_argument("--text", default=MEMORY_TO_IMPLANT["story"],
                    help="要植入的故事文本")
    ap.add_argument("--valence", type=float,
                    default=MEMORY_TO_IMPLANT["valence"], help="效价 [-1,1]")
    ap.add_argument("--intensity", type=float,
                    default=MEMORY_TO_IMPLANT["intensity"], help="强度 [0,1]")
    ap.add_argument("--danger", type=float,
                    default=MEMORY_TO_IMPLANT["danger_score"], help="危险度 [0,1]")
    ap.add_argument("--importance", type=float,
                    default=MEMORY_TO_IMPLANT["importance"], help="事件重要度 [0,1]")
    ap.add_argument("--dir", default=None, help="会话目录（默认 sessions/）")
    ap.add_argument("--dry-run", action="store_true", help="只计算不写盘")
    ap.add_argument("--show", action="store_true", help="只打印现有记忆，不植入")
    ap.add_argument("--simulate", default="", metavar="QUERY",
                    help="按给定话语做一次召回自检（只读）")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.show:
        show_session(args.session, args.dir)
        return 0
    if args.simulate:
        simulate_retrieval(args.session, args.simulate, args.dir)
        return 0

    text = args.text
    if not text:
        text = input("请输入要植入的故事文本：").strip()
    implant(args.session, text,
            valence=args.valence, intensity=args.intensity,
            danger_score=args.danger, importance=args.importance,
            dir_path=args.dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
