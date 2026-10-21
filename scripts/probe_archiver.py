# ============================================================
# probe_archiver.py · M2-B 探针
# 验证 Archiver 归档流程
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from short_term_buffer import ShortTermBuffer
from archiver import Archiver


def make_round(rid, user, ai, ts):
    return {
        "round_id": rid,
        "user_input": user,
        "ai_output": ai,
        "timestamp": ts,
    }


def check(label, condition, detail=""):
    if condition:
        print(f"  {label} ✓")
        return True
    print(f"  {label} ✗  {detail}")
    return False


def main():
    buf = ShortTermBuffer(default_keep=2)
    archiver = Archiver(buf, threshold=6)

    # 1. 追加 7 轮
    dialogs = [
        (1, "你好啊", "你好，有什么可以帮你的？"),
        (2, "今天心情怎么样", "还不错，就正常吧"),
        (3, "你这个做的不行", "……抱歉，我再看看"),
        (4, "算了你也做不好", "我会努力改进的"),
        (5, "上次那个问题", "哪个问题呀？"),
        (6, "就是代码那个", "哦，我记着呢"),
        (7, "你记得挺清楚嘛", "哈哈谢谢"),
    ]
    for rid, user, ai in dialogs:
        buf.append(make_round(rid, user, ai, 1000.0 + rid))

    print(f"=== 阶段一：7 轮追加后 ===")
    print(f"  buffer 长度 = {len(buf)}")

    # 2. should_archive
    print()
    print("=== should_archive 检查 ===")
    sa = archiver.should_archive()
    ok = check("should_archive == True", sa == True, f"实际 = {sa}")
    if not ok:
        print("TODO_REVIEW: should_archive 未按预期返回 True")
        return 1

    # 3. 第一次归档
    print()
    print("=== 第一次 archive() ===")
    pkg = archiver.archive()
    if pkg is None:
        print("TODO_REVIEW: archive 返回 None")
        return 1

    print("  MemoryPackage 全字段：")
    print(f"    seq                    = {pkg.seq}")
    print(f"    emotion_value          = {pkg.emotion_value}")
    print(f"    emotion_polarity       = {pkg.emotion_polarity}")
    print(f"    keywords               = {pkg.keywords}")
    print(f"    raw_pointer            = {pkg.raw_pointer}")
    print(f"    narrative_cause        = {pkg.narrative_cause}")
    print(f"    narrative_trigger      = {pkg.narrative_trigger}")
    print(f"    narrative_consequence  = {pkg.narrative_consequence}")
    print(f"    narrative_evaluation   = {pkg.narrative_evaluation}")
    print(f"    narrative_emotion      = {pkg.narrative_emotion}")
    print(f"    narrative_thought      = {pkg.narrative_thought}")

    # 4. 第一次断言
    print()
    print("=== 第一次断言 ===")
    all_ok = True

    all_ok &= check("缓冲区剩 2 轮", len(buf) == 2, f"实际 = {len(buf)}")
    all_ok &= check("_packages 长度 == 1", len(archiver.get_packages()) == 1,
                    f"实际 = {len(archiver.get_packages())}")
    all_ok &= check("seq == 1", pkg.seq == 1, f"实际 = {pkg.seq}")

    narrative_fields = [
        ("narrative_cause", pkg.narrative_cause),
        ("narrative_trigger", pkg.narrative_trigger),
        ("narrative_consequence", pkg.narrative_consequence),
        ("narrative_evaluation", pkg.narrative_evaluation),
        ("narrative_emotion", pkg.narrative_emotion),
        ("narrative_thought", pkg.narrative_thought),
    ]
    for name, val in narrative_fields:
        all_ok &= check(f"{name} 非空", val != "", "为空字符串")

    all_ok &= check("keywords 非空", len(pkg.keywords) > 0, f"实际 = {pkg.keywords}")
    all_ok &= check("raw_pointer 非空", pkg.raw_pointer != "", "为空字符串")

    if not all_ok:
        print("TODO_REVIEW: 第一次断言失败")
        return 1

    # 5. 再追加 6 轮
    print()
    print("=== 阶段二：再追加 6 轮 ===")
    more = [
        (8, "说真的你挺厉害", "谢谢，你也很棒"),
        (9, "我们换个话题吧", "好呀，聊什么？"),
        (10, "你喜欢什么", "我喜欢思考问题"),
        (11, "那你思考过人生吗", "偶尔会想"),
        (12, "想出来了吗", "还没有答案"),
        (13, "哈哈正常", "嗯，慢慢来"),
    ]
    for rid, user, ai in more:
        buf.append(make_round(rid, user, ai, 1000.0 + rid))

    print(f"  buffer 长度 = {len(buf)}")

    # 6. 第二次归档
    print()
    print("=== 第二次 archive() ===")
    pkg2 = archiver.archive()
    if pkg2 is None:
        print("TODO_REVIEW: 第二次 archive 返回 None")
        return 1

    print(f"  seq = {pkg2.seq}")

    print()
    print("=== 第二次断言 ===")
    all_ok2 = True

    all_ok2 &= check("缓冲区剩 2 轮", len(buf) == 2, f"实际 = {len(buf)}")
    all_ok2 &= check("_packages 长度 == 2", len(archiver.get_packages()) == 2,
                     f"实际 = {len(archiver.get_packages())}")
    all_ok2 &= check("seq == 2", pkg2.seq == 2, f"实际 = {pkg2.seq}")

    if not all_ok2:
        print("TODO_REVIEW: 第二次断言失败")
        return 1

    print()
    print("探针完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
