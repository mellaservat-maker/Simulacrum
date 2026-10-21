# ============================================================
# probe_decompressor.py · M2-C 探针
# 验证 RawLog + Archiver + Decompressor 全链路
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from short_term_buffer import ShortTermBuffer
from raw_log import RawLog
from archiver import Archiver
from decompressor import Decompressor


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
    raw_log = RawLog()
    archiver = Archiver(buf, raw_log, threshold=6)
    decomp = Decompressor(raw_log)

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

    print(f"=== 追加 7 轮，buffer 长度 = {len(buf)} ===")

    # 2. 归档
    print()
    print("=== 执行归档 ===")
    pkg = archiver.archive()
    if pkg is None:
        print("TODO_REVIEW: archive 返回 None")
        return 1
    print(f"  seq = {pkg.seq}")
    print(f"  raw_pointer = {pkg.raw_pointer}")
    print(f"  buffer 剩余 = {len(buf)} 轮")

    # 3. 检查 raw_log 里有 5 轮
    print()
    print("=== raw_log 检查 ===")
    print(f"  raw_log 长度 = {len(raw_log)}")
    print(f"  all_ids = {raw_log.all_ids()}")

    all_ok = True
    all_ok &= check("raw_log 有 5 轮", len(raw_log) == 5, f"实际 = {len(raw_log)}")
    all_ok &= check("all_ids = [1,2,3,4,5]", raw_log.all_ids() == [1, 2, 3, 4, 5],
                    f"实际 = {raw_log.all_ids()}")

    # 4. 四个器官专用方法
    print()
    print("=== for_emotion_evaluator ===")
    r = decomp.for_emotion_evaluator(pkg)
    print(f"  emotion_value    = {r['emotion_value']}")
    print(f"  emotion_polarity = {r['emotion_polarity']}")
    all_ok &= check("有 emotion_value", "emotion_value" in r)
    all_ok &= check("有 emotion_polarity", "emotion_polarity" in r)

    print()
    print("=== for_danger_evaluator ===")
    r = decomp.for_danger_evaluator(pkg)
    print(f"  consequence      = {r['consequence']}")
    print(f"  evaluation       = {r['evaluation']}")
    print(f"  raw 条数         = {len(r['raw'])}")
    all_ok &= check("有 consequence", "consequence" in r)
    all_ok &= check("有 evaluation", "evaluation" in r)
    all_ok &= check("raw 有 5 条", len(r["raw"]) == 5, f"实际 = {len(r['raw'])}")

    print()
    print("=== for_policy ===")
    r = decomp.for_policy(pkg)
    print(f"  consequence      = {r['consequence']}")
    print(f"  thought          = {r['thought']}")
    print(f"  raw 条数         = {len(r['raw'])}")
    all_ok &= check("有 consequence", "consequence" in r)
    all_ok &= check("有 thought", "thought" in r)
    all_ok &= check("raw 有 5 条", len(r["raw"]) == 5, f"实际 = {len(r['raw'])}")

    print()
    print("=== for_xinguang ===")
    r = decomp.for_xinguang(pkg)
    print(f"  keywords         = {r['keywords']}")
    print(f"  emotion_value    = {r['emotion_value']}")
    all_ok &= check("有 keywords", "keywords" in r)
    all_ok &= check("有 emotion_value", "emotion_value" in r)

    # 5. get_fields 通用方法
    print()
    print("=== get_fields(pkg, ['emotion_value', 'raw']) ===")
    r = decomp.get_fields(pkg, ["emotion_value", "raw"])
    print(f"  emotion_value    = {r['emotion_value']}")
    print(f"  raw 条数         = {len(r['raw'])}")
    all_ok &= check("emotion_value 正确", r["emotion_value"] == 0.5, f"实际 = {r['emotion_value']}")
    all_ok &= check("raw 返回 5 条原文", len(r["raw"]) == 5, f"实际 = {len(r['raw'])}")

    # 6. 解压器返回的 raw 内容与 raw_log 里存的完全一致
    print()
    print("=== raw 内容一致性检查 ===")
    danger_raw = decomp.for_danger_evaluator(pkg)["raw"]
    log_records = raw_log.get_many([1, 2, 3, 4, 5])
    match = True
    for i in range(5):
        if danger_raw[i] is not log_records[i]:
            # 不是同一个对象没关系，比较内容
            if danger_raw[i]["round_id"] != log_records[i]["round_id"]:
                match = False
                break
            if danger_raw[i]["user_input"] != log_records[i]["user_input"]:
                match = False
                break
    all_ok &= check("解压器 raw 与 raw_log 内容一致", match)

    # 7. 解析失败场景：空 raw_pointer 返回空列表
    print()
    print("=== 解析失败场景 ===")
    from schemas import MemoryPackage
    bad_pkg = MemoryPackage(raw_pointer="")
    bad_raw = decomp.for_danger_evaluator(bad_pkg)["raw"]
    all_ok &= check("空 raw_pointer 返回空列表", bad_raw == [], f"实际 = {bad_raw}")

    bad_pkg2 = MemoryPackage(raw_pointer="abc-xyz")
    bad_raw2 = decomp.for_danger_evaluator(bad_pkg2)["raw"]
    all_ok &= check("非法格式返回空列表", bad_raw2 == [], f"实际 = {bad_raw2}")

    print()
    if all_ok:
        print("探针完成，全部断言通过")
        return 0
    else:
        print("TODO_REVIEW: 断言失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
