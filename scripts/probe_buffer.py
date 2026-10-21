# ============================================================
# probe_buffer.py · M2-A 探针
# 验证 ShortTermBuffer 的追加 / 读取 / 刷出逻辑
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from short_term_buffer import ShortTermBuffer


def make_round(rid, user, ai, ts):
    return {
        "round_id": rid,
        "user_input": user,
        "ai_output": ai,
        "timestamp": ts,
    }


def main():
    buf = ShortTermBuffer(default_keep=2)

    # 1. 追加 5 轮
    buf.append(make_round(1, "你好", "你好呀，有什么事吗？", 1000.0))
    buf.append(make_round(2, "你今天怎么样", "还行吧，就那样", 1001.0))
    buf.append(make_round(3, "你这都做不好", "……抱歉，我再试试", 1002.0))
    buf.append(make_round(4, "做得不错", "谢谢夸奖，我会继续努力的", 1003.0))
    buf.append(make_round(5, "还有个问题", "你说", 1004.0))

    print(f"追加 5 轮后，buffer 长度 = {len(buf)}")

    # 2. 打印全部内容
    print()
    print("=== 全部内容 ===")
    for r in buf.get_all():
        print(f"  round {r['round_id']}: user='{r['user_input']}' ai='{r['ai_output']}' ts={r['timestamp']}")

    # 3. 刷出，保留最近 2 轮
    flushed, remaining = buf.flush_keep_recent(2)

    print()
    print("=== 被清掉的部分（前 3 轮） ===")
    for r in flushed:
        print(f"  round {r['round_id']}: user='{r['user_input']}'")

    print()
    print("=== 保留的部分（后 2 轮） ===")
    for r in remaining:
        print(f"  round {r['round_id']}: user='{r['user_input']}'")

    # 4. 断言
    print()
    print("=== 断言检查 ===")

    ok = True

    # 清掉 3 轮
    if len(flushed) != 3:
        print(f"TODO_REVIEW: 被清掉的轮数 = {len(flushed)}，期望 3")
        ok = False
    else:
        print(f"  被清掉轮数 = {len(flushed)} ✓")

    # 保留 2 轮
    if len(remaining) != 2:
        print(f"TODO_REVIEW: 保留轮数 = {len(remaining)}，期望 2")
        ok = False
    else:
        print(f"  保留轮数 = {len(remaining)} ✓")

    # 保留的是最后 2 轮（round 4 和 5）
    if remaining[0]["round_id"] != 4 or remaining[1]["round_id"] != 5:
        print(f"TODO_REVIEW: 保留的不是最后 2 轮，实际 round_id = {[r['round_id'] for r in remaining]}")
        ok = False
    else:
        print(f"  保留的是最后 2 轮（round 4, 5） ✓")

    # buffer 里剩 2 条
    if len(buf) != 2:
        print(f"TODO_REVIEW: 刷出后 buffer 长度 = {len(buf)}，期望 2")
        ok = False
    else:
        print(f"  刷出后 buffer 长度 = {len(buf)} ✓")

    # 被清掉的是前 3 轮（round 1, 2, 3）
    flushed_ids = [r["round_id"] for r in flushed]
    if flushed_ids != [1, 2, 3]:
        print(f"TODO_REVIEW: 被清掉的不是前 3 轮，实际 round_id = {flushed_ids}")
        ok = False
    else:
        print(f"  被清掉的是前 3 轮（round 1, 2, 3） ✓")

    # get_recent 默认 n=2
    recent = buf.get_recent()
    if len(recent) != 2:
        print(f"TODO_REVIEW: get_recent() 默认返回 {len(recent)} 条，期望 2")
        ok = False
    else:
        print(f"  get_recent() 默认返回 2 条 ✓")

    print()
    if ok:
        print("断言全部通过")
    else:
        print("TODO_REVIEW: 断言失败")

    print()
    print("探针完成")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
