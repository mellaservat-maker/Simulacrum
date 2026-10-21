# ============================================================
# scripts/probe_embedder.py · M3-T3 第一阶段探针
# 验证可插拔 embedder：维度、语义区分度、hash 兼容性
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embedder import get_embedder
from event_parser import hash_to_8d


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    bge = get_embedder("bge")
    hash_emb = get_embedder("hash")

    print("=== 1. 维度 ===")
    print("get_embedder('bge').dim  =", bge.dim)
    print("get_embedder('hash').dim =", hash_emb.dim)

    t1 = "你真蠢"
    t2 = "你太笨了"
    t3 = "今天天气真好"

    print()
    print("=== 2. 编码三组文字 ===")
    v1 = bge.encode(t1)
    v2 = bge.encode(t2)
    v3 = bge.encode(t3)
    print("encode('%s') shape = %s" % (t1, np.asarray(v1).shape))
    print("encode('%s') shape = %s" % (t2, np.asarray(v2).shape))
    print("encode('%s') shape = %s" % (t3, np.asarray(v3).shape))

    print()
    print("=== 3. 两两 cosine ===")
    c_near = cosine(v1, v2)
    c_far = cosine(v1, v3)
    print("cosine(%s, %s) = %.6f   (预期 > 0.7)" % (t1, t2, c_near))
    print("cosine(%s, %s) = %.6f   (预期 < 0.5)" % (t1, t3, c_far))

    print()
    print("=== 4. 断言 ===")
    assert c_near > 0.7, "近义对相似度应 > 0.7，实际 %.6f" % c_near
    assert c_far < 0.5, "远义对相似度应 < 0.5，实际 %.6f" % c_far
    assert c_near > c_far, "近义对相似度应大于远义对"
    print("近义 > 0.7 : PASS")
    print("远义 < 0.5 : PASS")
    print("前者 > 后者 : PASS")

    print()
    print("=== 5. hash 兼容性（逐元素一致）===")
    h_now = np.asarray(hash_emb.encode(t1))
    h_ref = np.asarray(hash_to_8d(t1))
    print("embedder.encode('%s') = %s" % (t1, h_now.tolist()))
    print("event_parser.hash_to_8d('%s') = %s" % (t1, h_ref.tolist()))
    assert np.array_equal(h_now, h_ref), "hash embedder 与 event_parser.hash_to_8d 不一致"
    print("逐元素完全一致 : PASS")

    print()
    print("全部断言通过")


if __name__ == "__main__":
    main()