# -*- coding: utf-8 -*-
"""LLM 通道存活探测（017.5 施工前确认）：evaluator 提取 + archiver 缓存统计"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluator import extract_semantic_signals
from llm_client import get_llm, cache_stats

S, src = extract_semantic_signals("你这个AI真蠢，什么都做不好。")
print("S =", S, "source =", src)
S2, src2 = extract_semantic_signals("你刚才的回答太棒了，帮了大忙！")
print("S2 =", S2, "source =", src2)
print("cache_stats =", cache_stats())
