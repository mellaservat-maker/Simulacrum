# -*- coding: utf-8 -*-
"""embedder 单例缓存验证（017.5 使能基础设施）"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from embedder import get_embedder

a = get_embedder("bge")
b = get_embedder("bge")
c = get_embedder("hash")
d = get_embedder("hash")
print("bge same instance:", a is b)
print("hash same instance:", c is d)
print("cross-type distinct:", a is not c)
