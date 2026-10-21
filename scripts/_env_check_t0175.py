# -*- coding: utf-8 -*-
"""环境探测：venv 依赖可用性（017.5 施工前环境确认）"""
import importlib.util
import sys

print("python:", sys.version.split()[0])
for pkg in ("numpy", "torch", "sentence_transformers", "requests", "pytest"):
    print("%s: %s" % (pkg, "OK" if importlib.util.find_spec(pkg) else "MISSING"))
