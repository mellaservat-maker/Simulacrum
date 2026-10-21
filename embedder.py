# ============================================================
# embedder.py · V4.5 · 可插拔嵌入器（M3-T3 第一阶段）
# 只新建，不改动任何现有 .py 文件
# 接口：
#   get_embedder(name="bge") 工厂函数
#   每个实例有 dim 属性与 encode(texts) 方法
#   HashEmbedder：复刻 event_parser.hash_to_8d 的 8 维 hash
#   BGEEmbedder ：BGE-Large-zh-v1.5，1024 维，懒加载，GPU 优先
# 默认后端：BGE
# ============================================================

import hashlib

import numpy as np


class HashEmbedder:
    """复刻 event_parser.hash_to_8d 的 8 维 hash 嵌入，输出必须完全一致。"""

    dim = 8

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = np.stack([self._hash_to_8d(t) for t in items])
        return vecs[0] if single else vecs

    @staticmethod
    def _hash_to_8d(raw_text):
        h = hashlib.md5(raw_text.encode()).digest()
        vec = np.array([h[i] / 255.0 for i in range(8)])
        return vec / (np.linalg.norm(vec) + 1e-8)


class BGEEmbedder:
    """BGE-Large-zh-v1.5 语义嵌入，1024 维，懒加载，GPU 优先。"""

    dim = 1024
    MODEL_NAME = "BAAI/bge-large-zh-v1.5"

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts):
        self._load()
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = self._model.encode(
            items, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs[0] if single else vecs


# [017.5] 实例缓存：同名 embedder 全系统共用一个实例。
#   背景：event_parser / archiver / decompressor / pipeline 各自持有模块级
#   单例（BGEEmbedder 懒加载，构造轻量但 _load 是一份 2.5G 模型），若无缓存，
#   每模块各加载一份 → 内存翻倍且加载耗时叠加。缓存只共享实例，不改变任何
#   编码语义（encode 为无状态纯函数式调用）。
_INSTANCE_CACHE = {}


def _new_embedder(key):
    """按 key 构造全新实例（无缓存，仅供 get_embedder 首次创建用）。"""
    if key in ("bge", "bge-large", "bge-large-zh-v1.5"):
        return BGEEmbedder()
    if key in ("hash", "hash8", "hash_to_8d"):
        return HashEmbedder()
    raise ValueError(f"Unknown embedder name: {key}")


def get_embedder(name="bge"):
    """工厂函数，默认后端为 BGE。[017.5] 同名实例全系统共享（见 _INSTANCE_CACHE）。"""
    key = (name or "bge").lower()
    inst = _INSTANCE_CACHE.get(key)
    if inst is None:
        inst = _new_embedder(key)
        _INSTANCE_CACHE[key] = inst
    return inst