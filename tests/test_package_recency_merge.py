# ============================================================
# test_package_recency_merge.py · 053 号工作
#
# 覆盖阶段 1 的两项读取侧改动（都不需要 BGE，向量手工构造）：
#   T053-B1 包级时间衰减：同语义下，新包赢过旧包
#   T053-B2 衰减不改变 raw 分（matched_sim_raw 与 recency_decay 可反推）
#   T053-B3 去重合并：mean_vec 余弦 > 阈值时并入旧包，不新增包
#   T053-B4 合并后 pointer 拼接、n_rounds 相加、峰值取情绪更强的一侧
#   T053-B5 低于阈值时不合并（包数正常增长）
# ============================================================

import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config                       # noqa: E402
import pipeline                     # noqa: E402
from archiver import Archiver       # noqa: E402
from schemas import MemoryPackage   # noqa: E402
from short_term_buffer import ShortTermBuffer   # noqa: E402
from raw_log import RawLog          # noqa: E402
from state import reset_state       # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _pkg(seq, pointer, vec, ev=0.5, pol="neutral", n=4,
         mean_val=0.0, mean_danger=0.4, max_danger=0.6):
    return MemoryPackage(
        seq=seq, emotion_value=ev, emotion_polarity=pol,
        raw_pointer=pointer, mean_vec=_unit(vec), peak_vec=_unit(vec),
        mean_valence=mean_val, n_rounds=n,
        mean_danger=mean_danger, max_danger=max_danger,
    )


class _StubDecomp:
    """只提供 _recall_match 需要的 package_max_round。"""

    @staticmethod
    def package_max_round(pkg):
        ids = [int(x) for x in (pkg.raw_pointer or "").split("-") if x.isdigit()]
        return max(ids) if ids else None

    @staticmethod
    def for_emotion_evaluator_matched(pkg, now_vec=None, now_valence=None, **kw):
        return {
            "emotion_value": pkg.emotion_value,
            "emotion_polarity": pkg.emotion_polarity,
            "recall_source": "package_level",
            "recalled_rounds": [],
        }


class _StubEmbedder:
    """把任意查询都编码成同一个固定向量，使 raw 余弦可精确预期。

    必须替换 pipeline._pkg_embedder：_recall_match 内部用真实 BGE 编码
    查询文本，手工构造的 1024 维基向量与 BGE 空间无关，余弦会趋近 0。
    """

    def __init__(self, vec):
        self._v = _unit(vec)

    def encode(self, texts):
        if isinstance(texts, str):
            return self._v
        return np.stack([self._v for _ in texts])


@pytest.fixture
def stub_embedder(monkeypatch):
    def _make(vec):
        monkeypatch.setattr(pipeline, "_pkg_embedder", _StubEmbedder(vec))
    return _make


def _state():
    st = reset_state(seed=42)
    st["decompressor"] = _StubDecomp()
    return st


# ---- T053-B1 / B2：时间衰减 ----

def test_recency_decay_prefers_newer_package(stub_embedder):
    """同一查询、同语义强度：旧包因衰减输给新包。"""
    base = np.zeros(1024, dtype=np.float32)
    base[0] = 1.0
    stub_embedder(base)
    old = _pkg(1, "1-2-3-4", base)              # 包内最大轮 4，age=36
    new = _pkg(2, "37-38-39-40", base)          # 包内最大轮 40，age=0
    st = _state()

    got = pipeline._recall_match(st, [old, new], "任意查询", 40,
                                 now_valence=0.0)
    assert got is not None
    assert got["matched_pkg_seq"] == 2, "新包应赢得 argmax"
    assert abs(got["recency_decay"] - 1.0) < 1e-9
    assert abs(got["matched_sim_raw"] - 1.0) < 1e-6


def test_decay_keeps_raw_sim_recoverable(stub_embedder):
    """matched_sim = raw × decay × (1 - penalty)，三项都要能反推。"""
    base = np.zeros(1024, dtype=np.float32)
    base[0] = 1.0
    stub_embedder(base)
    old = _pkg(1, "1-2-3-4", base, mean_val=0.0)   # age=10 → decay≈0.607
    st = _state()
    got = pipeline._recall_match(st, [old], "任意查询", 14, now_valence=0.0)
    assert got is not None, "decay≈0.607 应仍高于 TAU_FLOOR=0.45"
    raw = got["matched_sim_raw"]
    decay = got["recency_decay"]
    pen = got["polarity_penalty"]
    assert abs(got["matched_sim"] - raw * decay * (1.0 - pen)) < 1e-6
    assert decay < 1.0, "age>0 时衰减必须小于 1"
    assert abs(decay - float(np.exp(-10.0 / 20.0))) < 1e-6


def test_decay_does_not_block_recall(stub_embedder):
    """[054] 解耦后：极旧包仍然会被召回，只是排序垫底。

    053 的口径是衰减乘进 sim_eff 后再比地板，age 396 轮时 decay≈2e-9，
    要过 0.45 需要原始相似度 2e8，数学上不可能 —— 老记忆等于被删除。
    054 把地板改为只判 sim_raw：相关性由语义决定，新鲜度只影响排序。
    本用例断言的就是这个语义。
    """
    base = np.zeros(1024, dtype=np.float32)
    base[0] = 1.0
    stub_embedder(base)
    very_old = _pkg(1, "1-2-3-4", base)
    st = _state()
    got = pipeline._recall_match(st, [very_old], "任意查询", 400,
                                 now_valence=0.0)
    assert got is not None, "解耦后极旧包不该被时间挡掉"
    assert got["recency_decay"] < 0.01, "但它在排序上应该几乎垫底"
    assert got["matched_sim_raw"] > config.TAU_FLOOR, "相关性闸应通过"


def test_floor_still_blocks_irrelevant(stub_embedder):
    """相关性闸仍然只看语义：全新的包，只要不相关就照样不召回。"""
    base = np.zeros(1024, dtype=np.float32)
    base[0] = 1.0
    stub_embedder(base)
    # 与查询向量正交 → 余弦 0，远低于 TAU_FLOOR
    orth = np.zeros(1024, dtype=np.float32)
    orth[1] = 1.0
    fresh_but_irrelevant = _pkg(1, "97-98-99-100", orth)   # age=0，decay=1
    st = _state()
    got = pipeline._recall_match(st, [fresh_but_irrelevant], "任意查询", 100,
                                 now_valence=0.0)
    assert got is None, "语义无关的包即便最新也不该召回"


# ---- T053-B3 / B4 / B5：去重合并 ----

def _rig():
    buf = ShortTermBuffer(default_keep=2)
    raw = RawLog()
    return buf, raw, Archiver(buf, raw, threshold=6)


def _feed(buf, start, n=4, intensity=0.5, valence=-0.2, danger=0.4):
    for i in range(start, start + n):
        buf.append({
            "round_id": i, "user_input": "第%d轮输入" % i,
            "ai_output": "回复%d" % i, "timestamp": 1700000000.0 + i,
            "intensity": intensity, "valence": valence, "danger_score": danger,
        })


def test_duplicate_package_is_merged(monkeypatch):
    """mean_vec 余弦 1.0（同文本）→ 不新增包，并入旧包。"""
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")
    buf, raw, ar = _rig()
    _feed(buf, 1)
    p1 = ar.archive()
    assert p1 is not None and len(ar.get_packages()) == 1

    _feed(buf, 5)
    p2 = ar.archive()
    assert p2 is not None
    assert len(ar.get_packages()) == 1, "同主题应合并而不是新增"
    assert ar.stats()["merged_into"] == 1
    # pointer 拼接、n_rounds 相加。
    #   注意重叠：首次归档 flush 掉 1-4 但保留 3-4 在缓冲区，
    #   第二包因此是 3-6，合并后为 1-2-3-4-5-6（不是 1-8）。
    assert ar.get_packages()[0].raw_pointer == "1-2-3-4-5-6"
    assert ar.get_packages()[0].n_rounds == 6


def test_merge_keeps_stronger_peak(monkeypatch):
    """合并时峰值取情绪更强的一侧，并同步 emotion_value / polarity。"""
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")
    buf, raw, ar = _rig()
    _feed(buf, 1, intensity=0.2, valence=0.3)
    ar.archive()
    first_ev = ar.get_packages()[0].emotion_value

    _feed(buf, 5, intensity=0.9, valence=-0.8)
    ar.archive()
    pkg = ar.get_packages()[0]
    assert pkg.emotion_value > first_ev
    assert abs(pkg.emotion_value - 0.9) < 1e-9
    assert pkg.emotion_polarity == "negative"


def test_different_packages_do_not_merge(monkeypatch):
    """主题不同（余弦远低于阈值）→ 正常新增包。"""
    monkeypatch.setattr(config, "LLM_BACKEND", "mock")
    buf, raw, ar = _rig()
    # 两个正交向量：余弦 0
    orig = Archiver._compute_pkg_vectors

    calls = {"n": 0}

    def fake_vectors(self, records):
        calls["n"] += 1
        v = np.zeros(1024, dtype=np.float32)
        v[0 if calls["n"] == 1 else 1] = 1.0
        return v, v.copy(), 0.0

    ar._compute_pkg_vectors = fake_vectors.__get__(ar, Archiver)
    _feed(buf, 1)
    ar.archive()
    _feed(buf, 5)
    ar.archive()
    assert len(ar.get_packages()) == 2
    assert ar.stats()["merged_into"] == 0
