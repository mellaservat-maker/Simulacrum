# ============================================================
# test_archive_backfill.py · 053 号工作
#
# 覆盖归档丢包修复（archiver.archive 失败回灌）：
#   T053-A1 失败时把这批 flushed 记录按原顺序插回缓冲区队首
#   T053-A2 回灌后 should_archive() 仍为真 → 下一轮自然重试
#   T053-A3 失败计数可观测（stats 的 drop_events / retry_backfilled）
#   T053-A4 连续失败超过 ARCHIVE_MAX_RETRY 后放弃本批，但原文仍在 raw_log
#   T053-A5 成功路径不受影响（包照常产出，计数保持 0）
#   T053-A6 LLMUnavailableError 同样走回灌（不打断整轮对话）
# ============================================================

import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import config                      # noqa: E402
from archiver import Archiver      # noqa: E402
from short_term_buffer import ShortTermBuffer   # noqa: E402
from raw_log import RawLog         # noqa: E402
from llm_client import LLMUnavailableError      # noqa: E402


@pytest.fixture(autouse=True)
def _force_mock_backend():
    """成功路径必须走 mock 桩，否则会真的外呼 LLM（测试不可离线、不可复现）。"""
    saved = config.LLM_BACKEND
    config.LLM_BACKEND = "mock"
    try:
        yield
    finally:
        config.LLM_BACKEND = saved


def _fill(buf, n, start=1):
    for i in range(start, start + n):
        buf.append({
            "round_id": i,
            "user_input": "第%d轮输入" % i,
            "ai_output": "第%d轮回复" % i,
            "timestamp": 1700000000.0 + i,
            "intensity": 0.5,
            "valence": -0.2,
            "danger_score": 0.4,
        })


def _rig():
    buf = ShortTermBuffer(default_keep=2)
    raw = RawLog()
    ar = Archiver(buf, raw, threshold=config.MEMORY_ARCHIVE_THRESHOLD)
    return buf, raw, ar


def _boom(records):
    raise ValueError("模拟：归档 LLM 返回非 JSON")


def test_backfill_restores_records_in_order():
    buf, raw, ar = _rig()
    _fill(buf, 6)
    assert len(buf) == 6

    ar.organize_narrative = _boom
    pkg = ar.archive()

    assert pkg is None
    # 回灌：6 轮全部还在，且顺序不乱
    assert len(buf) == 6
    assert [r["round_id"] for r in buf.get_all()] == [1, 2, 3, 4, 5, 6]
    # 原文进 raw_log：只有 flushed 的 4 轮（5/6 两轮按 flush_keep_recent(2)
    #   留在缓冲区，尚未进 raw_log，这是既有设计）
    assert len(raw) == 4
    assert ar.get_packages() == []
    # 计数可观测
    s = ar.stats()
    assert s["drop_events"] == 1
    assert s["retry_backfilled"] == 1
    assert s["giveup_batches"] == 0
    assert s["consecutive_fail"] == 1


def test_backfill_triggers_retry_next_round():
    buf, raw, ar = _rig()
    _fill(buf, 6)
    ar.organize_narrative = _boom
    ar.archive()
    # 回灌后仍达阈值 → 下一轮会再次尝试归档（而不是等到再攒 4 轮）
    assert ar.should_archive() is True

    # 第二次成功：包产出，计数复位，缓冲区只剩最近 2 轮
    del ar.organize_narrative        # 去掉桩，回到真实实现（已强制 mock 后端）
    pkg = ar.archive()
    assert pkg is not None
    assert len(ar.get_packages()) == 1
    assert ar.stats()["consecutive_fail"] == 0
    assert len(buf) == 2
    # 4 轮进了包，2 轮留在缓冲区
    assert pkg.raw_pointer == "1-2-3-4"


def test_giveup_after_max_retry_keeps_raw_log():
    buf, raw, ar = _rig()
    _fill(buf, 6)
    ar.organize_narrative = _boom

    for _ in range(config.ARCHIVE_MAX_RETRY):
        ar.archive()
        # 每次失败都回灌，缓冲区不减
        assert len(buf) == 6

    # 第 MAX+1 次：放弃本批（不再回灌），但原文仍在 raw_log
    ar.archive()
    s = ar.stats()
    assert s["drop_events"] == config.ARCHIVE_MAX_RETRY + 1
    assert s["giveup_batches"] == 1
    assert s["retry_backfilled"] == config.ARCHIVE_MAX_RETRY
    # 放弃后缓冲区只剩最近 2 轮
    assert len(buf) == 2
    # 原文一轮都没丢：flushed 的 4 轮全程在 raw_log（保留的 2 轮本就未写入）
    assert len(raw) == 4
    assert ar.get_packages() == []


def test_success_path_unaffected():
    buf, raw, ar = _rig()
    _fill(buf, 6)
    pkg = ar.archive()
    assert pkg is not None
    s = ar.stats()
    assert s["drop_events"] == 0
    assert s["retry_backfilled"] == 0
    assert s["giveup_batches"] == 0
    assert s["packages"] == 1


def test_llm_unavailable_also_backfills():
    buf, raw, ar = _rig()
    _fill(buf, 6)

    def _unavail(records):
        raise LLMUnavailableError("模拟：通道重试耗尽")

    ar.organize_narrative = _unavail
    assert ar.archive() is None
    assert len(buf) == 6
    assert ar.stats()["retry_backfilled"] == 1


# ---- [054] 任务 3：叙事失败也成包，彻底不丢 ----

def test_narrative_failure_still_produces_package(monkeypatch):
    """LLM 叙事失败时：包照常产出，数值/向量保留，叙事留空，计数 +1。

    053 的路径是抛错 -> 回灌 -> 连续 3 次放弃，那 4 轮就只能躺在 raw_log。
    054 把数值与叙事解耦：叙事只是可选的一层描述，失败不该连累数值。
    """
    import archiver as arch_mod

    monkeypatch.setattr(config, "LLM_BACKEND", "deepseek_official")  # 走 LLM 分支
    monkeypatch.setattr(arch_mod, "get_llm", lambda: object())

    def _boom(llm, prompt, temperature=None, ttl=None):
        raise ValueError("模拟：归档 LLM 返回非 JSON")

    monkeypatch.setattr(arch_mod, "cached_generate", _boom)

    buf, raw, ar = _rig()
    _fill(buf, 6)
    pkg = ar.archive()

    assert pkg is not None, "叙事失败不该丢掉整包"
    assert pkg.raw_pointer == "1-2-3-4"
    assert pkg.n_rounds == 4
    # 数值来自主链路聚合，与 LLM 无关
    assert pkg.emotion_value > 0.0
    assert pkg.mean_danger is not None and pkg.mean_danger > 0.0
    assert pkg.mean_vec is not None and len(pkg.mean_vec) == 1024
    # 叙事留空，但留痕可查
    assert pkg.narrative_cause == ""
    assert "叙事缺失" in pkg.narrative_thought
    assert ar.stats()["narrative_fallback"] == 1
    # 没有走回灌路径：这批记录正常成包了
    assert ar.stats()["drop_events"] == 0
    assert len(buf) == 2
