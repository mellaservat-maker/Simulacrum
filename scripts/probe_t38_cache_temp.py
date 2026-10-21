# ============================================================
# 015 任务 1 探针：温度分层与同输入缓存闭环
# 剧本（全离线，mock LLM 计数，不碰真网络）：
#   1. 缓存命中：同 (temperature, prompt) 第二次不调 LLM
#   2. TTL 过期：越过 TTL 后重新调用 LLM
#   3. 温度入 key：同 prompt 不同 temperature 不互相命中
#   4. evaluator 实际使用温度 0.0（透传验证）
#   5. archiver 实际使用温度 0.3（透传验证）
#   6. 失败不入缓存：LLM 抛异常后下次仍重试
# ============================================================
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import llm_client
from llm_client import cached_generate, cache_clear, cache_stats
import config


def check(name, cond, detail=""):
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))
    if not cond:
        check.failed += 1
check.failed = 0


class CountingLLM:
    """记录调用次数与收到 temperature 的 mock。"""

    def __init__(self, reply='{"valence": 0.5, "arousal": 0.5, "dominance": 0.5, "intensity": 0.5}'):
        self.calls = []
        self.reply = reply

    def generate(self, prompt, temperature=None):
        self.calls.append((prompt, temperature))
        return self.reply


class FlakyLLM:
    """第一次抛 LLMUnavailableError，之后成功——验证失败不入缓存。"""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, temperature=None):
        self.calls += 1
        if self.calls == 1:
            raise llm_client.LLMUnavailableError("模拟瞬时故障")
        return "ok"


def main():
    print("=== 015 任务 1：温度分层与同输入缓存闭环 ===")
    print("config：EVALUATOR_LLM_TEMPERATURE=%s，ARCHIVER_LLM_TEMPERATURE=%s，"
          "LLM_CACHE_TTL_SECONDS=%s" % (
              getattr(config, "EVALUATOR_LLM_TEMPERATURE", "<缺失>"),
              getattr(config, "ARCHIVER_LLM_TEMPERATURE", "<缺失>"),
              getattr(config, "LLM_CACHE_TTL_SECONDS", "<缺失>")))
    print()

    # ---------- 1. 缓存命中 ----------
    print("=== 1. 缓存命中（同 temperature+prompt 第二次零调用）===")
    cache_clear()
    m = CountingLLM()
    p = "同一句话"
    r1 = cached_generate(m, p, temperature=0.0, ttl=60)
    r2 = cached_generate(m, p, temperature=0.0, ttl=60)
    print("两次调用后 LLM 实际被调 %d 次，结果一致=%s" % (len(m.calls), r1 == r2))
    check("1. 同 key 第二次命中缓存（LLM 只调 1 次）", len(m.calls) == 1 and r1 == r2)
    print()

    # ---------- 2. TTL 过期 ----------
    print("=== 2. TTL 过期（ttl=0.3s，睡 0.4s 后重新调用）===")
    m2 = CountingLLM(reply="ok")
    cached_generate(m2, "过期测试", temperature=0.0, ttl=0.3)
    time.sleep(0.4)
    cached_generate(m2, "过期测试", temperature=0.0, ttl=0.3)
    print("TTL 过期后 LLM 被调 %d 次（应 2）" % len(m2.calls))
    check("2. TTL 过期后重新调用 LLM", len(m2.calls) == 2)
    print()

    # ---------- 3. 温度入 key ----------
    print("=== 3. 温度参与缓存 key ===")
    m3 = CountingLLM()
    cached_generate(m3, "同文不同温", temperature=0.0, ttl=60)
    cached_generate(m3, "同文不同温", temperature=0.3, ttl=60)
    cached_generate(m3, "同文不同温", temperature=0.0, ttl=60)
    print("三次调用（0.0/0.3/0.0）后 LLM 被调 %d 次（应 2：0.3 未命中 0.0）" % len(m3.calls))
    print("  收到的 temperature 序列: %s" % [t for _, t in m3.calls])
    check("3. 不同 temperature 不互相命中（LLM 调 2 次）", len(m3.calls) == 2)
    print()

    # ---------- 4/5. evaluator 与 archiver 的实际温度 ----------
    print("=== 4. evaluator 端到端温度验证（注入 CountingLLM）===")
    from evaluator import extract_semantic_signals
    cache_clear()
    me = CountingLLM(reply='{"valence": -0.9, "arousal": 0.8, "dominance": 0.9, "intensity": 0.9}')
    S, src = extract_semantic_signals("测试敌意", llm=me)
    got_temp = me.calls[0][1] if me.calls else None
    print("evaluator 实际传入 temperature=%s（应 0.0），S=%s, source=%s" % (got_temp, S, src))
    check("4. evaluator 使用温度 0.0", got_temp == 0.0)
    # 同输入第二次应命中缓存（evaluator 端到端缓存验证）
    S2, _ = extract_semantic_signals("测试敌意", llm=me)
    print("evaluator 第二次同输入后 LLM 调用 %d 次（应仍 1，缓存生效）" % len(me.calls))
    check("4b. evaluator 同输入命中缓存", len(me.calls) == 1)
    print()

    print("=== 5. archiver 端到端温度验证（注入 CountingLLM）===")
    import archiver
    from short_term_buffer import ShortTermBuffer
    from raw_log import RawLog
    from state import reset_state
    cache_clear()
    # 构造最小 archiver：直接测 organize_narrative 的真 LLM 路径温度透传
    records = [{"round_id": 1, "user_input": "你好", "ai_output": "你好呀"}]
    orig_get_llm = archiver.get_llm
    ma = CountingLLM(reply='{"emotion_value": 0.5, "emotion_polarity": "neutral", '
                           '"keywords": ["k1"], "narrative_cause": "c", "narrative_trigger": "t", '
                           '"narrative_consequence": "r", "narrative_evaluation": "e", '
                           '"narrative_emotion": "n", "narrative_thought": "th"}')
    archiver.get_llm = lambda: ma
    # 临时确保非 mock 后端走真 LLM 分支
    orig_backend = config.LLM_BACKEND
    config.LLM_BACKEND = "deepseek_official"
    try:
        buf = ShortTermBuffer()
        raw_log = RawLog()
        arch = archiver.Archiver(buf, raw_log, threshold=1)
        pkg = arch.organize_narrative(records)
        got_temp_a = ma.calls[0][1] if ma.calls else None
        print("archiver 实际传入 temperature=%s（应 0.3），归档 emotion_value=%s"
              % (got_temp_a, pkg.emotion_value))
        check("5. archiver 使用温度 0.3", got_temp_a == 0.3)
    finally:
        archiver.get_llm = orig_get_llm
        config.LLM_BACKEND = orig_backend
    print()

    # ---------- 6. 失败不入缓存 ----------
    print("=== 6. 失败不入缓存 ===")
    cache_clear()
    f = FlakyLLM()
    raised = False
    try:
        cached_generate(f, "会失败一次", temperature=0.0, ttl=60)
    except llm_client.LLMUnavailableError:
        raised = True
    r = cached_generate(f, "会失败一次", temperature=0.0, ttl=60)
    print("第一次抛异常=%s，第二次拿到结果=%r，LLM 共调 %d 次" % (raised, r, f.calls))
    check("6. 失败不入缓存（第二次重新尝试并成功）", raised and r == "ok" and f.calls == 2)
    print()

    stats = cache_stats()
    print("缓存统计：%s" % stats)
    print()
    print("=== 结论: %s（失败 %d 项）===" % ("全部通过" if check.failed == 0 else "存在失败", check.failed))
    return 0 if check.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
