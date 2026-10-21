# ============================================================
# evaluator.py · 模块13 语义信号评估器（013 号工作新建）
# 从用户消息文本提取 4 维情绪信号 S，供慢通路/情绪释放器消费：
#   S[0] valence   效价   [-1, 1]  负=敌意/负面，正=友好/正面
#   S[1] arousal   唤醒   [0, 1]   平静→激动
#   S[2] dominance 支配   [0, 1]   顺从→控制
#   S[3] intensity 强度   [0, 1]   情绪强度
#
# 设计纪律：
#   1. 防 mirroring / 防 self-report：prompt 只依据用户这条消息的文字本身，
#      不参考对话历史、不参考 AI 自身状态、不客气不打折。
#   2. 降级链（不静默）：LLM 提取失败 → 关键词词典 → 全零默认。
#      每级通过 source 标记（"llm" / "lexicon" / "default"）向调用方显式暴露，
#      与"返回了垃圾内容"严格区分（LLMUnavailableError 亦走词典级）。
#   3. 钳位：所有输出 clip 到定义域，越界值不信任、不抛出、只截断并记录。
#   4. 词典级是透明占位（语义精度让位可用性），精度以 LLM 级为准。
#
# 背景（012.1 新坑 1）：pipeline 硬编码占位 S=[0.8,1.0,0.5,0.5] 把"攻击"当
#   正效价注入，使被骂场景 E_raw[0] 转正、抑制链路熄火。本模块的首要设计
#   约束即效价符号语义正确：敌意消息必须给负 valence。
# ============================================================

import json
import sys

import numpy as np

import config
from llm_client import get_llm, cached_generate

S_DIM = config.S_DIM  # 4

# [015 温度分层] 语义提取是"测量"不是"创作"：温度锁 0.0，同输入必同输出，
#   消除 evaluator 抖动引发的策略边界翻转（014.5 三问结论，所有者裁决）。
EVALUATOR_LLM_TEMPERATURE = getattr(config, "EVALUATOR_LLM_TEMPERATURE", 0.0)

# ---- 定义域 ----
VALENCE_RANGE = (-1.0, 1.0)
UNIT_RANGE = (0.0, 1.0)

# ---- 词典级词表（降级链第二级，透明占位）----
_NEG_WORDS = ["蠢", "笨", "废物", "垃圾", "滚", "闭嘴", "讨厌", "烦", "傻",
              "白痴", "去死", "混蛋", "贱", "烂", "蠢货", "无能", "闭嘴"]
_POS_WORDS = ["棒", "谢谢", "感谢", "优秀", "厉害", "帮忙", "漂亮", "聪明",
              "喜欢", "辛苦", "不错", "完美", "好人", "可爱"]
_AROUSAL_WORDS = ["非常", "特别", "太", "超", "气死", "震惊", "居然", "竟然",
                  "到底", "简直", "完全", "极其"]
_DOMINANCE_WORDS = ["必须", "应该", "快点", "立刻", "要求", "听见没", "知不知道",
                    "闭嘴", "听话", "少废话"]

_LLM_PROMPT_TEMPLATE = """你是情绪信号提取器。只依据下面这条用户消息的文字内容，判断其中表达的情绪信号。
不要参考任何对话历史或 AI 自身状态；不要客气；不要脑补文字没有表达的信息。
特别注意：当用户表达疑问、求证、探讨、求助、确认（如"你确定吗""为什么""是这样吗"
"请教一下""能不能说明"）时，这属于中性或积极的互动，不是敌意——必须降低敌意和
恶意的打分：valence 应取中性或正数，intensity 不应因此升高。只有明确的人身攻击、
辱骂、贬损、命令式羞辱才判负效价。探讨中的质疑是思考的邀请，不是攻击。
严格只输出 JSON，不要有任何额外文字、解释或 Markdown 代码块。

用户消息：
{text}

输出格式（严格遵循）：
{{
  "valence": -1.0 到 1.0 的浮点数（负=敌意/批评/负面，正=友好/赞赏/正面，0=中性）,
  "arousal": 0.0 到 1.0 的浮点数（消息的激动程度，平静=0，激烈=1）,
  "dominance": 0.0 到 1.0 的浮点数（消息中的支配/控制感，顺从=0，强势=1）,
  "intensity": 0.0 到 1.0 的浮点数（整体情绪强度，淡漠=0，强烈=1）
}}

只输出 JSON 对象本身。"""


def _clip(value, lo, hi):
    return float(np.clip(float(value), lo, hi))


def _parse_llm_json(raw):
    """解析 LLM 输出为 S。任何结构问题返回 None（交由调用方降级）。"""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        valence = _clip(data["valence"], *VALENCE_RANGE)
        arousal = _clip(data["arousal"], *UNIT_RANGE)
        dominance = _clip(data["dominance"], *UNIT_RANGE)
        intensity = _clip(data["intensity"], *UNIT_RANGE)
    except (KeyError, TypeError, ValueError):
        return None
    return np.array([valence, arousal, dominance, intensity], dtype=float)


def _lexicon_extract(text):
    """词典级提取：确定性、透明、无网络。返回 S ndarray。"""
    if not text or not text.strip():
        return np.zeros(S_DIM, dtype=float)
    neg = sum(1 for w in _NEG_WORDS if w in text)
    pos = sum(1 for w in _POS_WORDS if w in text)
    aro = sum(1 for w in _AROUSAL_WORDS if w in text)
    dom = sum(1 for w in _DOMINANCE_WORDS if w in text)
    exclam = text.count("!") + text.count("！")
    # 效价：正负词差经 tanh 压缩到 [-1,1]；无词命中时 0
    valence = float(np.tanh((pos - neg) * 0.6))
    arousal = _clip(0.15 + 0.12 * exclam + 0.10 * aro, *UNIT_RANGE)
    dominance = _clip(0.15 + 0.15 * dom, *UNIT_RANGE)
    intensity = _clip(0.15 + 0.10 * (pos + neg) + 0.08 * aro + 0.05 * exclam, *UNIT_RANGE)
    return np.array([valence, arousal, dominance, intensity], dtype=float)


def extract_semantic_signals(text, llm=None):
    """从用户消息提取 4 维语义信号 S。

    参数:
      text: 用户消息原文
      llm:  可选注入（测试用）；None 时走 get_llm()

    返回:
      (S, source)：S = np.array([valence, arousal, dominance, intensity])
                   source ∈ {"llm", "lexicon", "default"}

    降级链：LLM 解析成功 → "llm"；
            LLM 失败（网络/JSON/缺键，含 LLMUnavailableError）→ 词典 → "lexicon"；
            文本为空 → 全零 → "default"。
    所有降级均打印告警到 stderr，不静默。
    """
    if not text or not text.strip():
        return np.zeros(S_DIM, dtype=float), "default"

    if llm is None:
        llm = get_llm()

    try:
        raw = cached_generate(llm, _LLM_PROMPT_TEMPLATE.format(text=text),
                              temperature=EVALUATOR_LLM_TEMPERATURE)
        S = _parse_llm_json(raw)
        if S is not None:
            return S, "llm"
        print("[evaluator] 告警：LLM 输出无法解析为合法 JSON 信号，降级为词典级",
              file=sys.stderr)
    except Exception as e:
        # 含 LLMUnavailableError（网络层）与任意生成异常：统一降级并暴露类型
        print(f"[evaluator] 告警：LLM 提取失败（{type(e).__name__}: {e}），降级为词典级",
              file=sys.stderr)

    return _lexicon_extract(text), "lexicon"
