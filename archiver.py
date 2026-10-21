# ============================================================
# archiver.py · M2-C · 归档器（带桩 + RawLog 写入）
# 把短期缓冲区吐出的记录整理成 MemoryPackage，并写入原始日志
# ============================================================

import sys
from typing import List, Dict, Any, Optional

import numpy as np

from schemas import MemoryPackage
from short_term_buffer import ShortTermBuffer
from raw_log import RawLog
import config
from llm_client import get_llm, cached_generate
from embedder import get_embedder

# [015 温度分层] 记忆叙事整理是"创作"：温度 0.3（主对话 0.7 与 evaluator
#   0.0 之间），保留记忆重构的人味，同时收敛波动幅度（014.5 三问结论）。
ARCHIVER_LLM_TEMPERATURE = getattr(config, "ARCHIVER_LLM_TEMPERATURE", 0.3)

# [017.5 T31] 包级双向量嵌入器：与 event_parser 同款 BGE（embedder 已带实例
#   缓存，全系统共用一个 1024 维模型，不重复加载）。
_pkg_embedder = get_embedder(getattr(config, "PACKAGE_EMBEDDER", "bge"))


class Archiver:
    """归档器：将短期缓冲区的记录整理成 MemoryPackage。

    M2-C：接入 RawLog，归档时把 flushed 记录同步写入 raw_log。
    organize_narrative 仍为桩，不接 memory_graph，不接 LLM。
    """

    def __init__(self, buffer: ShortTermBuffer, raw_log: RawLog, threshold: int = 6):
        self._buffer = buffer
        self._raw_log = raw_log
        self._threshold = threshold
        self._packages: List[MemoryPackage] = []

    def should_archive(self) -> bool:
        """判断是否该归档：缓冲区长度 >= 阈值时返回 True。"""
        return len(self._buffer) >= self._threshold

    def archive(self) -> Optional[MemoryPackage]:
        """执行归档流程。返回生成的 MemoryPackage，若无可归档内容返回 None。"""
        flushed, _ = self._buffer.flush_keep_recent(2)
        if not flushed:
            return None

        # 把 flushed 里每一轮写进 raw_log
        for record in flushed:
            self._raw_log.append(record)

        # [020.1] 归档降级：LLM 返回非 JSON / 空响应时，organize_narrative
        #   抛 ValueError（不静默：内部已打印响应诊断片段）；此处接住并
        #   跳过该包——本轮记录已写入 raw_log（数据不丢），记忆包不产出，
        #   对话链路继续（此前会直接抛 JSONDecodeError 打断整轮）。
        try:
            pkg = self.organize_narrative(flushed)
        except (ValueError, KeyError) as e:
            print(f"[archiver] 本轮归档跳过（{type(e).__name__}: {e}），"
                  f"记录已留在 raw_log，下一轮继续积累", file=sys.stderr)
            return None
        self._packages.append(pkg)
        return pkg

    def organize_narrative(self, records: List[Dict[str, Any]]) -> MemoryPackage:
        # 【桩】此处未来替换为 LLM 整理，当前为占位实现
        seq = len(self._packages) + 1

        # [017.5 T31] 包级双向量索引（mock/LLM 两条路径共用，向量与叙事来源无关）
        # [019] 第三返回值 mean_valence：包内各轮 S_valence 均值（整体极性）
        mean_vec, peak_vec, mean_valence = self._compute_pkg_vectors(records)

        if config.LLM_BACKEND == "mock":
            # 桩逻辑：固定拼接
            round_ids = [str(r.get("round_id", "")) for r in records]
            raw_pointer = "-".join(round_ids)

            # 关键词：取前 2 条记录 user_input 的前 4 个字
            keywords = []
            for r in records[:2]:
                text = r.get("user_input", "")
                if text:
                    keywords.append(text[:4])
            keywords = keywords[:3]

            # 叙事字段：桩占位
            first_user = records[0].get("user_input", "") if records else ""
            last_user = records[-1].get("user_input", "") if records else ""
            count = len(records)

            narrative_cause = f"桩：起因是用户在第{round_ids[0]}轮说'{first_user}'"
            narrative_trigger = f"桩：触发点是{count}轮对话积累"
            narrative_consequence = f"桩：后果是到第{round_ids[-1]}轮用户说'{last_user}'"
            narrative_evaluation = f"桩：我的评价是中性，共{count}轮"
            narrative_emotion = "桩：我的情绪是 neutral（占位）"
            narrative_thought = f"桩：我的想法是这{count}轮可以归纳为一段记忆"

            return MemoryPackage(
                seq=seq,
                emotion_value=0.5,
                emotion_polarity="neutral",
                keywords=keywords,
                raw_pointer=raw_pointer,
                mean_vec=mean_vec,
                peak_vec=peak_vec,
                mean_valence=mean_valence,
                narrative_cause=narrative_cause,
                narrative_trigger=narrative_trigger,
                narrative_consequence=narrative_consequence,
                narrative_evaluation=narrative_evaluation,
                narrative_emotion=narrative_emotion,
                narrative_thought=narrative_thought,
            )
        else:
            # 真 LLM 路径：调用 LLM 生成 JSON，解析为 MemoryPackage
            round_ids = [str(r.get("round_id", "")) for r in records]
            raw_pointer = "-".join(round_ids)

            # 构造对话原文
            dialog_lines = []
            for r in records:
                user_text = r.get("user_input", "")
                ai_text = r.get("ai_output", "")
                dialog_lines.append(f"用户：{user_text}")
                dialog_lines.append(f"AI：{ai_text}")
            dialog_text = "\n".join(dialog_lines)

            prompt = f"""请阅读以下对话片段，将其整理为一段记忆。
严格只输出 JSON，不要有任何额外文字、解释或 Markdown 代码块。

对话：
{dialog_text}

输出格式（严格遵循）：
{{
  "emotion_value": 0.0~1.0之间的浮点数,
  "emotion_polarity": "positive" / "negative" / "neutral",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "narrative_cause": "起因（一句话）",
  "narrative_trigger": "触发点（一句话）",
  "narrative_consequence": "后果（一句话）",
  "narrative_evaluation": "我的评价（一句话）",
  "narrative_emotion": "我的情绪（一句话）",
  "narrative_thought": "我的想法（一句话）"
}}

只输出 JSON 对象本身。"""

            llm_output = cached_generate(get_llm(), prompt,
                                         temperature=ARCHIVER_LLM_TEMPERATURE)

            # 解析 JSON
            import json
            raw = llm_output.strip()
            # 去掉可能的首尾 ```json ... ```
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # [020.1] 解析失败不吞（保留可观测性）但不静默：打印响应前
            #   120 字诊断片段（不含密钥）后抛 ValueError，由 archive()
            #   捕获并跳过该包（记录已在 raw_log，数据不丢，链路不炸）。
            try:
                data = json.loads(raw)
            except ValueError as e:
                _diag = raw[:120].replace("\n", " ")
                print(f"[archiver] 告警：归档 LLM 返回无法解析为 JSON"
                      f"（{type(e).__name__}: {e}）；响应前 120 字：{_diag!r}",
                      file=sys.stderr)
                raise

            return MemoryPackage(
                seq=seq,
                emotion_value=float(data.get("emotion_value", 0.0)),
                emotion_polarity=str(data.get("emotion_polarity", "neutral")),
                keywords=list(data.get("keywords", [])),
                raw_pointer=raw_pointer,
                mean_vec=mean_vec,
                peak_vec=peak_vec,
                mean_valence=mean_valence,
                narrative_cause=str(data.get("narrative_cause", "")),
                narrative_trigger=str(data.get("narrative_trigger", "")),
                narrative_consequence=str(data.get("narrative_consequence", "")),
                narrative_evaluation=str(data.get("narrative_evaluation", "")),
                narrative_emotion=str(data.get("narrative_emotion", "")),
                narrative_thought=str(data.get("narrative_thought", "")),
            )

    def get_packages(self) -> List[MemoryPackage]:
        """返回全部已归档的 MemoryPackage。"""
        return list(self._packages)

    # [017.5 T31] 包级双向量索引（017.txt 雷点1补丁(2)，与标定探针同口径）
    # [019] 扩展第三返回值 mean_valence：包内各轮 S_valence 均值（整体极性），
    #   供读取侧极性一致性过滤（018 议题二）。
    def _compute_pkg_vectors(self, records: List[Dict[str, Any]]):
        """归档时计算包级双向量 + 整体极性：mean_vec（主题均值）+ peak_vec
        （情绪最强轮）+ mean_valence（包内各轮 S_valence 均值）。

        口径（与 scripts/probe_t39_calibration.py 逐字对齐，第三项为 019 新增）：
          mean_vec = 包内各轮 user_input embedding 的均值（归一化）——保主题；
          peak_vec = 包内 intensity 最高轮的单轮 embedding——保强度。
            intensity 来源为 pipeline 写入缓冲记录的 record["intensity"]
            （= evaluator 提取的 S[3]）；旧记录无该字段时 intensity 记 0，
            若全包 intensity 均为 0（无强度信息），peak_vec 退化为 mean_vec。
          mean_valence = 包内各轮 record["valence"]（= 该轮 S[0]）的均值；
            旧记录无该字段不参与均值；全轮 valence 缺失 → None（读取侧
            对 None 包不过滤，无信息不惩罚）。
        降级（不崩）：编码异常或无有效文本 → 三项均返回 None，
          该包不参与读取侧匹配（读取侧对无嵌入包跳过）。
        """
        pairs = []
        for r in records:
            t = r.get("user_input", "")
            if t and t.strip():
                v_raw = r.get("valence", None)
                pairs.append((t,
                              float(r.get("intensity", 0.0) or 0.0),
                              None if v_raw is None else float(v_raw)))
        if not pairs:
            return None, None, None
        try:
            vecs = np.stack([_pkg_embedder.encode(t) for t, _, _ in pairs])
        except Exception as e:
            print(f"[archiver] 告警：包级嵌入失败（{type(e).__name__}: {e}），"
                  f"本包不写入双向量（读取侧将跳过该包）", file=sys.stderr)
            return None, None, None
        mean_vec = vecs.mean(axis=0)
        n = np.linalg.norm(mean_vec)
        if n > 1e-8:
            mean_vec = mean_vec / n
        # [019] 整体极性：全轮 valence 均值（旧轮无 valence 不参与）
        _known_val = [v for _, _, v in pairs if v is not None]
        mean_val = float(np.mean(_known_val)) if _known_val else None
        intensities = [v for _, v, _ in pairs]
        if max(intensities) <= 0.0:
            return mean_vec, mean_vec.copy(), mean_val  # 无强度信息：peak 退化为 mean
        peak_vec = vecs[int(np.argmax(intensities))]
        return mean_vec, peak_vec, mean_val
