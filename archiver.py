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
from llm_client import get_llm, cached_generate, LLMUnavailableError
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
        # [053] 归档失败统计（修复前失败即静默丢 4 轮，无任何计数）
        #   drop_events       归档失败事件数（每次 = 一批 flushed 未成包）
        #   retry_backfilled  其中已回灌缓冲区、等待下轮重试的次数
        #   giveup_batches    连续失败超过 ARCHIVE_MAX_RETRY 后放弃的批数
        #                     （原文仍在 raw_log，只是不再尝试成包）
        self._drop_events = 0
        self._retry_backfilled = 0
        self._giveup_batches = 0
        self._consecutive_fail = 0
        # [053] 去重合并次数（命中去重时包数不增，便于观测压缩效果）
        self._merged_into = 0
        # [054] 叙事降级次数：LLM 失败但包照常产出（数值保留、叙事留空）
        self._narrative_fallback = 0

    def should_archive(self) -> bool:
        """判断是否该归档：缓冲区长度 >= 阈值时返回 True。"""
        return len(self._buffer) >= self._threshold

    def archive(self) -> Optional[MemoryPackage]:
        """执行归档流程。返回生成的 MemoryPackage，若无可归档内容返回 None。

        [053] 失败不再静默丢数据：
          失败时把这批 flushed 记录按原顺序插回缓冲区队首（prepend_many），
          下一轮 should_archive() 仍为真，自然重试；同一批连续失败超过
          config.ARCHIVE_MAX_RETRY 次才放弃（原文已在 raw_log，不丢原文），
          并计入 stats() 的丢包统计。
          修复前：flush 已发生、包未产出、记录既不回 buffer 也不在任何包里，
          这 4 轮永久失去被归档的机会（只在 raw_log 里躺着）。
        """
        flushed, _ = self._buffer.flush_keep_recent(2)
        if not flushed:
            return None

        # 把 flushed 里每一轮写进 raw_log（幂等：同 round_id 覆盖）
        for record in flushed:
            self._raw_log.append(record)

        # [020.1] 归档降级：LLM 返回非 JSON / 空响应时，organize_narrative
        #   抛 ValueError（不静默：内部已打印响应诊断片段）。
        # [053] 追加 LLMUnavailableError：通道重试耗尽也不该把整轮打断。
        try:
            pkg = self.organize_narrative(flushed)
        except (ValueError, KeyError, LLMUnavailableError) as e:
            self._drop_events += 1
            self._consecutive_fail += 1
            if self._consecutive_fail <= config.ARCHIVE_MAX_RETRY:
                self._buffer.prepend_many(flushed)
                self._retry_backfilled += 1
                print(f"[archiver] 归档失败（{type(e).__name__}: {e}），"
                      f"已回灌 {len(flushed)} 轮到缓冲区队首等待重试"
                      f"（连续失败 {self._consecutive_fail}/"
                      f"{config.ARCHIVE_MAX_RETRY}，"
                      f"累计丢包事件 {self._drop_events} 次）",
                      file=sys.stderr)
            else:
                self._giveup_batches += 1
                print(f"[archiver] 归档连续失败已达上限 "
                      f"{config.ARCHIVE_MAX_RETRY}，放弃本批 {len(flushed)} 轮"
                      f"（原文仍在 raw_log，不再尝试成包）；"
                      f"累计丢包事件 {self._drop_events} 次、"
                      f"放弃 {self._giveup_batches} 批",
                      file=sys.stderr)
                # 放弃后重置计数：下一批重新获得完整重试额度，
                # 避免一次长故障把后续所有批次都变成"立即放弃"。
                self._consecutive_fail = 0
            return None
        self._consecutive_fail = 0
        # [053] 写入侧去重合并：同主题（mean_vec 余弦 > PACKAGE_MERGE_SIM）
        #   不新增包，直接并入旧包，返回旧包（调用方只关心"产出了一个包"）。
        dup, sim = self._find_duplicate(pkg)
        if dup is not None:
            self._merge_into(dup, pkg, sim)
            return dup
        self._packages.append(pkg)
        return pkg

    def stats(self) -> Dict[str, int]:
        """归档健康度（丢包可观测）。"""
        return {
            "drop_events": int(self._drop_events),
            "retry_backfilled": int(self._retry_backfilled),
            "giveup_batches": int(self._giveup_batches),
            "consecutive_fail": int(self._consecutive_fail),
            "merged_into": int(self._merged_into),
            "narrative_fallback": int(self._narrative_fallback),
            "packages": len(self._packages),
        }

    # ---- [053] 写入侧去重合并（修 B2：长会话包无合并、噪声累积）----

    @staticmethod
    def _parse_ids(raw_pointer: str):
        ids = []
        for p in (raw_pointer or "").split("-"):
            p = p.strip()
            if not p:
                continue
            try:
                ids.append(int(p))
            except ValueError:
                continue
        return ids

    @staticmethod
    def _cos(a, b) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _find_duplicate(self, pkg: MemoryPackage):
        """找与新包 mean_vec 余弦 > PACKAGE_MERGE_SIM 的旧包（取最相似）。"""
        if not getattr(config, "PACKAGE_MERGE_ENABLED", False):
            return None, -1.0
        if pkg.mean_vec is None or len(pkg.mean_vec) == 0:
            return None, -1.0
        best, best_sim = None, -1.0
        for old in self._packages:
            mv = getattr(old, "mean_vec", None)
            if mv is None or len(mv) != len(pkg.mean_vec):
                continue
            s = self._cos(pkg.mean_vec, mv)
            if s > best_sim:
                best, best_sim = old, s
        if best is None or best_sim < float(config.PACKAGE_MERGE_SIM):
            return None, best_sim
        return best, best_sim

    def _merge_into(self, old: MemoryPackage, new: MemoryPackage,
                    sim: float) -> None:
        """把 new 并入 old：拼 pointer、按轮数加权重算向量与统计。

        取舍说明：
          · mean_vec 按 n_rounds 加权后重新归一化（近似"全部轮次的真实均值"）；
          · peak_vec 取"情绪更强"的一侧（emotion_value 已是主导轮强度），
            与"取较大峰值"的要求一致；
          · mean_valence / mean_danger 按轮数加权，max_danger 取最大；
          · 叙事字段保留旧包（同一段经历，不重复调 LLM，也避免无限增长）。
        """
        n1 = max(1, int(old.n_rounds or 0))
        n2 = max(1, int(new.n_rounds or 0))
        ids = sorted(set(self._parse_ids(old.raw_pointer))
                     | set(self._parse_ids(new.raw_pointer)))
        if ids:
            old.raw_pointer = "-".join(str(i) for i in ids)

        mv = (np.asarray(old.mean_vec, dtype=float) * n1
              + np.asarray(new.mean_vec, dtype=float) * n2) / (n1 + n2)
        nrm = float(np.linalg.norm(mv))
        old.mean_vec = mv / nrm if nrm > 1e-8 else mv

        if float(new.emotion_value) > float(old.emotion_value):
            old.peak_vec = new.peak_vec
            old.emotion_value = float(new.emotion_value)
            old.emotion_polarity = str(new.emotion_polarity)

        kv = [(v, w) for v, w in ((old.mean_valence, n1),
                                  (new.mean_valence, n2)) if v is not None]
        if kv:
            old.mean_valence = float(sum(v * w for v, w in kv)
                                     / sum(w for _, w in kv))
        kd = [(v, w) for v, w in ((old.mean_danger, n1),
                                  (new.mean_danger, n2)) if v is not None]
        if kd:
            old.mean_danger = float(sum(v * w for v, w in kd)
                                    / sum(w for _, w in kd))
        maxes = [v for v in (old.max_danger, new.max_danger) if v is not None]
        old.max_danger = float(max(maxes)) if maxes else None

        old.n_rounds = int(old.n_rounds or 0) + int(new.n_rounds or 0)
        self._merged_into += 1
        print(f"[archiver] 去重合并：新包（{new.raw_pointer}）与 seq={old.seq}"
              f" 余弦 {sim:.3f} > {config.PACKAGE_MERGE_SIM}，并入旧包；"
              f"合并后 pointer={old.raw_pointer}，n_rounds={old.n_rounds}",
              file=sys.stderr)

    def organize_narrative(self, records: List[Dict[str, Any]]) -> MemoryPackage:
        # 【桩】此处未来替换为 LLM 整理，当前为占位实现
        # [053] seq 改为"现有最大 +1"：去重合并时包数不增，
        #   用 len+1 会产生重复 seq（matched_pkg_seq 观测会混淆）。
        seq = max([p.seq for p in self._packages], default=0) + 1

        # [017.5 T31] 包级双向量索引（mock/LLM 两条路径共用，向量与叙事来源无关）
        # [019] 第三返回值 mean_valence：包内各轮 S_valence 均值（整体极性）
        mean_vec, peak_vec, mean_valence = self._compute_pkg_vectors(records)
        # [029] 包级危险度聚合（写入侧原料；旧记录无 danger_score → None）
        n_rounds, mean_danger, max_danger = self._compute_pkg_danger(records)
        # [053] 包级情绪值：由主链路真实 S 聚合，两条路径共用
        #   （LLM 只负责叙事，不再编造数值——修 L23 / B4 口径不一致）
        emo_value, emo_polarity = self._compute_pkg_emotion(records)

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
                emotion_value=emo_value,
                emotion_polarity=emo_polarity,
                keywords=keywords,
                raw_pointer=raw_pointer,
                mean_vec=mean_vec,
                peak_vec=peak_vec,
                mean_valence=mean_valence,
                n_rounds=n_rounds,
                mean_danger=mean_danger,
                max_danger=max_danger,
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
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "narrative_cause": "起因（一句话）",
  "narrative_trigger": "触发点（一句话）",
  "narrative_consequence": "后果（一句话）",
  "narrative_evaluation": "我的评价（一句话）",
  "narrative_emotion": "我的情绪（一句话）",
  "narrative_thought": "我的想法（一句话）"
}}

只输出 JSON 对象本身。"""

            # [054] 叙事生成失败不再丢包。数值（S 聚合 / 双向量 / danger）
            #   全部与 LLM 无关，已经算好；叙事只是可选的一层描述。
            #   053 的做法是抛错 -> archive() 回灌 -> 连续 3 次放弃，
            #   那 4 轮就只能躺在 raw_log 里等死。现在改为：叙事留空照样成包。
            nar = {
                "keywords": [],
                "narrative_cause": "",
                "narrative_trigger": "",
                "narrative_consequence": "",
                "narrative_evaluation": "",
                "narrative_emotion": "",
                "narrative_thought": "",
            }
            try:
                llm_output = cached_generate(get_llm(), prompt,
                                             temperature=ARCHIVER_LLM_TEMPERATURE)
                nar.update(self._parse_narrative(llm_output))
            except (ValueError, KeyError, LLMUnavailableError) as e:
                self._narrative_fallback += 1
                nar["narrative_thought"] = (
                    "[054] 叙事缺失（LLM 失败：%s），本包仅保留数值与向量"
                    % type(e).__name__)
                print(f"[archiver] 叙事生成失败（{type(e).__name__}: {e}），"
                      f"本包按空叙事产出，数值与向量照常保留；"
                      f"累计叙事降级 {self._narrative_fallback} 次",
                      file=sys.stderr)

            return MemoryPackage(
                seq=seq,
                # [053] 数值来自主链路聚合，LLM 的返回值即使带了也不采用
                emotion_value=emo_value,
                emotion_polarity=emo_polarity,
                keywords=list(nar.get("keywords") or []),
                raw_pointer=raw_pointer,
                mean_vec=mean_vec,
                peak_vec=peak_vec,
                mean_valence=mean_valence,
                n_rounds=n_rounds,
                mean_danger=mean_danger,
                max_danger=max_danger,
                narrative_cause=str(nar.get("narrative_cause", "")),
                narrative_trigger=str(nar.get("narrative_trigger", "")),
                narrative_consequence=str(nar.get("narrative_consequence", "")),
                narrative_evaluation=str(nar.get("narrative_evaluation", "")),
                narrative_emotion=str(nar.get("narrative_emotion", "")),
                narrative_thought=str(nar.get("narrative_thought", "")),
            )

    @staticmethod
    def _parse_narrative(llm_output: str) -> Dict[str, Any]:
        """解析归档 LLM 的 JSON 输出；失败抛 ValueError（带诊断片段）。

        [054] 解析逻辑从 organize_narrative 抽出：失败时由调用方降级为
        空叙事成包，而不是把整批记录丢掉。
        """
        import json
        raw = (llm_output or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            data = json.loads(raw)
        except ValueError as e:
            _diag = raw[:120].replace("\n", " ")
            print(f"[archiver] 告警：归档 LLM 返回无法解析为 JSON"
                  f"（{type(e).__name__}: {e}）；响应前 120 字：{_diag!r}",
                  file=sys.stderr)
            raise
        if not isinstance(data, dict):
            raise ValueError("归档 LLM 返回的 JSON 不是对象：%r" % (type(data),))
        return data

    def get_packages(self) -> List[MemoryPackage]:
        """返回全部已归档的 MemoryPackage。"""
        return list(self._packages)

    # [050 A3] 公开写入接口：会话恢复与记忆植入都不再直接改私有列表。
    def append_package(self, pkg: MemoryPackage) -> MemoryPackage:
        """追加一个已构造好的记忆包（记忆植入 / 外部工具用）。"""
        self._packages.append(pkg)
        return pkg

    def replace_packages(self, pkgs) -> None:
        """整体替换记忆包列表（从存档恢复时用）。"""
        self._packages = [p for p in pkgs if p is not None]

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

    # [053] 包级情绪值：由主链路真实 S 聚合，取代"LLM 事后编造"。
    #   口径与读取侧轮级回捞一致：取包内 intensity 最高轮为主导轮，
    #   emotion_value = 该轮 intensity，emotion_polarity = 该轮 valence 符号。
    #   为什么不用均值：017.5 已实证整包均值会稀释情绪（3 夸 1 骂的包里骂轮
    #   被平均掉），读取侧因此改成轮级回捞；包级若仍取均值，旧路径
    #   （无向量的旧包 / RECALL_MATCH_ENABLED=False）会退回被稀释的口径。
    #   强度与方向必须来自同一轮，否则混合包会出现"高强度 + 正极性"的矛盾。
    #   缺信息：无 intensity（旧记录）→ 0.0/neutral；有 intensity 无 valence
    #   → 强度保留、极性 neutral（无信息不编造）。
    def _compute_pkg_emotion(self, records: List[Dict[str, Any]]):
        best_val, best_int = None, -1.0
        for r in records:
            try:
                it = float(r.get("intensity", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if it > best_int:
                best_int = it
                v = r.get("valence", None)
                best_val = None if v is None else float(v)
        if best_int <= 0.0:
            return 0.0, "neutral"
        if best_val is None:
            return float(best_int), "neutral"
        pol = ("positive" if best_val > 0.0
               else ("negative" if best_val < 0.0 else "neutral"))
        return float(best_int), pol

    # [029 任务3] 包级危险度聚合：写入侧原料。
    #   口径：对包内各轮 record["danger_score"] 求均值与最大值；
    #   旧记录无该字段 → 不参与均值；全包缺失 → (n_rounds, None, None)。
    #   n_rounds 始终为包内记录总数，供后续 HR 反推与去重时判断包规模。
    def _compute_pkg_danger(self, records: List[Dict[str, Any]]):
        vals = []
        for r in records:
            d = r.get("danger_score", None)
            if d is None:
                continue
            try:
                vals.append(float(d))
            except (TypeError, ValueError):
                continue
        n_rounds = len(records)
        if not vals:
            return n_rounds, None, None
        return n_rounds, float(np.mean(vals)), float(np.max(vals))
