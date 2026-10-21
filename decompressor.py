# ============================================================
# decompressor.py · M2-C · 解压器
# 从 MemoryPackage 按不同器官的需求取出对应信息
# [017.5 T31] 新增轮级回捞：命中包后取与当前输入余弦最高的 top2 轮，
#   用 evaluator（温度 0.0 已锁）现场重提 S——注入内容轮级化，
#   告别整包 emotion_value 的均值稀释（017.txt 雷点1补丁(1)）。
# ============================================================

import sys
from typing import List, Dict, Any, Optional

import numpy as np

from schemas import MemoryPackage
from raw_log import RawLog
from embedder import get_embedder
from evaluator import extract_semantic_signals
import config

# [017.5 T31] 轮级回捞用嵌入器：与 event_parser/archiver 同款 BGE（实例缓存共享）
_emb = get_embedder("bge")


def _cos(a, b) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Decompressor:
    """解压器：从 MemoryPackage 按器官需求取出信息。

    字段来源：
      - MemoryPackage 直接字段（seq, emotion_value, narrative_* 等）
      - raw_pointer 解析出的轮次 → 去 RawLog 取原文
    """

    def __init__(self, raw_log: RawLog):
        self._raw_log = raw_log

    # ---- 解析 raw_pointer ----

    def _parse_raw_pointer(self, raw_pointer: str) -> List[int]:
        """解析 raw_pointer（"1-2-3" 格式），返回 round_id 列表。
        解析失败时返回空列表。"""
        if not raw_pointer:
            return []
        parts = raw_pointer.split("-")
        ids = []
        for p in parts:
            p = p.strip()
            if not p:
                return []
            try:
                ids.append(int(p))
            except ValueError:
                return []
        return ids

    def _get_raw(self, pkg: MemoryPackage) -> List[Dict[str, Any]]:
        ids = self._parse_raw_pointer(pkg.raw_pointer)
        if not ids:
            return []
        results = self._raw_log.get_many(ids)
        # 过滤掉 None（缺失的轮次），但保持顺序
        return [r for r in results if r is not None]

    # ---- 器官专用方法 ----

    def for_emotion_evaluator(self, pkg: MemoryPackage) -> Dict[str, Any]:
        return {
            "emotion_value": pkg.emotion_value,
            "emotion_polarity": pkg.emotion_polarity,
        }

    # ---- [017.5 T31] 轮级回捞（方案 C 修订版）----

    def package_max_round(self, pkg: MemoryPackage) -> Optional[int]:
        """包内最大 round_id（时间窗判定用）；解析失败返回 None。"""
        ids = self._parse_raw_pointer(pkg.raw_pointer)
        return max(ids) if ids else None

    def for_emotion_evaluator_matched(self, pkg: MemoryPackage,
                                      now_vec: Optional[np.ndarray] = None,
                                      top_k: int = 2,
                                      now_valence: Optional[float] = None) -> Dict[str, Any]:
        """命中包的轮级新鲜语义（017.txt 4.1 检索层落地）。

        链路：raw_pointer 回捞原文 → 各轮 user_input 现编码 → 取与 now_vec
        余弦最高的 top_k 轮 → evaluator 对每轮现场重提 S → 取强度最高轮
        为主导（ev=intensity，sign=valence 符号）。
        [019] 轮级极性一致性过滤（与 pipeline._recall_match 包级软惩罚
        同参同构的不对称设计，018 沙盘实证后落地）：包级软惩罚挡不住混合包
        内轮的极性混杂——4 骂 3 夸的包 mean_valence 为负能过包级闸，其
        top2 轮却可能全是夸轮（主题语义更近），dominant 轮为夸 → 符号
        背离依旧。故轮级 top-k 排序同样应用：
          暴怒区 now_val < -THETA_RAGE：valence > THETA_POS 的轮硬过滤；
          轻度威胁区：positive 轮软惩罚 sim_eff = sim x (1 - LAMBDA*pv*(-now_val))；
          安全区/中性区：不惩罚（脱敏通道与 014 质疑校准不受影响）。
        轮级 valence 来源：raw_log 记录的 record["valence"]（017.5 起写入，
        旧记录无该字段 → 该轮不过滤，无信息不惩罚）。
        包内轮全被极性过滤 → 返回 recall_source="polarity_exhausted"
        （上层 pipeline 转为无召回，静默好过召回夸包）。
        降级（不崩、不静默）：
          now_vec 为空或原文缺失/编码失败 → 退回整包 emotion_value/polarity，
          recall_source 标记 "package_level"（与旧行为等价）。
        """
        base = {
            "emotion_value": pkg.emotion_value,
            "emotion_polarity": pkg.emotion_polarity,
            "recall_source": "package_level",
            "recalled_rounds": [],
        }
        if now_vec is None:
            return base
        raws = self._get_raw(pkg)
        if not raws:
            return base
        try:
            round_vecs = [_emb.encode(r.get("user_input", "")) for r in raws]
        except Exception as e:
            print(f"[decompressor] 告警：轮级回捞嵌入失败（{type(e).__name__}: {e}），"
                  f"退回整包情绪值", file=sys.stderr)
            return base
        sims = [_cos(now_vec, v) for v in round_vecs]
        # [019] 轮级极性一致性过滤（不对称，参数容错读取）
        _pol_on = getattr(config, "POLARITY_FILTER_ENABLED", False)
        _th_rage = getattr(config, "THETA_RAGE", 0.60)
        _th_pos = getattr(config, "THETA_POS", 0.30)
        _th_mild = getattr(config, "THETA_MILD", 0.15)
        _lam = getattr(config, "LAMBDA_POL", 0.50)
        eff = []
        for i, s in enumerate(sims):
            v_raw = raws[i].get("valence", None)
            rv = None if v_raw is None else float(v_raw)
            s_eff = s
            if _pol_on and now_valence is not None and rv is not None:
                nv = float(now_valence)
                if nv < -_th_rage and rv > _th_pos:
                    continue   # 暴怒区硬过滤 positive 轮
                elif -_th_rage <= nv < -_th_mild and rv > 0.0:
                    s_eff = s * (1.0 - min(1.0, _lam * rv * (-nv)))
            eff.append((i, s_eff))
        if not eff:
            # 包内轮全被极性过滤（纯 positive 包在暴怒区无黑历史可捞）：
            # 返回 exhausted 标记，pipeline 侧转为无召回（静默好过召回夸包）。
            # 不返回 base：base 的 package_level 极性可能是 positive，会漏闸。
            return {
                "emotion_value": 0.0,
                "emotion_polarity": "neutral",
                "recall_source": "polarity_exhausted",
                "recalled_rounds": [],
            }
        order = [i for i, _ in sorted(eff, key=lambda t: t[1], reverse=True)[:top_k]]

        # evaluator 现场重提 S（温度 0.0，同输入缓存；失败自动降词典级，不静默）
        extracted = []
        for i in order:
            text = raws[i].get("user_input", "")
            if not text or not text.strip():
                continue
            S, src = extract_semantic_signals(text)
            extracted.append({
                "round_id": raws[i].get("round_id"),
                "valence": float(S[0]),
                "intensity": float(S[3]),
                "S_source": src,
            })
        if not extracted:
            return base
        dominant = max(extracted, key=lambda d: d["intensity"])
        sign = 1.0 if dominant["valence"] > 0 else (-1.0 if dominant["valence"] < 0 else 0.0)
        polarity = "positive" if sign > 0 else ("negative" if sign < 0 else "neutral")
        return {
            "emotion_value": float(dominant["intensity"]),
            "emotion_polarity": polarity,
            "recall_source": "round_level",
            "recalled_rounds": extracted,
        }

    def for_danger_evaluator(self, pkg: MemoryPackage) -> Dict[str, Any]:
        return {
            "consequence": pkg.narrative_consequence,
            "evaluation": pkg.narrative_evaluation,
            "raw": self._get_raw(pkg),
        }

    def for_policy(self, pkg: MemoryPackage) -> Dict[str, Any]:
        return {
            "consequence": pkg.narrative_consequence,
            "thought": pkg.narrative_thought,
            "raw": self._get_raw(pkg),
        }

    def for_xinguang(self, pkg: MemoryPackage) -> Dict[str, Any]:
        return {
            "keywords": pkg.keywords,
            "emotion_value": pkg.emotion_value,
        }

    # ---- 通用方法 ----

    def get_fields(self, pkg: MemoryPackage, fields: List[str]) -> Dict[str, Any]:
        """按字段名列表取，返回 dict。

        特殊字段 "raw"：返回该压缩包对应的全量原文列表。
        其他字段：直接从 MemoryPackage 上取属性，不存在则返回 None。
        """
        result = {}
        for f in fields:
            if f == "raw":
                result[f] = self._get_raw(pkg)
            else:
                result[f] = getattr(pkg, f, None)
        return result
