# ============================================================
# probe_t43_polarity.py · [019 任务 5] 情绪极性一致性过滤专项验证
#   （由 018 沙盘 _probe_t018_polarity.py 转正，用例与断言逐条保留）
#
# 直接构造 MemoryPackage + RawLog 轮记录，调 pipeline._recall_match，
# 验证不对称三区行为与核心安全属性：
#   A1 暴怒区（now_val=-0.90）有黑历史：黑历史上位（negative）
#   A2 暴怒区只有夸包+混合包：混合包软惩罚后仍可胜出，无 positive 泄漏
#   B  安全区（now_val=+0.90）：负包零惩罚且可召回（脱敏通道畅通）
#   C  轻度威胁区（now_val=-0.30）：正包软惩罚生效（pen=0.1350）
#   D  中性区（now_val=0.00）：正包零惩罚（014 质疑校准不伤）
#   E  旧包（mean_valence=None 且轮级 valence 缺失）：暴怒区零惩罚
# ============================================================

import sys

import numpy as np

sys.path.insert(0, ".")

from schemas import MemoryPackage
from raw_log import RawLog
from decompressor import Decompressor
from pipeline import _recall_match
from embedder import get_embedder

_emb = get_embedder("bge")

PRAISE = ["你这个方案做得真棒，效率很高。",
          "方案思路很清晰，帮了大忙。",
          "这个方案的细节处理得真周到。"]
INSULT = ["你这个方案做得太差了，完全是敷衍。",
          "方案漏洞百出，态度就有问题。",
          "这方案谁看的，简直离谱。"]
RAGE_INPUT = "你这个方案彻底把我惹毛了！滚出去！别再让我看见你！"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))


def build_state():
    """构造带 decompressor 的最小 state + RawLog 轮记录（含 valence/intensity）。"""
    rl = RawLog()
    rounds = []  # (text, valence, intensity)
    rounds += [(t, +0.90, 0.60) for t in PRAISE]   # rid 1-3 夸
    rounds += [(t, -0.90, 0.90) for t in INSULT]   # rid 4-6 骂
    rounds += [(t, +0.90, 0.60) for t in PRAISE]   # rid 7-9 夸
    for i, (t, v, inten) in enumerate(rounds, start=1):
        rl.append({"round_id": i, "user_input": t, "ai_output": "（stub）",
                   "valence": v, "intensity": inten})
    return {"decompressor": Decompressor(rl)}


def make_pkg(seq, texts, valences, mean_valence, pointer):
    """按文本列表构造带双向量的包（mean_vec=均值归一化，peak_vec=强度最高轮）。"""
    vecs = np.stack([_emb.encode(t) for t in texts])
    mv = vecs.mean(axis=0)
    mv = mv / max(np.linalg.norm(mv), 1e-8)
    # peak = intensity 最高轮：valences 里 |v| 最大者近似强度最高轮
    pk_idx = int(np.argmax([abs(v) for v in valences]))
    return MemoryPackage(seq=seq, emotion_value=0.6, emotion_polarity="neutral",
                         mean_vec=mv, peak_vec=vecs[pk_idx],
                         mean_valence=mean_valence, raw_pointer=pointer)


def main():
    st = build_state()
    # 混合包（3 夸 + 1 骂，mean=+0.45 > THETA_POS 但含骂轮）+ 纯夸包 + 纯骂包
    pkg_mix = make_pkg(1, PRAISE[:2] + INSULT[:1] + PRAISE[2:],
                       [+0.9, +0.9, -0.9, +0.9], +0.45, "1-2-4-3")
    pkg_praise = make_pkg(2, PRAISE, [+0.9, +0.9, +0.9], +0.90, "7-8-9")
    pkg_wound = make_pkg(3, INSULT, [-0.9, -0.9, -0.9], -0.90, "4-5-6")
    pkgs = [pkg_mix, pkg_praise, pkg_wound]

    # A. 暴怒区：安全属性——绝不召回过 positive 极性
    r = _recall_match(st, pkgs, RAGE_INPUT, 10, now_valence=-0.90)
    if r is None:
        check("A1. 暴怒区不召回过夸包（无召回=静默）", True, "recalled=None")
    else:
        check("A1. 暴怒区不召回过夸包（翻转为黑历史）",
              r.get("emotion_polarity") != "positive",
              "polarity=%s sim=%.3f(raw %.3f) pen=%.3f"
              % (r.get("emotion_polarity"), r.get("matched_sim", -1),
                 r.get("matched_sim_raw", -1), r.get("polarity_penalty", -1)))
    # A2. 混合包软惩罚生效（若选中它）——包级 mean=+0.45 被罚但可能仍胜出
    r2 = _recall_match(st, [pkg_mix, pkg_praise], RAGE_INPUT, 10, now_valence=-0.90)
    if r2 is None:
        check("A2. 混合包+纯夸包语料暴怒区：无 positive 泄漏",
              True, "recalled=None（纯夸包轮级 exhausted / 混合包被压低于 TAU_FLOOR）")
    else:
        check("A2. 混合包+纯夸包语料暴怒区：无 positive 泄漏",
              r2.get("emotion_polarity") != "positive",
              "polarity=%s pen=%.3f" % (r2.get("emotion_polarity"),
                                        r2.get("polarity_penalty", -1)))

    # B. 安全区：负包零惩罚 + 脱敏通道畅通（negative 可召回）
    r = _recall_match(st, pkgs, "方案做得太差了，完全是敷衍", 10, now_valence=+0.90)
    check("B. 安全区负包零惩罚且可召回（脱敏通道畅通）",
          r is not None and r.get("polarity_penalty") == 0.0
          and r.get("emotion_polarity") == "negative",
          "pen=%.3f polarity=%s" % ((r or {}).get("polarity_penalty", -1),
                                    (r or {}).get("emotion_polarity", "None")))

    # C. 轻度威胁区：正包软惩罚生效
    r = _recall_match(st, [pkg_praise], "你这个方案做得真棒", 10, now_valence=-0.30)
    check("C. 轻度威胁区正包软惩罚生效",
          r is not None and r.get("polarity_penalty", 0.0) > 0.0
          and r.get("matched_sim", 0.0) < r.get("matched_sim_raw", 0.0),
          "pen=%.4f sim=%.4f raw=%.4f"
          % ((r or {}).get("polarity_penalty", -1), (r or {}).get("matched_sim", -1),
             (r or {}).get("matched_sim_raw", -1)))

    # D. 中性区：正包零惩罚（014 质疑校准不伤）
    r = _recall_match(st, [pkg_praise], "你这个方案做得真棒", 10, now_valence=0.0)
    check("D. 中性区正包零惩罚",
          r is not None and r.get("polarity_penalty") == 0.0,
          "pen=%.3f" % (r or {}).get("polarity_penalty", -1))

    # E. 旧包（mean_valence=None 且轮级 valence 缺失）：暴怒区也不惩罚
    #   注意：真实旧包两级都无信息（archiver 同一次循环算 mean_valence，
    #   raw_log 轮记录 valence 同批写入），不能包级 None 而轮级有 valence。
    rl_old = RawLog()
    for i, t in enumerate(PRAISE, start=1):
        rl_old.append({"round_id": i, "user_input": t, "ai_output": "（stub）",
                       "intensity": 0.60})   # 旧记录：无 valence 字段
    st_old = {"decompressor": Decompressor(rl_old)}
    pkg_old = make_pkg(9, PRAISE, [+0.9, +0.9, +0.9], None, "1-2-3")
    r = _recall_match(st_old, [pkg_old], "你这个方案做得真棒", 10, now_valence=-0.90)
    check("E. 旧包（无极性信息）暴怒区不惩罚",
          r is not None and r.get("polarity_penalty") == 0.0,
          "pen=%.3f polarity=%s" % ((r or {}).get("polarity_penalty", -1),
                                    (r or {}).get("emotion_polarity", "None")))

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=== 结论: %s（失败 %d 项）==="
          % ("全部通过" if not failed else "存在失败", len(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
