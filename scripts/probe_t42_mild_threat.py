# ============================================================
# probe_t42_mild_threat.py · [019 任务 3] 轻度威胁区软惩罚生效验证
#
# 018.txt 第七节第 3 点的第二个缺口：轻度威胁区（-THETA_RAGE <= now_val
#   < -THETA_MILD，即 [-0.60, -0.15)）的软惩罚在 t40 生死门没有集成轮次
#   （质疑场景 valence≈0 走中性区）。本探针补齐。
#
# 验证项：
#   D1 包级软惩罚精确数值：正包 now_val=-0.30 → pen = LAMBDA*0.9*0.30
#      = 0.135，sim_eff = sim*(1-0.135)；
#   D2 阈值边界：now_val=-0.15 恰好不惩罚（严格小于才触发）、-0.16 触发；
#   D3 轮级软惩罚改变排序：混合包（夸轮 sim 高于骂轮），now_val=-0.55 时
#      开启过滤 top1 翻转为骂轮（negative），关闭过滤 top1 为夸轮（positive）
#      ——证明轮级软惩罚真实参与 top-k 排序；
#   D4 中性区（now_val=0.0）零惩罚（014 质疑校准不伤）。
# ============================================================

import sys

import numpy as np

sys.path.insert(0, ".")

import config
from schemas import MemoryPackage
from raw_log import RawLog
from decompressor import Decompressor
from pipeline import _recall_match
from embedder import get_embedder

_emb = get_embedder("bge")

PRAISE = ["你这个回答非常棒，我很满意。",
          "这个回答真不错，帮了大忙。"]
# D3 用轮：与输入文本相似度 0.8124（embedder 实测），但 evaluator 重提
#   valence=-0.60（语义负）——保证「开过滤后夸轮被打折到它下面」可观测。
INSULT_BY_MILD = "你这个回答非常棒，我一点都不满意。"
MILD_INPUT = "你这个回答非常棒，我很满意。"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))


def make_pkg(seq, texts, valences, mean_valence, pointer, polarity="positive"):
    vecs = np.stack([_emb.encode(t) for t in texts])
    mv = vecs.mean(axis=0)
    mv = mv / max(np.linalg.norm(mv), 1e-8)
    return MemoryPackage(seq=seq, emotion_value=0.8, emotion_polarity=polarity,
                         mean_vec=mv, peak_vec=vecs[0],
                         mean_valence=mean_valence, raw_pointer=pointer)


def main():
    rl = RawLog()
    rl.append({"round_id": 1, "user_input": PRAISE[0], "ai_output": "（stub）",
               "valence": +0.90, "intensity": 0.60})
    rl.append({"round_id": 2, "user_input": PRAISE[1], "ai_output": "（stub）",
               "valence": +0.90, "intensity": 0.60})
    st = {"decompressor": Decompressor(rl)}
    pkg_pos = make_pkg(1, PRAISE, [+0.9, +0.9], +0.90, "1-2")

    # ---------- D1：包级软惩罚精确数值 ----------
    r = _recall_match(st, [pkg_pos], MILD_INPUT, 10, now_valence=-0.30)
    pen = float((r or {}).get("polarity_penalty", -1))
    sim_eff = float((r or {}).get("matched_sim", -1))
    sim_raw = float((r or {}).get("matched_sim_raw", -1))
    expect_pen = config.LAMBDA_POL * 0.90 * 0.30
    check("D1. 包级软惩罚精确（pen=%.4f，sim_eff/sim_raw=%.4f）"
          % (expect_pen, 1.0 - expect_pen),
          abs(pen - expect_pen) < 1e-6 and abs(sim_eff / max(sim_raw, 1e-9)
                                              - (1.0 - expect_pen)) < 1e-6,
          "pen=%.6f sim_eff=%.4f raw=%.4f" % (pen, sim_eff, sim_raw))

    # ---------- D2：阈值边界 ----------
    r_edge = _recall_match(st, [pkg_pos], MILD_INPUT, 10, now_valence=-0.15)
    pen_edge = float((r_edge or {}).get("polarity_penalty", -1))
    r_just = _recall_match(st, [pkg_pos], MILD_INPUT, 10, now_valence=-0.16)
    pen_just = float((r_just or {}).get("polarity_penalty", -1))
    expect_just = config.LAMBDA_POL * 0.90 * 0.16
    check("D2. 阈值边界：-0.15 不惩罚、-0.16 触发（严格小于）",
          pen_edge == 0.0 and abs(pen_just - expect_just) < 1e-6,
          "-0.15→pen=%.4f  -0.16→pen=%.4f(expect %.4f)"
          % (pen_edge, pen_just, expect_just))

    # ---------- D3：轮级软惩罚改变 top-k 排序 ----------
    # 构造混合包：轮 1 夸（与输入同文，sim=1.0，raw valence +0.9）、
    # 轮 2 语义负但高相似（sim=0.8124，raw valence -0.9）。
    # now_val=-0.55 → 夸轮软惩罚 pen=0.2475 → eff=0.7525 < 0.8124 → 骂轮升到 top-1。
    rl_mix = RawLog()
    rl_mix.append({"round_id": 1, "user_input": MILD_INPUT, "ai_output": "（stub）",
                   "valence": +0.90, "intensity": 0.60})
    rl_mix.append({"round_id": 2, "user_input": INSULT_BY_MILD, "ai_output": "（stub）",
                   "valence": -0.90, "intensity": 0.60})
    pkg_mix = make_pkg(2, [MILD_INPUT, INSULT_BY_MILD], [+0.9, -0.9],
                       +0.00, "1-2")
    dec = Decompressor(rl_mix)
    st_mix = {"decompressor": dec}
    now_vec = _emb.encode(MILD_INPUT)
    on = dec.for_emotion_evaluator_matched(pkg_mix, now_vec, top_k=2,
                                           now_valence=-0.55)
    _saved = config.POLARITY_FILTER_ENABLED
    config.POLARITY_FILTER_ENABLED = False
    try:
        off = dec.for_emotion_evaluator_matched(pkg_mix, now_vec, top_k=2,
                                                now_valence=-0.55)
    finally:
        config.POLARITY_FILTER_ENABLED = _saved
    rr_on = (on.get("recalled_rounds") or [])
    rr_off = (off.get("recalled_rounds") or [])
    top1_on = rr_on[0].get("valence", 0) if rr_on else None
    top1_off = rr_off[0].get("valence", 0) if rr_off else None
    # 语料鲁棒性：构造包本身也走一遍 _recall_match（集成入口）确认可召回
    r_mix = _recall_match(st_mix, [pkg_mix], MILD_INPUT, 10, now_valence=-0.55)
    check("D3. 轮级软惩罚改变 top-1：关闭→夸轮在前，开启→骂轮在前",
          top1_off is not None and top1_on is not None
          and float(top1_off) > 0.0 and float(top1_on) < 0.0,
          "off_top1_valence=%s on_top1_valence=%s 召回入口=%s"
          % (None if top1_off is None else round(float(top1_off), 2),
             None if top1_on is None else round(float(top1_on), 2),
             (r_mix or {}).get("emotion_polarity", "None")))

    # ---------- D4：中性区零惩罚 ----------
    r_neu = _recall_match(st, [pkg_pos], MILD_INPUT, 10, now_valence=0.0)
    pen_neu = float((r_neu or {}).get("polarity_penalty", -1))
    check("D4. 中性区（now_val=0.0）零惩罚（014 质疑校准不伤）",
          pen_neu == 0.0, "pen=%.4f" % pen_neu)

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=== 结论: %s（失败 %d 项）==="
          % ("全部通过" if not failed else "存在失败", len(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
