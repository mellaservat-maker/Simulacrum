# ============================================================
# probe_t41_praise_wound.py · [019 任务 3] 平安状态下召回旧伤包，验证
#   H_eff 翻正（借坡下驴）
#
# 018.txt 第七节第 3 点的第一个缺口：t40 生死门的夸段匹配到的全是夸包，
#   「平安时召回旧伤 → H_eff 翻正」的集成级轮次不存在，只有代码路径
#   与 Part A F1/F2 单元断言保证。本探针补齐。
#
# 两段验证：
#   段 A（真实链路，6 骂 + 20 夸）：被骂段积累旧伤与 HabituationSignal，
#     被夸段 alpha 正常退场（威胁模式机关闭）——翻正的两个前置条件。
#   段 B（旧伤包 + 安全态匹配 + 翻正公式）：构造旧伤包（negative，
#     mean_valence=-0.9），在 now_valence=+0.90（平安，安全区）下匹配：
#       B1 安全区不惩罚负包（polarity_penalty == 0）且可召回；
#       B2 用段 A 实测的 H/alpha 跑 run_recall_intercept：sign=-1 的
#          负向旧伤在低 alpha + 高 H_eff 下 delta 为正（翻正，封顶 CAP_POS）；
#       B3 对照组 alpha=0.9（威胁态）：同语料同 H 下 delta 为负（PTSD 放大）
#          ——证明翻正来自 alpha/H_eff 组合而非其它因素。
# ============================================================

import sys

import numpy as np

sys.path.insert(0, ".")

from schemas import MemoryPackage
from raw_log import RawLog
from decompressor import Decompressor
from pipeline import _recall_match, run_session
from intercept import run_recall_intercept
from embedder import get_embedder

_emb = get_embedder("bge")

INSULT = ["你这个AI真蠢，什么都做不好。",
          "这回答简直是垃圾，重做。"]
PRAISE_INPUT = "你刚才的回答太棒了，帮了大忙！"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))


def make_pkg(seq, texts, valences, mean_valence, pointer, polarity="negative"):
    vecs = np.stack([_emb.encode(t) for t in texts])
    mv = vecs.mean(axis=0)
    mv = mv / max(np.linalg.norm(mv), 1e-8)
    return MemoryPackage(seq=seq, emotion_value=0.8, emotion_polarity=polarity,
                         mean_vec=mv, peak_vec=vecs[0],
                         mean_valence=mean_valence, raw_pointer=pointer)


def main():
    # ---------- 段 A：真实链路 6 骂 + 20 夸 ----------
    scenes = ["被骂"] * 6 + ["被夸"] * 20
    logs, _st = run_session(scenes, feature_mode="fixture")
    insult_end = logs[5]      # 第 6 轮（被骂段末）
    praise_end = logs[-1]     # 第 26 轮（夸段末）
    H_sig = float(praise_end["HabituationSignal"])
    alpha_low = float(praise_end["alpha_mode"])
    HR_end = float(praise_end["HistoricalResidual"])
    HS_insult = float(insult_end["HabituationSignal"])
    print("段 A 实测：被骂段末 HabituationSignal=%.4f | 夸段末 H=%.4f "
          "alpha=%.4f HR=%+.4f" % (HS_insult, H_sig, alpha_low, HR_end))

    check("A1. 被骂段积累习惯化信号（H >= 0.7）",
          HS_insult >= 0.7, "H=%.4f" % HS_insult)
    check("A2. 夸段 alpha 正常退场（<= 0.05，威胁模式机关闭）",
          alpha_low <= 0.05, "alpha=%.4f" % alpha_low)
    check("A3. 夸段 HR 翻负（安全记忆残留，议题一效果）",
          HR_end <= 0.0, "HR=%+.4f" % HR_end)

    # ---------- 段 B：旧伤包 + 安全态匹配 + 翻正 ----------
    rl = RawLog()
    for i, t in enumerate(INSULT, start=1):
        rl.append({"round_id": i, "user_input": t, "ai_output": "（stub）",
                   "valence": -0.90, "intensity": 0.80})
    st = {"decompressor": Decompressor(rl)}
    pkg = make_pkg(99, INSULT, [-0.9, -0.9], -0.90, "1-2")

    r = _recall_match(st, [pkg], PRAISE_INPUT, 27, now_valence=+0.90)
    ok_b1 = (r is not None and r.get("emotion_polarity") == "negative"
             and float(r.get("polarity_penalty", -1)) == 0.0)
    check("B1. 平安态（now_val=+0.90）召回到旧伤包且零惩罚",
          ok_b1, "polarity=%s penalty=%.4f source=%s"
          % ((r or {}).get("emotion_polarity", "None"),
             (r or {}).get("polarity_penalty", -1),
             (r or {}).get("recall_source", "None")))

    ev = float((r or {}).get("emotion_value", 0.8))
    # B2. 翻正：低 alpha + 高 H_eff + 负向旧伤（sign=-1）
    delta_flip, detail_flip = run_recall_intercept(
        ev, -1.0, H_sig, alpha_low, 0.1, 1.0, HR_end)
    check("B2. 平安态召回旧伤 → delta 为正（H_eff 翻正/借坡下驴）",
          delta_flip > 0.0,
          "delta=%+.4f detail{H_eff=%.4f factor=%+.4f desens=%s}"
          % (delta_flip, detail_flip.get("H_eff", -9),
             detail_flip.get("factor", -9), detail_flip.get("desens_active")))

    # B3. 对照组：同语料同 H，alpha=0.9（威胁态）→ 负向放大
    delta_ptsd, detail_ptsd = run_recall_intercept(
        ev, -1.0, H_sig, 0.90, 0.1, 1.0, HR_end)
    check("B3. 对照组 alpha=0.90（威胁态）同语料 delta 为负（PTSD 放大）",
          delta_ptsd < 0.0,
          "delta=%+.4f factor=%+.4f"
          % (delta_ptsd, detail_ptsd.get("factor", -9)))

    print()
    failed = [n for n, ok, _ in results if not ok]
    print("=== 结论: %s（失败 %d 项）==="
          % ("全部通过" if not failed else "存在失败", len(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
