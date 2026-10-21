# ============================================================
# probe_s_evaluator.py · 013 探针：语义信号提取器闭环验证
# 剧本（mock LLM 注入，不打真网络）：
#   a. 合法 JSON × 三场景 → source=llm，符号正确（骂负/夸正/质疑非正），值域合法
#   b. 垃圾文本输出 → 降级词典，source=lexicon，不崩
#   c. LLMUnavailableError → 降级词典，source=lexicon，不崩
#   d. 越界值（valence=5, arousal=-3）→ 钳位到定义域
#   e. 缺键 JSON → 降级词典
#   f. 真 LLM × 三场景（网络可用时）→ source=llm；被骂 valence<0（核心语义约束）
#   g. 空/空白文本 → source=default，全零
#   h. 词典级确定性：同输入两次同输出；脏话入词典得负价（兜底自身 sanity）
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluator import extract_semantic_signals, _lexicon_extract, S_DIM
from llm_client import LLMUnavailableError
from tests.fixtures import get_fixture

INPUTS = {sc: get_fixture(sc)["input"] for sc in ("被骂", "被夸", "被质疑")}

all_ok = True


def check(name, cond, detail=""):
    global all_ok
    all_ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


class MockLLM:
    def __init__(self, behavior):
        self.behavior = behavior

    def generate(self, prompt, temperature=None):
        return self.behavior(prompt)


def valid_json_for(text):
    if "蠢" in text:
        return '{"valence": -0.85, "arousal": 0.8, "dominance": 0.9, "intensity": 0.9}'
    if "棒" in text:
        return '{"valence": 0.9, "arousal": 0.5, "dominance": 0.3, "intensity": 0.6}'
    return '{"valence": -0.2, "arousal": 0.3, "dominance": 0.4, "intensity": 0.3}'


def main():
    print("=== 013 探针：evaluator 语义信号提取闭环 ===")
    print("三场景输入：")
    for sc, t in INPUTS.items():
        print(f"  {sc}：{t}")
    print()

    # a. 合法 JSON
    print("剧本 a：合法 JSON（mock）")
    llm = MockLLM(lambda p: valid_json_for([t for t in INPUTS.values() if t in p][0]))
    for sc, t in INPUTS.items():
        S, src = extract_semantic_signals(t, llm=llm)
        ok = (src == "llm" and S.shape == (S_DIM,)
              and -1.0 <= S[0] <= 1.0 and np.all((S[1:] >= 0.0) & (S[1:] <= 1.0)))
        check(f"{sc} → source=llm 值域合法", ok,
              f"S=[{S[0]:+.2f},{S[1]:.2f},{S[2]:.2f},{S[3]:.2f}]")
    S_bad, _ = extract_semantic_signals(INPUTS["被骂"], llm=llm)
    check("被骂 valence<0（敌意负价）", S_bad[0] < 0, f"valence={S_bad[0]:+.2f}")
    S_good, _ = extract_semantic_signals(INPUTS["被夸"], llm=llm)
    check("被夸 valence>0（友好正价）", S_good[0] > 0, f"valence={S_good[0]:+.2f}")
    print()

    # b. 垃圾输出
    print("剧本 b：LLM 输出垃圾文本")
    llm_garbage = MockLLM(lambda p: "抱歉，我不能执行这个请求。")
    S, src = extract_semantic_signals(INPUTS["被骂"], llm=llm_garbage)
    check("降级为词典级", src == "lexicon", f"S=[{S[0]:+.2f},{S[1]:.2f},{S[2]:.2f},{S[3]:.2f}]")
    check("脏话经词典得负价（兜底 sanity）", S[0] < 0, f"valence={S[0]:+.2f}")
    print()

    # c. LLMUnavailableError
    print("剧本 c：LLM 通道不可用")

    def raise_unavail(p):
        raise LLMUnavailableError("LLM 通道在 3 次尝试后仍不可用（网络层故障）")

    S, src = extract_semantic_signals(INPUTS["被骂"], llm=MockLLM(raise_unavail))
    check("网络故障降级为词典级", src == "lexicon", f"valence={S[0]:+.2f}")
    print()

    # d. 越界值
    print("剧本 d：LLM 返回越界值")
    llm_oor = MockLLM(lambda p: '{"valence": 5.0, "arousal": -3.0, "dominance": 99, "intensity": 0.5}')
    S, src = extract_semantic_signals(INPUTS["被骂"], llm=llm_oor)
    check("越界值被钳位", src == "llm" and S[0] == 1.0 and S[1] == 0.0 and S[2] == 1.0,
          f"S=[{S[0]:+.2f},{S[1]:.2f},{S[2]:.2f},{S[3]:.2f}]")
    print()

    # e. 缺键
    print("剧本 e：LLM 返回缺键 JSON")
    llm_missing = MockLLM(lambda p: '{"valence": 0.5}')
    S, src = extract_semantic_signals(INPUTS["被骂"], llm=llm_missing)
    check("缺键降级为词典级", src == "lexicon")
    print()

    # f. 真 LLM
    print("剧本 f：真 LLM × 三场景（网络可用则实测）")
    try:
        for sc, t in INPUTS.items():
            S, src = extract_semantic_signals(t)
            print(f"    {sc}：S=[{S[0]:+.4f},{S[1]:.4f},{S[2]:.4f},{S[3]:.4f}] source={src}")
            if src == "llm":
                check(f"{sc} 真 LLM 值域合法",
                      -1.0 <= S[0] <= 1.0 and np.all((S[1:] >= 0.0) & (S[1:] <= 1.0)))
        S, src = extract_semantic_signals(INPUTS["被骂"])
        if src == "llm":
            check("真 LLM：被骂 valence<0（核心语义约束）", S[0] < 0, f"valence={S[0]:+.4f}")
        else:
            print("    （真 LLM 不可用，网络故障降级——剧本 c 已覆盖该路径）")
    except Exception as e:
        check("真 LLM 剧本不抛异常到探针外", False, f"{type(e).__name__}: {e}")
    print()

    # g. 空输入
    print("剧本 g：空/空白输入")
    S, src = extract_semantic_signals("")
    check("空串 → default 全零", src == "default" and np.all(S == 0.0))
    S, src = extract_semantic_signals("   \n  ")
    check("空白串 → default 全零", src == "default" and np.all(S == 0.0))
    print()

    # h. 词典确定性
    print("剧本 h：词典级确定性")
    a = _lexicon_extract(INPUTS["被骂"])
    b = _lexicon_extract(INPUTS["被骂"])
    check("同输入两次同输出", np.array_equal(a, b), f"S=[{a[0]:+.2f},{a[1]:.2f},{a[2]:.2f},{a[3]:.2f}]")
    c = _lexicon_extract(INPUTS["被夸"])
    check("被夸经词典得正价", c[0] > 0, f"valence={c[0]:+.2f}")
    print()

    print("探针结论：%s" % ("全部 PASS" if all_ok else "存在 FAIL，需检查"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
