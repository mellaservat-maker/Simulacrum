# ============================================================
# probe_real_prompt.py · 真实场景提示词探针
# 真跑三个场景，用真实算出的值调 state_to_prompt
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import run_scenario
from state_translator import state_to_prompt


def check(scenario):
    log = run_scenario(scenario, feature_mode="fixture")
    E_final = log["E_final"]
    I_inhibit = log["I_inhibit"]
    HabituationSignal = log["HabituationSignal"]
    HistoricalResidual = log["HistoricalResidual"]
    M_text = log["M_text"]

    prompt_out = state_to_prompt(E_final, I_inhibit, M_text,
                                 HistoricalResidual, HabituationSignal)

    print(f"=== 场景：{scenario} ===")
    print("真实值：")
    print(f"  E_final[0] = {E_final[0]:.6f}")
    print(f"  I_inhibit = {I_inhibit:.6f}")
    print(f"  HabituationSignal = {HabituationSignal:.6f}")
    print(f"  HistoricalResidual = {HistoricalResidual:.6f}")
    print(f"  M_text = {repr(M_text)}")
    print("state_to_prompt 输出：")
    print(prompt_out)
    print()


def main():
    for sc in ["被骂", "被夸", "被质疑"]:
        check(sc)

    print("=== 人工核对清单 ===")
    print("请确认：")
    print("  1. 被骂场景的感受应为'有点被冒犯，心里不太舒服'")
    print("  2. 被夸场景的感受应为'心情不错，挺开心的'")
    print("  3. 三场景的'隐约的回忆'均应基于真实 M_text 判断")
    print("     （空历史时必须是'脑子里没什么特别的念头'）")
    print()
    print("探针完成")


if __name__ == "__main__":
    main()
