# ============================================================
# probe_prompt_render.py · 提示词渲染探针
# 对三个场景分别调用 state_to_prompt，人工核对输出
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from state_translator import state_to_prompt


def render_and_print(label, E_final_val, I_inhibit_val, M_text_val,
                     HistoricalResidual_val, HabituationSignal_val):
    E_final = np.array([E_final_val, 0.0, 0.0, 0.0])
    out = state_to_prompt(E_final, I_inhibit_val, M_text_val,
                          HistoricalResidual_val, HabituationSignal_val)
    print(f"=== {label} ===")
    print(out)
    print()


def main():
    # 被骂：负情绪 + 高抑制 + 新鲜 + 有 M_text
    render_and_print(
        "被骂",
        E_final_val=-0.3,
        I_inhibit_val=0.85,
        M_text_val="记得上次被骂的时候很生气",
        HistoricalResidual_val=0.6,
        HabituationSignal_val=0.2,
    )

    # 被夸：积极 + 低抑制 + 新鲜 + 有 M_text
    render_and_print(
        "被夸",
        E_final_val=0.6,
        I_inhibit_val=0.1,
        M_text_val="记得上次被夸的时候很开心",
        HistoricalResidual_val=0.3,
        HabituationSignal_val=0.15,
    )

    # 被质疑：中性 + 中等抑制 + 有点熟悉 + 有 M_text
    render_and_print(
        "被质疑",
        E_final_val=0.0,
        I_inhibit_val=0.5,
        M_text_val="记得上次被质疑的时候有点纠结",
        HistoricalResidual_val=0.4,
        HabituationSignal_val=0.65,
    )

    # 人工核对清单
    print("=== 人工核对清单 ===")
    print("请检查：")
    print("  1. 输出中没有数字（除规则序号外）")
    print("  2. 输出中没有'竞争分'、'M_text'、'HabituationSignal'、'HistoricalResidual'等内部变量名")
    print("  3. 输出中包含'【你此刻的状态】'和'【输出规则】'两段")
    print("  4. 输出中'隐约的回忆'是模糊描述，不含 M_text 原文")
    print("  5. 输出中包含硬约束五条")
    print()
    print("探针完成")


if __name__ == "__main__":
    main()
