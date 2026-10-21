# ============================================================
# test_full_pipeline_smoke.py · V4.2
# 规格：第七部分 test_full_pipeline_smoke / Step 6 encoder 冒烟
# ============================================================

from pipeline import run_scenario


def test_encoder_mode_no_crash():
    for scenario in ["被骂", "被夸", "被质疑"]:
        log = run_scenario(scenario, feature_mode="encoder")
        for f in ["x", "x_fast", "B", "D_fast", "E_raw", "S_E", "I_inhibit", "E_final", "action"]:
            assert f in log