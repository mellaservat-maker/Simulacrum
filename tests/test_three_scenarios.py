# ============================================================
# test_three_scenarios.py · V4.2
# 规格：第七部分 test_three_scenarios / Step 6 验收
# ============================================================

import pytest

from pipeline import run_scenario


@pytest.mark.skip(reason="M1 三场景标准已弃，M3 重建中")
def test_action_expectations():
    assert run_scenario("被骂", feature_mode="fixture")["action"] == "suppress"
    assert run_scenario("被夸", feature_mode="fixture")["action"] == "task_first"
    assert run_scenario("被质疑", feature_mode="fixture")["action"] == "reappraise"