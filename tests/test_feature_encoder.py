# ============================================================
# test_feature_encoder.py · V4.2
# 规格：第七部分 test_feature_encoder
# ============================================================

from feature_encoder import pos_count, neg_count, punctuation_heuristic, task_relevance


def test_empty_text():
    assert pos_count("") == 0
    assert neg_count("") == 0


def test_punctuation_only():
    assert punctuation_heuristic("!!!") == 1.0
    assert punctuation_heuristic("???????") == 1.0
    assert punctuation_heuristic("...") == 0.0


def test_zuobuhao_not_misjudged():
    assert pos_count("做不好") == 0


def test_task_relevance():
    class E1:
        nodes = {"task": ["T1"]}
    class E2:
        nodes = {}
    assert task_relevance(E1()) == 1.0
    assert task_relevance(E2()) == 0.0