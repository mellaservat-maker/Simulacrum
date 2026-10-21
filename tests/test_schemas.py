# ============================================================
# test_schemas.py · V4.2
# 规格：第七部分 test_schemas
# ============================================================

from config import n, m, k, c, n_f, XG_W_ACT, XG_W_MATCH, w_f, w_e, POSITIVE_WORDS, DANGER_KEYWORDS
from schemas import HippocampalOutput, EventNode


def test_hippocampal_field_name():
    assert "M" in HippocampalOutput.__annotations__
    assert "CurrentThought_Projected" not in HippocampalOutput.__annotations__


# T3-B：M 维度由 8 改为 1024（语义嵌入接入）
def test_dimensions():
    assert n == 8 and m == 1024 and k == 4 and c == 4 and n_f == 4


def test_weight_lengths():
    assert len(XG_W_ACT) == 5
    assert len(XG_W_MATCH) == 4
    assert len(w_f) == n_f
    assert len(w_e) == c


def test_keyword_intersection():
    assert not (set(POSITIVE_WORDS) & set(DANGER_KEYWORDS))


def test_event_embedding_field():
    assert "event_embedding" in EventNode.__annotations__