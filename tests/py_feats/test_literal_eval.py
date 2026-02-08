from ast import literal_eval

import pytest


def test_literal_eval():
    assert literal_eval("True") is True
    assert literal_eval("False") is False
    assert literal_eval("None") is None
    with pytest.raises(ValueError):
        literal_eval("true")
    assert literal_eval("[1, 2, 3]") == [1, 2, 3]
    assert literal_eval("(1, 2, 3)") == (1, 2, 3)
    assert literal_eval("{'a': 1}") == {"a": 1}
