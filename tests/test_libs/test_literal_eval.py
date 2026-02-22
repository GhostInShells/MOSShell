from ast import literal_eval
import pytest


def test_literal_eval():
    value_err_cases = [
        'abc',
        '3 * 5',
        'none',
        'true',
        'false',
    ]
    for value in value_err_cases:
        with pytest.raises(ValueError):
            literal_eval(value)

    good_cases = [
        ('1', 1),
        ('None', None),
        ('False', False),
    ]

    for value, parsed in good_cases:
        assert literal_eval(value) == parsed
