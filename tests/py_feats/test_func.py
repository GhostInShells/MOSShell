import pytest


def test_func_with_args():
    def foo(*args) -> int:
        return len(list(args))

    assert foo() == 0


def test_raise_exception():
    def foo():
        raise ValueError

    with pytest.raises(ValueError):
        foo()


def test_passed_func():
    def foo():
        pass

    assert foo() is None
