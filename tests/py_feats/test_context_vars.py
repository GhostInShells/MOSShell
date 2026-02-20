import contextvars
import pytest


def test_context_vars_get_none():
    var = contextvars.ContextVar("var")

    def foo():
        return var.get()

    with pytest.raises(LookupError):
        foo()
