from ghoshell_codex import runtime_compile
import pytest


@pytest.mark.asyncio
async def test_runtime_compile_with_async_func():
    import math
    from math import floor, sin, pi
    import inspect
    async def math_example() -> float:
        return floor(sin(pi / 2))

    assert inspect.isbuiltin(floor)

    value = await math_example()
    # 直接用 math_example 做测试.
    source = inspect.getsource(math_example)
    compiler = runtime_compile(math, source)
    assert await compiler.get('math_example')() == value


def test_runtime_compile_invalid_code():
    code = """floor()"""
    with pytest.raises(SyntaxError):
        runtime_compile(None, code)


def test_contaminate_while_compile():
    code1 = """
a = 123
b = 'foo'
"""
    compiled1 = runtime_compile(None, code1)
    code2 = """
a = 456
b = 'bar'
"""
    compiled2 = runtime_compile(compiled1.compiled, code2)
    assert compiled2.get('a') != compiled1.get('a')
    assert compiled2.get('a') == 456
    assert compiled1.get('b') == 'foo'
