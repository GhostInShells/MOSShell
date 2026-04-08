from ghoshell_moss.core.codex.executor import Executor
from ghoshell_moss.core.codex import compiler
import asyncio


def test_execute_baseline():
    executor = Executor(
        compiler,

    )
    r = executor.execute(
        code=("if __name__ == '__execute__': "
              "    __result__ = 123")
    )
    assert r.returns == 123

    async def run():
        _r = executor.execute(
            code=("async def foo():"
                  "    return 123"),
            func_name="foo",
        )

        return await _r.returns

    assert asyncio.run(run()) == 123

    r = executor.execute(
        code=("if __name__ == '__execute__': "
              "    print('hello')")
    )
    assert r.std_output == 'hello\n'
    assert 'foo' not in compiler.__dict__
