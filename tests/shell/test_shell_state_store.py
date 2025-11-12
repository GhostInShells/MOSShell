

import pytest
from pydantic import Field

from ghoshell_moss import Interpreter
from ghoshell_moss.concepts.states import StateBaseModel, StateStore, MemoryStateStore, StateModel


@pytest.mark.asyncio
async def test_shell_state_store_baseline():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    chan = shell.main_channel.new_child('a')

    @chan.build.state_model()
    class TestStateModel(StateBaseModel):
        state_name = "test"
        state_desc = "test state model"

        value: int = Field(default=0, description="test value")

    @chan.build.command()
    async def set_value(value: int) -> int:
        test_state = await chan.broker.states.get_model(TestStateModel)
        test_state.value = value
        await chan.broker.states.save(test_state)
    
    @chan.build.command()
    async def get_value() -> int:
        test_state = await chan.broker.states.get_model(TestStateModel)
        return test_state.value


    async with shell:
        interpreter = await shell.interpreter()
        assert isinstance(interpreter, Interpreter)
        assert shell.is_running()
        set_cmd = await shell.get_command("a", "set_value")
        assert set_cmd is not None
        get_cmd = await shell.get_command("a", "get_value")
        assert get_cmd is not None
        async with interpreter:
            interpreter.feed("<a:set_value value=\"123\" /><a:get_value />")
            assert shell.is_running()
            tasks = await interpreter.wait_execution_done(1)

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.success()
                result.append(task.result())
            # 获取到结果.
            assert result == [None, 123]
            assert ['a', 'a'] == [t.exec_chan for t in tasks.values()]


@pytest.mark.asyncio
async def test_shell_state_store_share():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')
    b_chan = shell.main_channel.new_child('b')

    @a_chan.build.state_model()
    class TestStateModel(StateBaseModel):
        state_name = "test"
        state_desc = "test state model"

        value: int = Field(default=0, description="test value")
    
    @a_chan.build.command()
    async def set_value(value: int) -> int:
        test_state = await a_chan.broker.states.get_model(TestStateModel)
        test_state.value = value
        await a_chan.broker.states.save(test_state)

    @b_chan.build.command()
    async def get_value() -> int:
        test_state = await b_chan.broker.states.get_model(TestStateModel)
        return test_state.value

    async with shell:
        interpreter = await shell.interpreter()
        assert isinstance(interpreter, Interpreter)
        assert shell.is_running()
        set_cmd = await shell.get_command("a", "set_value")
        assert set_cmd is not None
        get_cmd = await shell.get_command("b", "get_value")
        assert get_cmd is not None
        async with interpreter:
            interpreter.feed("<a:set_value value=\"123\" /><b:get_value />")
            assert shell.is_running()
            tasks = await interpreter.wait_execution_done(1)

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.success()
                result.append(task.result())
            # 获取到结果.
            assert result == [None, 123]
            assert ['a', 'b'] == [t.exec_chan for t in tasks.values()]