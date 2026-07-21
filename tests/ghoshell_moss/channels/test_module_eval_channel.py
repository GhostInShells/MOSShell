import json
import pytest
from ghoshell_moss.channels.module_eval_channel import new_module_eval_channel
from ghoshell_moss.tools.module_eval import ModuleEval

COUNTER_MODULE = """
from collections import Counter
data = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
def top(n=3):
    '''Return the top n most common elements.'''
    return data.most_common(n)
"""


@pytest.fixture
def counter_py(tmp_path):
    """Write a .py file for ModuleEval to use as domain module."""
    p = tmp_path / "counter_domain.py"
    p.write_text(COUNTER_MODULE)
    return str(p)


@pytest.fixture
def simple_py(tmp_path):
    p = tmp_path / "simple.py"
    p.write_text("x = 1\n")
    return str(p)


class TestModuleEvalChannelCommands:
    """exec / vars / api — subprocess sandbox."""

    @pytest.mark.asyncio
    async def test_commands_registered(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            cmd_names = {c.name for c in runtime.self_meta().commands}
            assert cmd_names == {"exec", "vars", "api"}

    @pytest.mark.asyncio
    async def test_exec_runs_code(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "print(x + 1)"})
            assert "2" in result

    @pytest.mark.asyncio
    async def test_exec_variable_persistence(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("exec", kwargs={"text__": "x = x + 10"})
            result = await runtime.execute_command("api", args=("x",))
            assert "11" in result

    @pytest.mark.asyncio
    async def test_vars_shows_namespace(self, counter_py):
        chan = new_module_eval_channel(counter_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("vars")
            ns = json.loads(result)
            assert "data" in ns
            assert "top" in ns

    @pytest.mark.asyncio
    async def test_api_shows_name_detail(self, counter_py):
        chan = new_module_eval_channel(counter_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("data",))
            assert "Counter" in result
            assert "most_common" in result

    @pytest.mark.asyncio
    async def test_api_missing_name(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("nonexistent",))
            assert "not defined" in result

    @pytest.mark.asyncio
    async def test_api_without_name(self, counter_py):
        chan = new_module_eval_channel(counter_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api")
            assert "Counter" in result
            assert "from collections import Counter" in result


class TestModuleEvalChannelInstruction:
    """Source as instruction."""

    @pytest.mark.asyncio
    async def test_instruction_contains_source(self, counter_py):
        chan = new_module_eval_channel(counter_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
            assert "Counter" in meta.instruction
            assert "most_common" in meta.instruction

    @pytest.mark.asyncio
    async def test_channel_meta(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="my_eval", description="custom desc")
        async with chan.bootstrap() as runtime:
            meta = runtime.self_meta()
            assert meta.name == "my_eval"
            assert meta.description == "custom desc"


class TestModuleEvalChannelSecurity:
    """Sandbox: AI code blocked from __import__."""

    @pytest.mark.asyncio
    async def test_exec_blocks_open(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "open('/etc/passwd')"})
            assert "NameError" in result

    @pytest.mark.asyncio
    async def test_exec_blocks_import(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "import os"})
            assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_exec_allows_safe_builtins(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "print(len('hello'))"})
            assert "5" in result


class TestModuleEvalChannelErrorHandling:
    """Exception → traceback, namespace preserved."""

    @pytest.mark.asyncio
    async def test_exception_returns_traceback(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "1/0"})
            assert "ZeroDivisionError" in result

    @pytest.mark.asyncio
    async def test_namespace_preserved_after_error(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("exec", kwargs={"text__": "1/0"})
            result = await runtime.execute_command("api", args=("x",))
            assert "1" in result
