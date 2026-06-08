import pytest
from ghoshell_moss.channels.module_eval_channel import new_module_eval_channel

SIMPLE_MODULE = """
from collections import Counter
data = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
def top(n=3):
    return data.most_common(n)
"""


class TestModuleEvalChannelCommands:
    """核心命令：exec / vars / api。"""

    @pytest.mark.asyncio
    async def test_commands_registered(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            cmd_names = {c.name for c in runtime.self_meta().commands}
            assert cmd_names == {"exec", "vars", "api"}

    @pytest.mark.asyncio
    async def test_exec_runs_code(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "print(x + 1)"})
            assert "2" in result

    @pytest.mark.asyncio
    async def test_exec_variable_persistence(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("exec", kwargs={"text__": "x = x + 10"})
            result = await runtime.execute_command("vars", args=("x",))
            assert "11" in result

    @pytest.mark.asyncio
    async def test_vars_no_args_lists_public(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("vars")
            assert "data" in result
            assert "top" in result
            assert "Counter" in result

    @pytest.mark.asyncio
    async def test_vars_with_names_shows_values(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("vars", args=("data",))
            assert "data:" in result
            assert "Counter" in result

    @pytest.mark.asyncio
    async def test_vars_missing_name(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("vars", args=("nonexistent",))
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_api_list_methods(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("data",))
            assert "most_common" in result
            assert "Public methods of 'data'" in result

    @pytest.mark.asyncio
    async def test_api_specific_method(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("data", "most_common"))
            assert "most_common" in result
            assert "n most common" in result.lower()

    @pytest.mark.asyncio
    async def test_api_missing_object(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("nonexistent",))
            assert "not found" in result

    @pytest.mark.asyncio
    async def test_api_missing_method(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("api", args=("data", "no_such_method"))
            assert "not found" in result


class TestModuleEvalChannelInstruction:
    """模块源码即 instruction。"""

    @pytest.mark.asyncio
    async def test_instruction_contains_source(self):
        chan = new_module_eval_channel(SIMPLE_MODULE, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
            assert "Counter" in meta.instruction
            assert "most_common" in meta.instruction

    @pytest.mark.asyncio
    async def test_channel_meta(self):
        chan = new_module_eval_channel("x = 1", channel_name="my_eval", description="custom desc")
        async with chan.bootstrap() as runtime:
            meta = runtime.self_meta()
            assert meta.name == "my_eval"
            assert meta.description == "custom desc"


class TestModuleEvalChannelSecurity:
    """安全边界：模型代码受 builtins 限制。"""

    @pytest.mark.asyncio
    async def test_exec_blocks_open(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "open('/etc/passwd')"})
            assert "NameError" in result

    @pytest.mark.asyncio
    async def test_exec_blocks_import(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "import os"})
            assert "ImportError" in result

    @pytest.mark.asyncio
    async def test_exec_allows_safe_builtins(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "print(len('hello'))"})
            assert "5" in result


class TestModuleEvalChannelErrorHandling:
    """异常返回 traceback。"""

    @pytest.mark.asyncio
    async def test_exception_returns_traceback(self):
        chan = new_module_eval_channel("x = 1", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("exec", kwargs={"text__": "1/0"})
            assert "ZeroDivisionError" in result
            assert "Traceback" in result

    @pytest.mark.asyncio
    async def test_namespace_preserved_after_error(self):
        chan = new_module_eval_channel("x = 42", channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("exec", kwargs={"text__": "1/0"})
            result = await runtime.execute_command("vars", args=("x",))
            assert "42" in result
