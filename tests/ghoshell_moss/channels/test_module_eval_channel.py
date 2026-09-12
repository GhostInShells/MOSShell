"""ModuleEval channel — live module-level stateful runtime.

行为测试: exec/aexec/history 命令面, 有状态持久化, 真超时, builtins 放开,
hub 治理多 module (open/close/list + 子 channel 可执行)。
"""
import asyncio

import pytest

from ghoshell_moss.channels.module_eval_channel import (
    new_module_eval_channel,
    new_sandbox_hub_channel,
)
from ghoshell_moss.core.concepts.errors import CommandError

COUNTER_MODULE = """
from collections import Counter
data = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
def top(n=3):
    '''Return the top n most common elements.'''
    return data.most_common(n)
"""


@pytest.fixture
def counter_py(tmp_path):
    p = tmp_path / "counter_domain.py"
    p.write_text(COUNTER_MODULE)
    return str(p)


@pytest.fixture
def simple_py(tmp_path):
    p = tmp_path / "simple.py"
    p.write_text("x = 1\n")
    return str(p)


class TestModuleEvalChannelCommands:
    @pytest.mark.asyncio
    async def test_commands_registered(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            cmd_names = {c.name for c in runtime.self_meta().commands}
            assert cmd_names == {"exec", "aexec", "history"}

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
            result = await runtime.execute_command("exec", kwargs={"text__": "print(x)"})
            assert "11" in result

    @pytest.mark.asyncio
    async def test_exec_import_allowed(self, simple_py):
        """builtins 放开 — domain 源码即声明边界, 不强制白名单."""
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command(
                "exec", kwargs={"text__": "import json; print(json.dumps({'a': 1}))"}
            )
            assert '{"a": 1}' in result

    @pytest.mark.asyncio
    async def test_exec_timeout_raises(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            with pytest.raises(CommandError):
                await runtime.execute_command(
                    "exec", kwargs={"text__": "import time; time.sleep(2)", "timeout": 0.1}
                )

    @pytest.mark.asyncio
    async def test_aexec_lands_in_history(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            r = await runtime.execute_command("aexec", kwargs={"text__": "print('bg')"})
            assert "queued" in r
            await asyncio.sleep(0.3)
            h = await runtime.execute_command("history")
            assert "print('bg')" in h
            assert "bg" in h

    @pytest.mark.asyncio
    async def test_history_includes_results(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("exec", kwargs={"text__": "print(x + 1)"})
            h = await runtime.execute_command("history")
            assert "print(x + 1)" in h
            assert "2" in h


class TestModuleEvalChannelInstruction:
    @pytest.mark.asyncio
    async def test_instruction_declares_runtime_and_source(self, counter_py):
        chan = new_module_eval_channel(counter_py, channel_name="test")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.self_meta()
            assert "persistent Python runtime" in meta.instruction
            assert "Counter" in meta.instruction

    @pytest.mark.asyncio
    async def test_channel_meta(self, simple_py):
        chan = new_module_eval_channel(simple_py, channel_name="my_eval", description="custom desc")
        async with chan.bootstrap() as runtime:
            meta = runtime.self_meta()
            assert meta.name == "my_eval"
            assert meta.description == "custom desc"


class TestModuleEvalChannelErrors:
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
            result = await runtime.execute_command("exec", kwargs={"text__": "print(x)"})
            assert "1" in result


class TestSandboxHubChannel:
    @pytest.fixture
    def domains(self, tmp_path):
        d = tmp_path / "domains"
        d.mkdir()
        (d / "counter.py").write_text(COUNTER_MODULE)
        return str(d)

    @pytest.mark.asyncio
    async def test_open_close_list(self, domains):
        chan = new_sandbox_hub_channel(None, domains, name="sandbox")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            r = await runtime.execute_command("list")
            assert "counter" in r
            r = await runtime.execute_command("open", args=("counter",))
            assert "opened" in r
            await runtime.refresh_metas()
            assert runtime.get_child_channel("counter") is not None
            r = await runtime.execute_command("open", args=("nope",))
            assert "no domain" in r
            r = await runtime.execute_command("close", args=("counter",))
            assert "closed" in r

    @pytest.mark.asyncio
    async def test_child_channel_is_executable(self, domains):
        chan = new_sandbox_hub_channel(None, domains, name="sandbox")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("open", args=("counter",))
            await runtime.refresh_metas()
            sub = runtime.fetch_sub_runtime("counter")
            r = await sub.execute_command("exec", kwargs={"text__": "print(top(2))"})
            assert "a" in r
