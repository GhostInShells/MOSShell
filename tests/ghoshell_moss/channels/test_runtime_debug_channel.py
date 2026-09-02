"""Tests for runtime_debug_channel — 编译临时 module, await 其 main(container).

Covers:
- 命令面注册 (debug)
- instruction 声明"live runtime / async def / asyncio 已注入"语义
- async main: stdout + 返回值被捕获
- no main / main 非 async (拒绝, 防止阻塞 event loop)
- 编译错误 (SyntaxError) 与 运行错误 (异常) 兜底
- asyncio.to_thread 在模型体内可用 (asyncio 已注入)
- build_ 工厂按声明的 name 派生 channel
"""

from __future__ import annotations

import pytest

from ghoshell_moss.channels.runtime_debug_channel import (
    new_runtime_debug_channel,
    build_runtime_debug_channel,
)


def _chan(**kw):
    return new_runtime_debug_channel(**kw)


@pytest.mark.asyncio
async def test_commands_registered():
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"debug"}


@pytest.mark.asyncio
async def test_instruction_declares_contract():
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert "live" in meta.instruction
        assert "async def" in meta.instruction
        assert "asyncio" in meta.instruction


@pytest.mark.asyncio
async def test_async_main_returns_and_prints():
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": "async def main(container):\n    print('ok')\n    return {'n': 1}\n"},
        )
        assert "--- stdout ---" in result
        assert "ok" in result
        assert "--- result ---" in result
        assert "{'n': 1}" in result


@pytest.mark.asyncio
async def test_no_main_reports_error():
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": "async def other():\n    pass\n"},
        )
        assert "no `main`" in result
        assert "async def main(container)" in result


@pytest.mark.asyncio
async def test_plain_def_main_rejected():
    """同步 def 会被拒绝 —— 直接调用会阻塞事件循环."""
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": "def main(container):\n    return 1\n"},
        )
        assert "must be `async def`" in result
        assert "event loop" in result
        assert "asyncio.to_thread" in result


@pytest.mark.asyncio
async def test_syntax_error_reported():
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": "async def main(container):\n    x = \n"},
        )
        assert "COMPILE ERROR" in result
        assert "SyntaxError" in result


@pytest.mark.asyncio
async def test_runtime_error_reported_with_filtered_traceback():
    """运行异常兜底, traceback 只留模型代码 frame, 不含 channel 内部栈."""
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": "async def main(container):\n    return 1 / 0\n"},
        )
        assert "RUN ERROR" in result
        assert "ZeroDivisionError" in result
        assert "moss_codex_temp_module" in result
        assert "runtime_debug_channel.py" not in result


@pytest.mark.asyncio
async def test_to_thread_available_inside_main():
    """asyncio 已注入 —— 模型体内可用 asyncio.to_thread 包阻塞代码."""
    chan = _chan()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "debug",
            kwargs={"text__": (
                "async def main(container):\n"
                "    def block():\n"
                "        return 42\n"
                "    val = await asyncio.to_thread(block)\n"
                "    return {'val': val}\n"
            )},
        )
        assert "-- result --" in result
        assert "{'val': 42}" in result


@pytest.mark.asyncio
async def test_build_factory_yields_channel_with_declared_name():
    factory = build_runtime_debug_channel(name="dbg")
    chan = factory(None)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().name == "dbg"
