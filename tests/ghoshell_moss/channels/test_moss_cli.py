"""Tests for moss_cli channel — de-authorized moss CLI exposure.

Covers:
- 命令注册 + instruction 提及 moss_cli
- exec 桥接到 python -m ghoshell_moss.cli (codex concepts)
- 误带 moss/--ai 前缀被剥离
- codex eval 命令级拒绝 (去授权边界)
- 空命令友好提示
- which 报告真实解释器
"""

from __future__ import annotations

import sys

import pytest

from ghoshell_moss.channels.moss_cli import new_moss_cli_channel
from ghoshell_moss.core.subprocesses import SubprocessesImpl


def _observe_text(result) -> str:
    assert isinstance(result, str), f"expected str, got {type(result)}"
    return result


@pytest.mark.asyncio
async def test_commands_registered():
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names


@pytest.mark.asyncio
async def test_which_reports_real_interpreter():
    """which 的承诺是报真话 — 跑 exec 用的就是它报的那个解释器."""
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("which")
        assert sys.executable in _observe_text(result)


@pytest.mark.asyncio
async def test_instruction_is_name_agnostic():
    """同一 channel 会被挂到不同树上并可能改名 — instruction 不写死自己的地址."""
    chan = new_moss_cli_channel(SubprocessesImpl(), name="renamed_cli")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert instruction
        assert "renamed_cli" not in instruction
        assert "moss_cli" not in instruction


@pytest.mark.asyncio
async def test_exec_runs_moss_command():
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"text__": "codex concepts"})
        assert "concepts" in result or "blueprint" in result
        assert "exit: 0" in result


@pytest.mark.asyncio
async def test_exec_strips_moss_and_ai_prefix():
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec", kwargs={"text__": "moss --ai codex concepts"}
        )
        assert "exit: 0" in result


@pytest.mark.asyncio
async def test_exec_codex_eval_rejected_at_cli():
    """codex eval 已从 CLI 注册移除 — 通道不 deny, 由 typer 层拒绝."""
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec", kwargs={"text__": "codex eval print(1)"}
        )
        assert "No such command" in result
        assert "exit: 2" in result


@pytest.mark.asyncio
async def test_exec_empty_command_is_friendly():
    chan = new_moss_cli_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"text__": ""})
        assert "empty command" in _observe_text(result)
