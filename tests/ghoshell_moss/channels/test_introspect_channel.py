"""Tests for introspect channel — runtime reflection of the live MOSS runtime.

Covers:
- 命令面注册 (get-interface / get-source / where / list / architecture)
- instruction 声明 scope 边界 + "运行时读取"语义
- get-interface: 反射 to-scope 模块,scope 拒绝对 in-scope 之外
- get-source: 读活对象源码(来自加载出处)
- where: 权威=活对象 provenance
- list: 包子模块 + 模块成员
- architecture: 策展地图
- build_ 工厂按声明的 scope 派生 channel
"""

from __future__ import annotations

import pytest

from ghoshell_moss.channels.introspect_channel import (
    new_introspect_channel,
    build_introspect_channel,
)


@pytest.mark.asyncio
async def test_commands_registered():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"get-interface", "get-source", "where", "list", "architecture"}


@pytest.mark.asyncio
async def test_instruction_declares_scope_and_runtime_read():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert "Scope boundary" in meta.instruction
        assert "ghoshell_moss" in meta.instruction
        assert "runtime" in meta.instruction


@pytest.mark.asyncio
async def test_get_interface_reflects_in_scope_module():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("get-interface", args=("ghoshell_moss.core.concepts.channel",))
        assert "class Channel" in result


@pytest.mark.asyncio
async def test_get_interface_with_deps():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "get-interface",
            kwargs={"import_path": "ghoshell_moss.channels.module_eval_channel", "deps": True},
        )
        assert "more attr information" in result


@pytest.mark.asyncio
async def test_scope_gate_rejects_out_of_scope():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("get-interface", args=("os",))
        assert "outside the declared scope" in result


@pytest.mark.asyncio
async def test_scope_wide_narrow_declared_at_build():
    """scope 是构建时声明:扩宽到 core 子树则允许 core,仍拒绝其它."""
    chan = new_introspect_channel(scope=("ghoshell_moss.core",))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("where", args=("ghoshell_moss.core.concepts.command",))
        assert "Canonical:" in result
        result2 = await runtime.execute_command("where", args=("ghoshell_moss.channels.moss_cli",))
        assert "outside the declared scope" in result2


@pytest.mark.asyncio
async def test_get_source_reads_live_object_source():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("get-source", args=("ghoshell_moss.channels.introspect_channel",))
        assert "new_introspect_channel" in result
        assert "# file:" in result
        assert "# lines" in result


@pytest.mark.asyncio
async def test_get_source_line_range_pages():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        full = await runtime.execute_command("get-source", args=("ghoshell_moss.channels.introspect_channel",))
        head = await runtime.execute_command(
            "get-source",
            kwargs={"import_path": "ghoshell_moss.channels.introspect_channel", "lines": "1-5"},
        )
        assert "# lines 1-5 /" in head
        assert "def new_introspect_channel" not in head
        assert "def new_introspect_channel" in full


@pytest.mark.asyncio
async def test_get_source_line_range_invalid():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "get-source",
            kwargs={"import_path": "ghoshell_moss.channels.introspect_channel", "lines": "abc"},
        )
        assert "invalid line range" in result


@pytest.mark.asyncio
async def test_where_reports_live_provenance():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("where", args=("ghoshell_moss.core.concepts.channel:Channel",))
        assert "Canonical:" in result
        assert "ghoshell_moss.core.concepts.channel:Channel" in result
        assert "channel.py" in result


@pytest.mark.asyncio
async def test_list_package_submodules():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("list", args=("ghoshell_moss.channels",))
        assert "Package: ghoshell_moss.channels" in result
        assert "introspect_channel" in result
        assert "moss_cli" in result


@pytest.mark.asyncio
async def test_list_module_members():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("list", args=("ghoshell_moss.channels.moss_cli",))
        assert "def new_moss_cli_channel" in result
        assert "def build_moss_cli_channel" in result


@pytest.mark.asyncio
async def test_architecture_map():
    chan = new_introspect_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("architecture")
        assert "MOSS architecture map" in result
        assert "ghoshell_moss.core.concepts" in result


@pytest.mark.asyncio
async def test_build_factory_yields_channel_with_declared_scope():
    factory = build_introspect_channel(name="intro")
    chan = factory(None)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().name == "intro"
