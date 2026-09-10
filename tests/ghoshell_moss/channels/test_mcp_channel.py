"""Tests for MCP Hub Channel — 重新对齐后的 hub 契约。

MCPHub 是本源对象，build_mcp_hub_channel 投影成 channel。测试覆盖：
公开方法契约、config 增删（只写不连）、open/close 生命周期、子 channel 的
schema notice 与 call/acall 阻塞语义、虚拟子 channel 对齐，以及真实 stdio
MCP server 的端到端集成。
"""

import sys
import tempfile
from os.path import dirname, join
from pathlib import Path

import pytest
from mcp import types as mcp_types

from ghoshell_moss.channels.mcp_channel import (
    MCPHub,
    MCPServerSession,
    _new_server_channel,
    _reconcile_children,
    build_mcp_hub_channel,
    mcp_result_to_observe,
    render_input_schema,
    render_tools_notice,
)
from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.mcp.config import MCPHubConfig, MCPServerConfig

_HELPER_PATH = join(dirname(dirname(__file__)), "bridges", "mcp_channel", "helper", "mcp_server_demo.py")


def _new_config_store():
    return YamlConfigStore(LocalStorage(Path(tempfile.mkdtemp())))


def _hub(**kwargs) -> MCPHub:
    return MCPHub(config_store=_new_config_store(), name="mcp", **kwargs)


def _stdio_demo_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="demo",
        transport="stdio",
        command=sys.executable,
        args=[_HELPER_PATH],
    )


class _FakeMatrix:
    """只暴露 build_mcp_hub_channel 用到的 ``configs`` 面。"""

    def __init__(self, store):
        self.configs = store


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

class TestRenderInputSchema:
    def test_renders_params(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the text"},
                "n": {"type": "integer"},
            },
            "required": ["text"],
        }
        out = render_input_schema(schema)
        assert "`text`" in out
        assert "required" in out
        assert "`n`" in out

    def test_empty_schema(self):
        assert render_input_schema({}) == ""
        assert render_input_schema({"type": "object", "properties": {}}) == ""


class TestRenderToolsNotice:
    def test_disconnected(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        assert "disconnected" in render_tools_notice(session)

    def test_connected_renders_schema(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        session.state = "connected"
        session.tools = [
            mcp_types.Tool(
                name="add",
                description="add two numbers",
                input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            ),
        ]
        out = render_tools_notice(session)
        assert "add(" in out
        assert "`x`" in out


class TestResultToObserve:
    def test_text_content(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="hello world")],
            is_error=False,
        )
        obs = mcp_result_to_observe(result, server="s", tool="t")
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1

    def test_error_result(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="boom")],
            is_error=True,
        )
        obs = mcp_result_to_observe(result, server="s", tool="t")
        assert isinstance(obs, Observe)
        assert "error" in obs.messages[0].to_content_string()


# ---------------------------------------------------------------------------
# MCPHub 公开方法契约
# ---------------------------------------------------------------------------

class TestMCPHub:
    @pytest.mark.asyncio
    async def test_open_invalid_name(self):
        result = await _hub().open("bad-name")
        assert "invalid server name" in result

    @pytest.mark.asyncio
    async def test_open_missing_config(self):
        result = await _hub().open("nope")
        assert "not found in config" in result

    @pytest.mark.asyncio
    async def test_add_server_writes_config_only(self):
        hub = _hub(allow_config_edit=True)
        result = await hub.add_server(
            '{"name": "demo", "transport": "stdio", "command": "python", "connect": true}'
        )
        assert "config added" in result
        # 只写配置、不连接
        assert hub.sessions() == {}
        listing = await hub.list_servers()
        assert "demo" in listing
        assert "not open" in listing

    @pytest.mark.asyncio
    async def test_remove_server_removes_config(self):
        hub = _hub(allow_config_edit=True)
        await hub.add_server('{"name": "demo", "transport": "stdio", "command": "python"}')
        result = await hub.remove_server("demo")
        assert "removed" in result
        assert "No MCP servers configured." in await hub.list_servers()

    @pytest.mark.asyncio
    async def test_authorize_is_pass_through(self):
        assert await _hub().authorize("open", "x") is None

    @pytest.mark.asyncio
    async def test_open_connect_failure_is_recorded(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={
            "demo": MCPServerConfig(name="demo", transport="stdio", command="definitely-not-a-command"),
        }))
        hub = MCPHub(config_store=store)
        result = await hub.open("demo")
        assert "error" in result
        assert "demo" in hub.sessions()
        # close 保留 config，只断开
        close_result = await hub.close("demo")
        assert "closed" in close_result
        assert "demo" not in hub.sessions()
        # config 仍在，可再次 open
        assert "demo" in hub._load_config().servers


# ---------------------------------------------------------------------------
# server 子 channel 契约
# ---------------------------------------------------------------------------

class TestServerChannel:
    def test_name_and_notice(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        session.state = "connected"
        session.tools = [
            mcp_types.Tool(name="add", description="add", input_schema={"type": "object"}),
        ]
        chan = _new_server_channel(session)
        assert chan.name() == "demo"

    def test_call_blocking_acall_nonblocking(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        chan = _new_server_channel(session)
        assert chan.build.get_own_command("call").meta().blocking is True
        assert chan.build.get_own_command("acall").meta().blocking is False

    def test_no_config_commands(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        chan = _new_server_channel(session)
        assert set(chan.build.own_commands()) == {"call", "acall"}


# ---------------------------------------------------------------------------
# 虚拟子 channel 对齐
# ---------------------------------------------------------------------------

class TestReconcileChildren:
    def test_add_and_remove_virtual_child(self):
        hub = _hub()
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        session.state = "connected"
        hub._sessions["demo"] = session

        chan = new_channel(name="mcp")
        _reconcile_children(hub, chan)
        children = chan.virtual_children()
        assert "demo" in children
        assert children["demo"].name() == "demo"

        # 关掉后子 channel 被移除
        hub._sessions.pop("demo")
        _reconcile_children(hub, chan)
        assert chan.virtual_children() == {}


# ---------------------------------------------------------------------------
# build_mcp_hub_channel 命令面
# ---------------------------------------------------------------------------

class TestBuildChannel:
    def test_commands_without_config_edit(self):
        chan = build_mcp_hub_channel(_FakeMatrix(_new_config_store()))
        commands = set(chan.build.own_commands())
        assert {"list", "open", "close"} <= commands
        assert "add" not in commands
        assert "remove" not in commands

    def test_commands_with_config_edit(self):
        chan = build_mcp_hub_channel(_FakeMatrix(_new_config_store()), allow_config_edit=True)
        commands = set(chan.build.own_commands())
        assert {"list", "open", "close", "add", "remove"} <= commands


# ---------------------------------------------------------------------------
# 集成 — 真实 stdio MCP server
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_open_connect_call_close(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={"demo": _stdio_demo_config()}))
        hub = MCPHub(config_store=store)

        result = await hub.open("demo")
        assert "connected" in result

        session = hub.sessions()["demo"]
        assert session.state == "connected"
        assert {t.name for t in session.tools} >= {"add", "foo"}

        obs = await session.call_tool("add", {"x": 1, "y": 2})
        assert isinstance(obs, Observe)
        assert "3" in obs.messages[0].to_content_string()

        # 子 channel notice 反映 schema
        notice = render_tools_notice(session)
        assert "add(" in notice

        close_result = await hub.close("demo")
        assert "closed" in close_result
        assert "demo" not in hub.sessions()
