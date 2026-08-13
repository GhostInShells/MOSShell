"""Tests for MCP Hub Channel (as_channel 范式).

MCPHubState 是本源对象（state-first），new_channel_from_state 投影成 channel。
这些测试覆盖协议契约：公开方法、命令语义、投影身份、子 channel 辐射、
生命周期，以及真实 stdio MCP server 的端到端集成。
"""
import sys
import tempfile
from os.path import dirname, join
from pathlib import Path

import pytest
from mcp import types as mcp_types

from ghoshell_moss.channels.mcp_channel import (
    MCPHubState,
    MCPServerChannelState,
    MCPServerSession,
    mcp_result_to_observe,
    new_channel_from_state,
    render_input_schema,
)
from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.mcp.config import MCPHubConfig, MCPServerConfig

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _new_config_store():
    return YamlConfigStore(LocalStorage(Path(tempfile.mkdtemp())))


def _state():
    return MCPHubState(config_store=_new_config_store(), name="mcp", description="test hub")


# ---------------------------------------------------------------------------
# 本源对象 — 公开方法契约
# ---------------------------------------------------------------------------

class TestPublicSurface:
    def test_public_methods_exposed(self):
        state = _state()
        for method in ["connect_server", "disconnect_server", "list_servers", "call_tool", "sessions"]:
            assert callable(getattr(state, method)), method

    @pytest.mark.asyncio
    async def test_call_tool_disconnected(self):
        obs = await _state().call_tool("nope", "tool")
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1

    @pytest.mark.asyncio
    async def test_list_servers_empty(self):
        text = await _state().list_servers()
        assert "No servers configured." in text

    def test_sessions_empty(self):
        assert _state().sessions() == {}


# ---------------------------------------------------------------------------
# 命令契约
# ---------------------------------------------------------------------------

class TestCommands:
    def test_command_names(self):
        assert set(_state().own_commands()) == {"call", "acall", "list", "connect", "disconnect"}

    def test_call_blocking_acall_nonblocking(self):
        state = _state()
        assert state.get_own_command("call").meta().blocking is True
        assert state.get_own_command("acall").meta().blocking is False

    def test_all_always_observe(self):
        state = _state()
        for name, cmd in state.own_commands().items():
            assert cmd.meta().always_observe is True, name

    @pytest.mark.asyncio
    async def test_call_invalid_json(self):
        cmd = _state().get_own_command("call")
        obs = await cmd(server="demo", tool="add", text__="not json")
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1


# ---------------------------------------------------------------------------
# 投影契约 — new_channel_from_state
# ---------------------------------------------------------------------------

class TestProjection:
    def test_channel_identity_matches_state(self):
        state = _state()
        chan = new_channel_from_state(state, id=state.id())
        assert chan.name() == "mcp"
        assert chan.id() == state.id()
        assert chan.description() == "test hub"


# ---------------------------------------------------------------------------
# 温度模型 — help (warm) vs context (hot)
# ---------------------------------------------------------------------------

class TestTemperature:
    @pytest.mark.asyncio
    async def test_empty_help(self):
        assert await _state().get_help() == "No MCP servers connected."

    @pytest.mark.asyncio
    async def test_context_is_empty(self):
        # 工具目录在 help(warm) 与 list(on-demand) 中，context 无每帧热数据。
        assert await _state().get_context_messages() == []

    def test_no_virtual_children_when_empty(self):
        assert _state().get_virtual_children() == {}


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

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
        assert len(obs.messages) == 1


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


# ---------------------------------------------------------------------------
# 子 channel 辐射 — 信息辐射器
# ---------------------------------------------------------------------------

class TestServerChannelState:
    def test_is_information_radiator(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        sub = MCPServerChannelState(session)
        assert sub.own_commands() == {}
        assert sub.get_own_command("anything") is None

    @pytest.mark.asyncio
    async def test_help_disconnected(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        sub = MCPServerChannelState(session)
        assert "disconnected" in await sub.get_help()

    @pytest.mark.asyncio
    async def test_help_lists_tools(self):
        session = MCPServerSession(config=MCPServerConfig(name="demo", command="python"))
        session.tools = [
            mcp_types.Tool(name="add", description="add", input_schema={"type": "object"}),
            mcp_types.Tool(name="foo", description="foo", input_schema={"type": "object"}),
        ]
        session.state = "connected"
        sub = MCPServerChannelState(session)
        help_text = await sub.get_help()
        assert "2 tools" in help_text
        assert "add" in help_text


# ---------------------------------------------------------------------------
# 集成 — 真实 stdio MCP server
# ---------------------------------------------------------------------------

_HELPER_PATH = join(dirname(dirname(__file__)), "bridges", "mcp_channel", "helper", "mcp_server_demo.py")


class TestIntegration:
    @pytest.mark.asyncio
    async def test_connect_call_list_disconnect(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={
            "demo": MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_HELPER_PATH],
            ),
        }))
        state = MCPHubState(config_store=store)

        # 生命周期: on_startup 连接 auto_connect servers
        await state.on_startup()
        assert "demo" in state.sessions()
        assert state.sessions()["demo"].state == "connected"

        # 子 channel 辐射
        children = state.get_virtual_children()
        assert "demo" in children
        sub_channel = children["demo"]
        assert sub_channel.name() == "demo"

        # warm help 摘要
        help_text = await state.get_help()
        assert "[+] demo" in help_text

        # call (阻塞) 通过公开方法
        obs = await state.call_tool("demo", "add", {"x": 1, "y": 2})
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1

        # list 命令
        list_result = await state.get_own_command("list")()
        assert "+" in list_result
        assert "add" in list_result

        # disconnect 公开方法
        result = await state.disconnect_server("demo")
        assert "disconnected" in result
        assert "demo" not in state.sessions()

        # reconnect 公开方法
        result = await state.connect_server("demo")
        assert "connected" in result
        assert "demo" in state.sessions()

        # 生命周期: on_close 断开所有
        await state.on_close()
        assert state.sessions() == {}
