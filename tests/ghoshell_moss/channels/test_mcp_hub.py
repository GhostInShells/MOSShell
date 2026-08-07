"""Tests for MCP Hub Channel — ConfigStore-driven, no mock Matrix."""
import json
import sys
import tempfile
from os.path import dirname, join
from pathlib import Path

import pytest
from mcp import types as mcp_types

from ghoshell_moss.channels.mcp_hub import (
    MCPServerConfig,
    MCPHubConfig,
    MCPHubChannel,
    MCPHubState,
    build_mcp_hub_channel,
    mcp_result_to_observe,
    render_input_schema,
)
from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.concepts.command import Observe


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _new_config_store():
    return YamlConfigStore(LocalStorage(Path(tempfile.mkdtemp())))


def _state(*, allow_model_config=False):
    return MCPHubState(
        config_store=_new_config_store(),
        allow_model_config=allow_model_config,
        name="mcp",
        description="test hub",
    )


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    def test_defaults(self):
        cfg = MCPServerConfig(name="test")
        assert cfg.transport == "stdio"
        assert cfg.command == ""
        assert cfg.args == []
        assert cfg.url == ""

    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="fs",
            transport="stdio",
            command="python",
            args=["-m", "mcp_server"],
            env={"KEY": "$VAL"},
            description="filesystem server",
        )
        data = cfg.model_dump()
        reloaded = MCPServerConfig(**data)
        assert reloaded.command == "python"

    def test_sse_config(self):
        cfg = MCPServerConfig(
            name="remote",
            transport="sse",
            url="http://localhost:8080/sse",
            headers={"Authorization": "Bearer token"},
        )
        assert cfg.transport == "sse"
        assert cfg.url == "http://localhost:8080/sse"


class TestMCPHubConfig:
    def test_empty(self):
        cfg = MCPHubConfig()
        assert cfg.servers == {}

    def test_conf_name(self):
        assert MCPHubConfig.conf_name() == "mcp_hub"

    def test_config_store_roundtrip(self):
        store = _new_config_store()
        cfg = MCPHubConfig(
            servers={
                "demo": MCPServerConfig(
                    name="demo",
                    transport="stdio",
                    command="python",
                    args=["server.py"],
                )
            }
        )
        store.save(cfg)
        reloaded = store.get(MCPHubConfig)
        assert reloaded.servers["demo"].command == "python"
        assert reloaded.servers["demo"].args == ["server.py"]

    def test_get_or_create(self):
        store = _new_config_store()
        default = MCPHubConfig()
        created = store.get_or_create(default)
        assert created.servers == {}
        cached = store.get(MCPHubConfig)
        assert cached is created

    def test_invalidate(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={"a": MCPServerConfig(name="a", command="first")}))
        first = store.get(MCPHubConfig)
        assert first.servers["a"].command == "first"

        store.save(MCPHubConfig(servers={"a": MCPServerConfig(name="a", command="second")}))
        store.invalidate(MCPHubConfig)
        second = store.get(MCPHubConfig)
        assert second.servers["a"].command == "second"


# ---------------------------------------------------------------------------
# mcp_result_to_observe
# ---------------------------------------------------------------------------

class TestMCPResultToObserve:
    def test_text_content(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="hello world")],
            is_error=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="echo")
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1
        contents = list(obs.messages[0].as_contents())
        assert contents[0]['type'] == 'text'
        assert contents[0]['text'] == 'hello world'

    def test_image_content(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.ImageContent(type="image", data="base64data", mime_type="image/png")],
            is_error=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="screenshot")
        contents = list(obs.messages[0].as_contents())
        assert contents[0]['type'] == 'image'
        assert contents[0]['source']['media_type'] == 'image/png'

    def test_mixed_content(self):
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="analysis"),
                mcp_types.ImageContent(type="image", data="imgdata", mime_type="image/jpeg"),
            ],
            is_error=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="analyze")
        assert len(obs.messages) == 2

    def test_error_result(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="file not found")],
            is_error=True,
        )
        obs = mcp_result_to_observe(result, server="test", tool="read")
        msg = obs.messages[0].to_content_string()
        assert "error" in msg.lower()
        assert "file not found" in msg

    def test_empty_content(self):
        result = mcp_types.CallToolResult(content=[], is_error=False)
        obs = mcp_result_to_observe(result, server="test", tool="noop")
        assert isinstance(obs, Observe)
        assert obs.messages == []


# ---------------------------------------------------------------------------
# render_input_schema
# ---------------------------------------------------------------------------

class TestRenderInputSchema:
    def test_basic_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "input text"},
                "count": {"type": "integer"},
            },
            "required": ["text"],
        }
        result = render_input_schema(schema)
        assert "`text`" in result
        assert "string" in result
        assert "required" in result
        assert "input text" in result
        assert "`count`" in result

    def test_non_object_schema(self):
        assert render_input_schema({"type": "array"}) == ""

    def test_empty_schema(self):
        assert render_input_schema({}) == ""

    def test_none_schema(self):
        assert render_input_schema(None) == ""

    def test_no_properties(self):
        assert render_input_schema({"type": "object"}) == ""

    def test_all_optional(self):
        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean", "description": "enable feature"}},
        }
        result = render_input_schema(schema)
        assert "required" not in result
        assert "boolean" in result

    def test_multi_line_description(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "first line\nsecond line\nthird line"},
            },
        }
        result = render_input_schema(schema)
        assert "first line" in result
        assert "second line" not in result


# ---------------------------------------------------------------------------
# MCPHubState structure (no MCP server needed)
# ---------------------------------------------------------------------------

class TestMCPHubStateStructure:

    def test_command_names_without_model_config(self):
        state = _state(allow_model_config=False)
        cmds = state.own_commands()
        assert set(cmds.keys()) == {'exec', 'exec_blocking', 'list', 'connect', 'disconnect', 'reconnect'}

    def test_command_names_with_model_config(self):
        state = _state(allow_model_config=True)
        cmds = state.own_commands()
        assert set(cmds.keys()) == {'exec', 'exec_blocking', 'list', 'connect', 'disconnect', 'reconnect', 'register', 'unregister'}

    def test_exec_is_nonblocking(self):
        cmd = _state().get_own_command("exec")
        assert cmd.meta().blocking is False

    def test_exec_blocking_is_blocking(self):
        cmd = _state().get_own_command("exec_blocking")
        assert cmd.meta().blocking is True

    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await _state().get_own_command("list")()
        assert "No servers configured" in result

    @pytest.mark.asyncio
    async def test_context_messages_empty(self):
        msgs = await _state().get_context_messages()
        assert len(msgs) > 0
        assert "No MCP servers" in msgs[0]

    @pytest.mark.asyncio
    async def test_exec_nonexistent_server(self):
        cmd = _state().get_own_command("exec")
        result = await cmd(server="nonexistent", tool="foo", text__="{}")
        content = result.messages[0].to_content_string()
        assert "not connected" in content.lower()

    @pytest.mark.asyncio
    async def test_connect_nonexistent(self):
        cmd = _state().get_own_command("connect")
        result = await cmd(name="nonexistent")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self):
        cmd = _state().get_own_command("disconnect")
        result = await cmd(name="nonexistent")
        assert "not connected" in result

    def test_name_and_description(self):
        state = _state()
        assert state.name() == "mcp"
        assert "test hub" in state.description()

    def test_is_dynamic(self):
        assert _state().is_dynamic() is True


# ---------------------------------------------------------------------------
# MCPHubChannel structure
# ---------------------------------------------------------------------------

class TestMCPHubChannel:
    def test_channel_defaults(self):
        chan = MCPHubChannel()
        assert chan.name() == "mcp"
        assert "MCP Hub" in chan.description()
        assert chan.id()

    def test_channel_custom_name(self):
        chan = MCPHubChannel(name="tools", description="external tools")
        assert chan.name() == "tools"
        assert chan.description() == "external tools"

    def test_default_scopes_empty(self):
        chan = MCPHubChannel(name="mcp")
        assert chan._scopes == []

    def test_explicit_scopes(self):
        chan = MCPHubChannel(name="mcp", scopes=["ghost", "mode"])
        assert chan._scopes == ["ghost", "mode"]

    def test_allow_model_config_default(self):
        chan = MCPHubChannel(name="mcp")
        assert chan._allow_model_config is False

    def test_allow_model_config_true(self):
        chan = MCPHubChannel(name="mcp", allow_model_config=True)
        assert chan._allow_model_config is True


# ---------------------------------------------------------------------------
# _load_config via ConfigStore
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_never_none(self):
        state = _state()
        config = state._load_config()
        assert config is not None
        assert isinstance(config, MCPHubConfig)
        assert config.servers == {}

    def test_persists_empty_config(self):
        store = _new_config_store()
        state = MCPHubState(config_store=store)
        config = state._load_config()
        assert config.servers == {}
        reloaded = store.get(MCPHubConfig)
        assert reloaded is not None
        assert reloaded.servers == {}

    def test_returns_existing_config(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={"demo": MCPServerConfig(name="demo", command="python", args=["server.py"])}))
        state = MCPHubState(config_store=store)
        config = state._load_config()
        assert config.servers["demo"].command == "python"


# ---------------------------------------------------------------------------
# register / unregister (allow_model_config=True)
# ---------------------------------------------------------------------------

class TestRegisterUnregister:
    def test_register_not_available_without_flag(self):
        state = _state(allow_model_config=False)
        assert "register" not in state.own_commands()

    def test_register_available_with_flag(self):
        state = _state(allow_model_config=True)
        assert "register" in state.own_commands()

    @pytest.mark.asyncio
    async def test_register_persists_to_config_store(self):
        store = _new_config_store()
        state = MCPHubState(config_store=store, allow_model_config=True)

        cfg = MCPServerConfig(
            name="runtime_added",
            transport="stdio",
            command="echo",
            args=["hello"],
            env={"X": "1"},
            description="runtime test",
        )
        data = cfg.model_dump()
        data["connect"] = False
        cmd = state.get_own_command("register")
        result = await cmd(text__=json.dumps(data))
        assert "registered" in result

        reloaded = store.get(MCPHubConfig)
        assert reloaded.servers["runtime_added"].command == "echo"
        assert reloaded.servers["runtime_added"].args == ["hello"]
        assert reloaded.servers["runtime_added"].env == {"X": "1"}
        assert reloaded.servers["runtime_added"].description == "runtime test"

    @pytest.mark.asyncio
    async def test_register_invalid_json(self):
        cmd = _state(allow_model_config=True).get_own_command("register")
        result = await cmd(text__="not json")
        assert "invalid JSON" in result

    @pytest.mark.asyncio
    async def test_register_invalid_config(self):
        cmd = _state(allow_model_config=True).get_own_command("register")
        result = await cmd(text__='{}')
        # missing required 'name' field — should be caught
        assert "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_unregister_removes_from_store(self):
        store = _new_config_store()
        store.save(MCPHubConfig(servers={
            "demo": MCPServerConfig(name="demo", command="echo"),
        }))
        state = MCPHubState(config_store=store, allow_model_config=True)

        cmd = state.get_own_command("unregister")
        result = await cmd(name="demo")
        assert "unregistered" in result
        assert "demo" not in store.get(MCPHubConfig).servers

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self):
        cmd = _state(allow_model_config=True).get_own_command("unregister")
        result = await cmd(name="nonexistent")
        assert "not found" in result


# ---------------------------------------------------------------------------
# always_observe on all commands
# ---------------------------------------------------------------------------

class TestAlwaysObserve:
    def test_all_always_observe(self):
        state = _state()
        for name in state.own_commands():
            cmd = state._own_commands[name]
            assert cmd._always_observe is True, f"{name} should be always_observe"


# ---------------------------------------------------------------------------
# No hand-written instruction
# ---------------------------------------------------------------------------

class TestNoHandWrittenInstruction:
    def test_no_default_instruction_constant(self):
        import inspect
        src = inspect.getsource(MCPHubState._bootstrap)
        assert "_DEFAULT_INSTRUCTION" not in src

    def test_no_get_instruction_override(self):
        import inspect
        source = inspect.getsource(MCPHubState)
        assert 'get_instruction' not in source


# ---------------------------------------------------------------------------
# context messages with inputSchema
# ---------------------------------------------------------------------------

class TestContextMessagesWithSchema:
    @pytest.fixture
    def state_with_tools(self):
        from ghoshell_moss.channels.mcp_hub import MCPServerSession

        state = _state()
        tools = [
            mcp_types.Tool(
                name="echo",
                description="回显输入文本。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "the text to echo"},
                    },
                    "required": ["text"],
                },
            ),
            mcp_types.Tool(
                name="noop",
                description="无参数工具。",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
        session = MCPServerSession(config=MCPServerConfig(name="test", command="python"))
        session.tools = tools
        session.state = "connected"
        state._sessions["test"] = session
        return state

    @pytest.mark.asyncio
    async def test_context_contains_schema_params(self, state_with_tools):
        msgs = await state_with_tools.get_context_messages()
        context_text = msgs[0]
        assert "params:" in context_text
        assert "`text`" in context_text
        assert "string" in context_text

    @pytest.mark.asyncio
    async def test_context_no_params_for_empty_schema(self, state_with_tools):
        msgs = await state_with_tools.get_context_messages()
        context_text = msgs[0]
        params_lines = [l for l in context_text.split('\n') if l.strip().startswith('params:')]
        assert len(params_lines) == 1


# ---------------------------------------------------------------------------
# Integration: connect + exec with real MCP server
# ---------------------------------------------------------------------------

_HELPER_PATH = join(dirname(dirname(__file__)), "bridges", "mcp_channel", "helper", "mcp_server_demo.py")


class TestConnectAndExec:
    @pytest.mark.asyncio
    async def test_connect_and_exec(self):
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

        # on_startup connects all configured servers
        await state.on_startup()

        ctx = await state.get_context_messages()
        assert "add" in ctx[0]
        assert "foo" in ctx[0]

        # exec non-blocking
        result = await state.get_own_command("exec")(
            server="demo", tool="add", text__=json.dumps({"x": 1, "y": 2})
        )
        assert isinstance(result, Observe)
        assert len(result.messages) > 0

        # exec_blocking
        result = await state.get_own_command("exec_blocking")(
            server="demo", tool="add", text__=json.dumps({"x": 10, "y": 20})
        )
        assert isinstance(result, Observe)

        # list
        list_result = await state.get_own_command("list")()
        assert "+" in list_result  # connected
        assert "add" in list_result

        # disconnect
        await state.get_own_command("disconnect")(name="demo")
        assert "demo" not in state._sessions

        # reconnect
        await state.get_own_command("connect")(name="demo")
        assert "demo" in state._sessions

        await state.on_close()
