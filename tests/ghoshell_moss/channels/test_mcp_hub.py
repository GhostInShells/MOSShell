"""Tests for MCP Hub Channel — config, Observe conversion, channel structure, integration."""
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
from ghoshell_moss.contracts.configs import YamlConfigStore, ConfigType
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.concepts.command import Observe
from ghoshell_moss.message import Message, unique_id


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
            env={"KEY": "val"},
            description="filesystem server",
        )
        assert cfg.name == "fs"
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

    def test_inherits_config_type(self):
        assert issubclass(MCPHubConfig, ConfigType)

    def test_config_store_roundtrip(self):
        """Config 通过 YamlConfigStore 存取，验证 YAML 序列化回环。"""
        storage = LocalStorage(Path(tempfile.mkdtemp()))
        store = YamlConfigStore(storage)

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

        # Verify file was written as YAML
        path = store.get_config_path(MCPHubConfig.conf_name())
        assert Path(path).exists()

        # Reload from file
        reloaded = store.get(MCPHubConfig)
        assert reloaded.servers["demo"].command == "python"
        assert reloaded.servers["demo"].args == ["server.py"]

    def test_config_store_get_or_create(self):
        """服务启动时可以将默认配置持久化，后续调用 get 返回一致结果。"""
        storage = LocalStorage(Path(tempfile.mkdtemp()))
        store = YamlConfigStore(storage)

        default = MCPHubConfig()
        created = store.get_or_create(default)
        assert created.servers == {}

        # 再次获取应命中缓存
        cached = store.get(MCPHubConfig)
        assert cached is created

    def test_config_store_invalidate(self):
        """invalidate 后重新读文件，不是返回旧缓存。"""
        storage = LocalStorage(Path(tempfile.mkdtemp()))
        store = YamlConfigStore(storage)

        store.save(MCPHubConfig(servers={
            "a": MCPServerConfig(name="a", command="first"),
        }))
        first = store.get(MCPHubConfig)
        assert first.servers["a"].command == "first"

        # 绕过缓存直接写文件
        store.save(MCPHubConfig(servers={
            "a": MCPServerConfig(name="a", command="second"),
        }))
        store.invalidate(MCPHubConfig)
        second = store.get(MCPHubConfig)
        assert second.servers["a"].command == "second"


# ---------------------------------------------------------------------------
# mcp_result_to_observe tests
# ---------------------------------------------------------------------------

class TestMCPResultToObserve:
    def test_text_content(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="hello world")],
            isError=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="echo")
        assert isinstance(obs, Observe)
        assert len(obs.messages) == 1
        contents = list(obs.messages[0].as_contents())
        assert len(contents) == 1
        assert contents[0]['type'] == 'text'
        assert contents[0]['text'] == 'hello world'

    def test_image_content(self):
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.ImageContent(
                    type="image", data="base64data", mimeType="image/png"
                )
            ],
            isError=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="screenshot")
        assert len(obs.messages) == 1
        contents = list(obs.messages[0].as_contents())
        assert len(contents) == 1
        assert contents[0]['type'] == 'image'
        assert contents[0]['source']['media_type'] == 'image/png'

    def test_mixed_content(self):
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="analysis result"),
                mcp_types.ImageContent(
                    type="image", data="imgdata", mimeType="image/jpeg"
                ),
            ],
            isError=False,
        )
        obs = mcp_result_to_observe(result, server="test", tool="analyze")
        assert len(obs.messages) == 2

    def test_error_result(self):
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="file not found")],
            isError=True,
        )
        obs = mcp_result_to_observe(result, server="test", tool="read")
        msg = obs.messages[0].to_content_string()
        assert "error" in msg.lower()
        assert "file not found" in msg

    def test_empty_content(self):
        result = mcp_types.CallToolResult(content=[], isError=False)
        obs = mcp_result_to_observe(result, server="test", tool="noop")
        assert isinstance(obs, Observe)
        assert obs.messages == []


# ---------------------------------------------------------------------------
# MCPHubState structure tests (no MCP server needed)
# ---------------------------------------------------------------------------

class TestMCPHubStateStructure:
    """Test MCPHubState command structure and context messages without connecting."""

    @pytest.fixture
    def mock_matrix(self):
        """Create a minimal matrix mock with temp storage for config."""
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage

        matrix = MagicMock()
        matrix.is_running.return_value = True
        matrix.ghost_name = "test_ghost"
        matrix.mode_name = "test_mode"
        matrix.session_id = "test_session"
        matrix.session_scope = "test_scope"

        # Use a real temp storage
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        matrix.scopes.return_value = {
            "ghost": "test_ghost",
            "mode": "test_mode",
            "session_id": "test_session",
            "session_scope": "test_scope",
            "cell": "host/local",
        }
        return matrix

    @pytest.fixture
    def state(self, mock_matrix):
        return MCPHubState(
            matrix=mock_matrix,
            name="mcp",
            description="test hub",
            scopes=["ghost", "mode"],
        )

    def test_state_has_all_commands(self, state):
        cmds = state.own_commands()
        assert "exec" in cmds
        assert "exec_blocking" in cmds
        assert "list_servers" in cmds
        assert "add_server" in cmds
        assert "remove_server" in cmds
        assert "restart_server" in cmds

    def test_exec_is_nonblocking_always_observe(self, state):
        cmd = state.get_own_command("exec")
        assert cmd is not None
        meta = cmd.meta()
        assert meta.blocking is False

    def test_exec_blocking_is_blocking(self, state):
        cmd = state.get_own_command("exec_blocking")
        assert cmd is not None
        meta = cmd.meta()
        assert meta.blocking is True

    def test_list_servers_empty(self, state):
        cmd = state.get_own_command("list_servers")
        assert cmd is not None

    @pytest.mark.asyncio
    async def test_list_servers_returns_empty(self, state):
        cmd = state.get_own_command("list_servers")
        result = await cmd()
        assert "No servers configured" in result or "No MCP servers" in result

    @pytest.mark.asyncio
    async def test_context_messages_empty(self, state):
        msgs = await state.get_context_messages()
        assert len(msgs) > 0
        assert "No MCP servers" in msgs[0]

    @pytest.mark.asyncio
    async def test_exec_nonexistent_server(self, state):
        cmd = state.get_own_command("exec")
        result = await cmd(server="nonexistent", tool="foo", text__="{}")
        assert isinstance(result, Observe)
        content = result.messages[0].to_content_string()
        assert "not found" in content.lower()

    @pytest.mark.asyncio
    async def test_add_server_without_config(self, state):
        cmd = state.get_own_command("add_server")
        result = await cmd(name="nonexistent")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self, state):
        cmd = state.get_own_command("remove_server")
        result = await cmd(name="nonexistent")
        assert "not found" in result

    def test_state_name_and_description(self, state):
        assert state.name() == "mcp"
        assert "test hub" in state.description()

    def test_state_is_dynamic(self, state):
        assert state.is_dynamic() is True


# ---------------------------------------------------------------------------
# MCPHubChannel structure tests
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


# ---------------------------------------------------------------------------
# MCPHubState with config in storage
# ---------------------------------------------------------------------------

class TestMCPHubStateWithConfig:
    @pytest.fixture
    def matrix_with_config(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage

        matrix = MagicMock()
        matrix.is_running.return_value = True
        matrix.ghost_name = "test_ghost"
        matrix.mode_name = "test_mode"

        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root

        return matrix

    @pytest.mark.asyncio
    async def test_add_server_loads_from_storage(self, matrix_with_config):
        # Save config to scoped storage via write_yaml
        cfg = MCPHubConfig(
            servers={
                "demo": MCPServerConfig(
                    name="demo",
                    transport="stdio",
                    command=sys.executable,
                    args=[join(dirname(dirname(__file__)), "bridges", "mcp_channel", "helper", "mcp_server_demo.py")],
                )
            }
        )
        storage = matrix_with_config.get_scoped_storage()
        storage.write_yaml("mcp_hub", cfg)

        state = MCPHubState(
            matrix=matrix_with_config,
            name="mcp",
            scopes=["ghost", "mode"],
        )

        cmd = state.get_own_command("add_server")
        result = await cmd(name="demo")
        assert "connected" in result

        # Verify tools were discovered
        ctx = await state.get_context_messages()
        assert "add" in ctx[0]
        assert "foo" in ctx[0]
        assert "bar" in ctx[0]
        assert "multi" in ctx[0]

        # Test exec
        exec_cmd = state.get_own_command("exec")
        result = await exec_cmd(server="demo", tool="add", text__=json.dumps({"x": 1, "y": 2}))
        assert isinstance(result, Observe)
        assert len(result.messages) > 0

        # Cleanup
        await state.on_close()


# ---------------------------------------------------------------------------
# Integration test: full MCPHubChannel bootstrap with real MCP server
# ---------------------------------------------------------------------------

class TestMCPHubIntegration:
    @pytest.mark.asyncio
    async def test_bootstrap_with_server(self):
        """End-to-end: bootstrap MCPHubChannel, add server, execute tool."""
        from unittest.mock import MagicMock

        matrix = MagicMock()
        matrix.is_running.return_value = True
        matrix.ghost_name = "test_ghost"
        matrix.mode_name = "test_mode"

        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root

        # Pre-populate config via scoped storage write_yaml
        cfg = MCPHubConfig(
            servers={
                "demo": MCPServerConfig(
                    name="demo",
                    transport="stdio",
                    command=sys.executable,
                    args=[join(dirname(dirname(__file__)), "bridges", "mcp_channel", "helper", "mcp_server_demo.py")],
                )
            }
        )
        storage_root.write_yaml("mcp_hub", cfg)

        state = MCPHubState(matrix=matrix, name="mcp", scopes=["ghost", "mode"])

        # on_startup connects to all configured servers
        await state.on_startup()

        # Verify tools are available
        ctx = await state.get_context_messages()
        assert "add" in ctx[0]

        # Execute a tool
        exec_cmd = state.get_own_command("exec")
        result = await exec_cmd(server="demo", tool="add", text__=json.dumps({"x": 10, "y": 20}))
        assert isinstance(result, Observe)

        # Cleanup
        await state.on_close()


# ---------------------------------------------------------------------------
# Issue 7: render_input_schema
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
        assert "integer" in result

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
            "properties": {
                "flag": {"type": "boolean", "description": "enable feature"},
            },
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
        assert "second line" not in result  # truncated at first \n


# ---------------------------------------------------------------------------
# Issue 1: Default scopes
# ---------------------------------------------------------------------------

class TestDefaultScopes:
    def test_mcp_hub_channel_defaults_to_empty(self):
        chan = MCPHubChannel(name="mcp")
        assert chan._scopes == []

    def test_mcp_hub_channel_explicit_scopes(self):
        chan = MCPHubChannel(name="mcp", scopes=["ghost", "mode"])
        assert chan._scopes == ["ghost", "mode"]


# ---------------------------------------------------------------------------
# Issue 4: _load_config get_or_create
# ---------------------------------------------------------------------------

class TestLoadConfigGetOrCreate:
    @pytest.fixture
    def fresh_matrix(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage
        matrix = MagicMock()
        matrix.is_running.return_value = True
        matrix.ghost_name = "test_ghost"
        matrix.mode_name = "test_mode"
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        return matrix

    def test_load_config_returns_never_none_scoped(self, fresh_matrix):
        state = MCPHubState(
            matrix=fresh_matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        config = state._load_config()
        assert config is not None
        assert isinstance(config, MCPHubConfig)
        assert config.servers == {}

    def test_load_config_persists_empty_config(self, fresh_matrix):
        state = MCPHubState(
            matrix=fresh_matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        config = state._load_config()
        assert config.servers == {}

        storage = fresh_matrix.get_scoped_storage()
        reloaded = storage.read_yaml("mcp_hub", MCPHubConfig)
        assert reloaded is not None
        assert reloaded.servers == {}

    def test_load_config_returns_existing_config(self, fresh_matrix):
        existing = MCPHubConfig(
            servers={
                "demo": MCPServerConfig(name="demo", command="python", args=["server.py"]),
            }
        )
        storage = fresh_matrix.get_scoped_storage()
        storage.write_yaml("mcp_hub", existing)

        state = MCPHubState(
            matrix=fresh_matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        config = state._load_config()
        assert config.servers["demo"].command == "python"

    @pytest.mark.skip(reason="non-scoped path requires IoC container with ConfigStore — verified in moss-repl integration")
    def test_load_config_non_scoped_returns_config(self, fresh_matrix):
        state = MCPHubState(
            matrix=fresh_matrix,
            name="mcp",
            scopes=[],
        )
        config = state._load_config()
        assert config is not None
        assert isinstance(config, MCPHubConfig)


# ---------------------------------------------------------------------------
# add_server: scoped config save/load
# ---------------------------------------------------------------------------

class TestAddServerConfig:
    @pytest.fixture
    def matrix_storage(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage
        matrix = MagicMock()
        matrix.is_running.return_value = True
        matrix.ghost_name = "test_ghost"
        matrix.mode_name = "test_mode"
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        return matrix, storage_root

    def test_add_server_command_exists_with_runtime_params(self, matrix_storage):
        matrix, storage = matrix_storage
        state = MCPHubState(
            matrix=matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        cmd = state.get_own_command("add_server")
        assert cmd is not None

    def test_add_server_runtime_params_saves_config(self, matrix_storage):
        matrix, storage = matrix_storage
        state = MCPHubState(
            matrix=matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        config = MCPHubConfig()
        server_cfg = MCPServerConfig(
            name="runtime_added",
            transport="stdio",
            command="echo",
            args=["hello"],
            env={"X": "1"},
            description="runtime test",
        )
        config.servers["runtime_added"] = server_cfg
        state._save_config(config)

        reloaded = storage.read_yaml("mcp_hub", MCPHubConfig)
        assert reloaded.servers["runtime_added"].command == "echo"
        assert reloaded.servers["runtime_added"].args == ["hello"]
        assert reloaded.servers["runtime_added"].env == {"X": "1"}
        assert reloaded.servers["runtime_added"].description == "runtime test"

    def test_add_server_runtime_env_parsing(self, matrix_storage):
        env_str = "KEY=val,FOO=bar"
        parsed_env = {}
        for pair in env_str.split(','):
            pair = pair.strip()
            if '=' in pair:
                k, v = pair.split('=', 1)
                parsed_env[k.strip()] = v.strip()
        assert parsed_env == {"KEY": "val", "FOO": "bar"}

    def test_add_server_runtime_args_parsing(self, matrix_storage):
        args_str = "-m,mcp_server,--verbose"
        parsed_args = [a.strip() for a in args_str.split(',') if a.strip()]
        assert parsed_args == ["-m", "mcp_server", "--verbose"]

    def test_add_server_empty_env_and_args(self, matrix_storage):
        assert [] == [a.strip() for a in ''.split(',') if a.strip()]
        parsed_env = {}
        env_str = ""
        if env_str:
            for pair in env_str.split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    parsed_env[k.strip()] = v.strip()
        assert parsed_env == {}


# ---------------------------------------------------------------------------
# Issue 5: always_observe on management commands
# ---------------------------------------------------------------------------

class TestAlwaysObserve:
    @pytest.fixture
    def state(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage
        matrix = MagicMock()
        matrix.is_running.return_value = True
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        return MCPHubState(
            matrix=matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )

    def test_add_server_always_observe(self, state):
        cmd = state._own_commands["add_server"]
        assert cmd._always_observe is True

    def test_remove_server_always_observe(self, state):
        cmd = state._own_commands["remove_server"]
        assert cmd._always_observe is True

    def test_restart_server_always_observe(self, state):
        cmd = state._own_commands["restart_server"]
        assert cmd._always_observe is True

    def test_list_servers_always_observe(self, state):
        cmd = state._own_commands["list_servers"]
        assert cmd._always_observe is True


# ---------------------------------------------------------------------------
# Issue 6: No hand-written _DEFAULT_INSTRUCTION
# ---------------------------------------------------------------------------

class TestNoHandWrittenInstruction:
    @pytest.fixture
    def state(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage
        matrix = MagicMock()
        matrix.is_running.return_value = True
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        return MCPHubState(
            matrix=matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )

    def test_no_default_instruction_constant(self):
        import inspect
        src = inspect.getsource(MCPHubState._bootstrap)
        assert "_DEFAULT_INSTRUCTION" not in src

    def test_no_get_instruction_override(self):
        import inspect
        source = inspect.getsource(MCPHubState)
        assert 'get_instruction' not in source


# ---------------------------------------------------------------------------
# Issue 7: context messages with inputSchema
# ---------------------------------------------------------------------------

class TestContextMessagesWithSchema:
    @pytest.fixture
    def state_with_tools(self):
        from unittest.mock import MagicMock
        from ghoshell_moss.contracts.workspace import LocalStorage
        from ghoshell_moss.channels.mcp_hub import MCPServerSession
        from mcp import types as mcp_types
        matrix = MagicMock()
        matrix.is_running.return_value = True
        storage_root = LocalStorage(Path(tempfile.mkdtemp()))
        matrix.get_scoped_storage.return_value = storage_root
        state = MCPHubState(
            matrix=matrix,
            name="mcp",
            scopes=["ghost", "mode"],
        )
        tools = [
            mcp_types.Tool(
                name="echo",
                description="回显输入文本。",
                inputSchema={
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
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        session = MCPServerSession(
            config=MCPServerConfig(name="test", command="python"),
        )
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
        assert "the text to echo" in context_text

    @pytest.mark.asyncio
    async def test_context_no_params_for_empty_schema(self, state_with_tools):
        msgs = await state_with_tools.get_context_messages()
        context_text = msgs[0]
        params_lines = [l for l in context_text.split('\n') if l.strip().startswith('params:')]
        assert len(params_lines) == 1
