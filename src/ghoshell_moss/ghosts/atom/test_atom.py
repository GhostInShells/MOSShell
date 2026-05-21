"""Atom Ghost 原型测试.

只测真实数据路径和算法正确性，不 mock pydantic AI 内部行为.
"""

import os
from pathlib import Path

import pytest
from ghoshell_container import Container

from ghoshell_moss.core.blueprint.mindflow import Moment, Reaction
from ghoshell_moss.message import Message
from ghoshell_moss.contracts.system_prompter import SystemPrompter, BaseSystemPrompter
from ghoshell_moss.contracts.workspace import Workspace, LocalWorkspace
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger


# ── helpers ─────────────────────────────────────────


def _atom_meta(**kwargs):
    from ._meta import AtomMeta
    defaults = dict(name="test_atom", soul_content="you are a helpful assistant.")
    return AtomMeta(**{**defaults, **kwargs})


def _container(*bindings):
    c = Container()
    for contract, instance in bindings:
        c.set(contract, instance)
    return c


# ── soul 加载 ───────────────────────────────────────


class TestSoul:
    def test_soul_content_direct(self):
        meta = _atom_meta(soul_content="direct")
        assert meta.soul_content == "direct"

    def test_soul_content_none_is_empty(self):
        meta = _atom_meta(soul_content=None)
        assert meta.soul_content == ""

    def test_soul_path_resolves_to_name(self, tmp_path: Path):
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        (souls_dir / "test_atom.md").write_text("from file")
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(soul_path=None, soul_content=None)
        meta._load_soul(ws.root_path())
        assert meta.soul_content == "from file"

    def test_soul_path_explicit_str(self, tmp_path: Path):
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        (souls_dir / "custom.md").write_text("custom")
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(soul_path="custom", soul_content=None)
        meta._load_soul(ws.root_path())
        assert meta.soul_content == "custom"

    def test_soul_path_absolute(self, tmp_path: Path):
        file = tmp_path / "abs.md"
        file.write_text("absolute")
        meta = _atom_meta(soul_path=file, soul_content=None)
        meta._load_soul(Path("/irrelevant"))
        assert meta.soul_content == "absolute"

    def test_soul_content_skips_file_load(self, tmp_path: Path):
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        (souls_dir / "test_atom.md").write_text("should not load")
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(soul_content="preset")
        meta._load_soul(ws.root_path())
        assert meta.soul_content == "preset"


# ── build_instruction ───────────────────────────────


class TestBuildInstruction:
    def test_no_system_prompter(self):
        meta = _atom_meta(soul_content="my soul")
        c = Container()
        assert meta._build_instruction(c) == "my soul"

    def test_with_system_prompter(self):
        meta = _atom_meta(soul_content="my soul")
        prompter = BaseSystemPrompter(own_instruction="moss instruction")
        c = _container((SystemPrompter, prompter))
        instruction = meta._build_instruction(c)
        assert "moss instruction" in instruction
        assert "my soul" in instruction

    def test_order_moss_before_soul(self):
        meta = _atom_meta(soul_content="soul")
        prompter = BaseSystemPrompter(own_instruction="moss")
        c = _container((SystemPrompter, prompter))
        lines = meta._build_instruction(c).split("\n")
        assert lines[0] == "moss"
        assert lines[1] == "soul"


# ── build_agent (真依赖) ─────────────────────────────


class TestBuildAgent:
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_MODEL"),
        reason="ANTHROPIC_MODEL env var not set",
    )
    def test_with_env_model(self):
        meta = _atom_meta(model=None, soul_content="test")
        agent = meta.build_agent(Container())
        assert agent._model is not None

    def test_loads_soul_from_workspace(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        souls_dir = tmp_path / "souls"
        souls_dir.mkdir()
        (souls_dir / "test_atom.md").write_text("soul from ws")
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content=None,
        )
        c = _container((Workspace, ws))
        meta.build_agent(c)
        assert meta.soul_content == "soul from ws"

    def test_on_agent_build_called(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        calls = []
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
            on_agent_build=lambda a: calls.append(a),
        )
        c = _container((Workspace, LocalWorkspace(tmp_path)))
        agent = meta.build_agent(c)
        assert len(calls) == 1
        assert calls[0] is agent


# ── factory ─────────────────────────────────────────


class TestFactory:
    def test_returns_atom_with_workspace(self, tmp_path: Path):
        from ._runtime import Atom
        from pydantic_ai.models.anthropic import AnthropicModel
        ws = LocalWorkspace(tmp_path)
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        atom = meta.factory(c)
        assert isinstance(atom, Atom)
        assert atom.meta is meta


# ── 消息协议 ────────────────────────────────────────


class TestAtomMessages:
    def _atom(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        return meta.factory(c)

    def test_to_model_request(self, tmp_path: Path):
        from pydantic_ai.messages import ModelRequest
        atom = self._atom(tmp_path)
        msg = Message.new().with_content("hello")
        moment = Moment(percepts=[msg])
        request = atom.to_model_request(moment)
        assert isinstance(request, ModelRequest)

    def test_history_initially_empty(self, tmp_path: Path):
        atom = self._atom(tmp_path)
        assert atom.model_history() == []

    def test_save_adds_to_history(self, tmp_path: Path):
        from pydantic_ai.messages import ModelRequest, ModelResponse
        atom = self._atom(tmp_path)
        moment = Moment(percepts=[Message.new().with_content("hi")])
        response = ModelResponse(parts=[])
        atom.save_model_request(moment, response)
        history = atom.model_history()
        assert len(history) == 2
        assert isinstance(history[0], ModelRequest)
        assert isinstance(history[1], ModelResponse)


# ── system_prompt ───────────────────────────────────


class TestSystemPrompt:
    def test_returns_build_instruction(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        prompter = BaseSystemPrompter(own_instruction="moss hi")
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="i am atom",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container(
            (Workspace, ws),
            (SystemPrompter, prompter),
            (LoggerItf, get_moss_logger()),
        )
        atom = meta.factory(c)
        prompt = atom.system_prompt()
        assert "moss hi" in prompt
        assert "i am atom" in prompt


# ── 生命周期 ────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_enter_exit(self, tmp_path: Path):
        from pydantic_ai.models.anthropic import AnthropicModel
        meta = _atom_meta(
            model=AnthropicModel(model_name="claude-sonnet-4-6"),
            soul_content="test",
        )
        ws = LocalWorkspace(tmp_path)
        c = _container((Workspace, ws), (LoggerItf, get_moss_logger()))
        atom = meta.factory(c)
        async with atom as ctx:
            assert ctx is atom


# ── Adapter ─────────────────────────────────────────


class TestAdapter:
    def test_messages_to_parts_text(self):
        from ._adapter import messages_to_parts
        from pydantic_ai import TextContent
        msg = Message.new().with_content("hello")
        parts = messages_to_parts([msg])
        assert len(parts) == 1
        assert isinstance(parts[0], TextContent)

    def test_messages_to_parts_multiple(self):
        from ._adapter import messages_to_parts
        msgs = [Message.new().with_content("a"), Message.new().with_content("b")]
        parts = messages_to_parts(msgs)
        assert len(parts) == 2

    def test_moment_to_request(self):
        from ._adapter import moment_to_request
        from pydantic_ai.messages import ModelRequest
        moment = Moment(percepts=[Message.new().with_content("test")])
        request = moment_to_request(moment)
        assert isinstance(request, ModelRequest)
        assert len(request.parts) > 0

    def test_moment_to_request_includes_percepts(self):
        from ._adapter import moment_to_request
        msg = Message.new().with_content("percept content")
        request = moment_to_request(Moment(percepts=[msg]))
        assert any("percept content" in str(p) for p in request.parts)

    def test_moment_to_request_includes_previous_outcomes(self):
        from ._adapter import moment_to_request
        prev = Reaction(moment_id="p1",
                        outcomes=[Message.new().with_content("outcome")])
        request = moment_to_request(Moment(previous=prev))
        assert any("outcome" in str(p) for p in request.parts)

    def test_moment_to_request_empty(self):
        from ._adapter import moment_to_request
        from pydantic_ai.messages import ModelRequest
        request = moment_to_request(Moment())
        assert isinstance(request, ModelRequest)
        assert len(request.parts) == 0
