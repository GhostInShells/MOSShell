"""Contract tests for the formal agent_func protocol — no dsh involved."""

from __future__ import annotations

import asyncio
import inspect
from types import ModuleType

import pytest

from ghoshell_moss.ghosts.dolores.agent_func import (
    AgentFuncConfig,
    AgentMode,
    agent_context,
    agent_ctx,
    agent_func,
    build_instruction,
    build_module_namespace,
    compile_generated,
    module_title,
    parse_answer,
    render_module_context,
    resolve_session_key,
)


def test_parse_answer_reads_tags_and_ignores_surrounding_fence() -> None:
    text = "```python\n<run>\nasync def __run__() -> str:\n    return 'ok'\n</run>\n```"
    parsed = parse_answer(text)
    assert parsed.run == "async def __run__() -> str:\n    return 'ok'"
    assert parsed.debug is None


def test_parse_answer_reads_debug_and_run() -> None:
    text = (
        "<debug>\nasync def debug(*args, **kwargs) -> str:\n    return 'x'\n</debug>\n"
        "<run>\nasync def __run__() -> str:\n    return 'ok'\n</run>"
    )
    parsed = parse_answer(text)
    assert parsed.debug is not None
    assert "async def debug" in parsed.debug
    assert parsed.run is not None
    assert "async def __run__" in parsed.run


def test_parse_answer_keeps_inner_backticks_inside_the_tag() -> None:
    body = 'async def __run__() -> str:\n    """```"""\n    return "ok"'
    parsed = parse_answer(f"<run>{body}</run>")
    assert parsed.run == body


def test_parse_answer_falls_back_to_a_language_fence() -> None:
    text = "```python run\nasync def __run__() -> str:\n    return 'ok'\n```"
    parsed = parse_answer(text)
    assert parsed.run == "async def __run__() -> str:\n    return 'ok'"


def test_build_instruction_includes_source_protocol_and_context() -> None:
    module = ModuleType("demo_module")
    module.__doc__ = "A demo module."
    text = build_instruction(module, source="x = 1", context="the live state is 7")
    assert "`demo_module`" in text
    assert "x = 1" in text
    assert "<run>" in text
    assert "<debug>" in text
    assert "agent_ctx.emit" in text
    assert "the live state is 7" in text


def test_module_title_prefers_agent_metadata_and_docstring() -> None:
    module = ModuleType("demo_module")
    module.__doc__ = "A readable first line.\n\nMore."
    assert module_title(module).startswith("demo_module · A readable first line.")
    module.__agent__ = {"name": "Demo", "description": "does demo things"}
    assert module_title(module) == "demo_module · Demo — does demo things"


def test_render_module_context_runs_once_and_degrades_on_error() -> None:
    async def _run() -> None:
        namespace = ModuleType("ctx_module")

        async def context() -> str:
            return "live context"

        namespace.__dict__["__context__"] = context
        assert await render_module_context(namespace) == "live context"

        def broken() -> str:
            raise RuntimeError("boom")

        namespace.__dict__["__context__"] = broken
        assert "boom" in await render_module_context(namespace)

    asyncio.run(_run())


def test_agent_context_notes_and_emit() -> None:
    async def _run() -> None:
        async with agent_context(effort="high") as ctx:
            agent_ctx.emit("one")
            agent_ctx.note("key", "two")
            assert ctx.notes == ["one", "key: two"]
            assert ctx.overrides == {"effort": "high"}

    asyncio.run(_run())


def test_agent_func_config_validation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        AgentFuncConfig(session_id="a", session_key="b")
    with pytest.raises(ValueError, match="shared mode"):
        AgentFuncConfig(mode=AgentMode.SHARED)
    with pytest.raises(ValueError, match="ego preset"):
        AgentFuncConfig(agent_preset="ego")


def test_resolve_session_key_modes() -> None:
    per_module = AgentFuncConfig()
    assert resolve_session_key(per_module, "demo", "abc123") == "module:demo:abc123:"
    per_call = AgentFuncConfig(mode=AgentMode.PER_CALL)
    assert resolve_session_key(per_call, "demo", "abc123") != resolve_session_key(
        per_call, "demo", "abc123"
    )
    shared = AgentFuncConfig(mode=AgentMode.SHARED, session_key="team")
    assert resolve_session_key(shared, "demo", "abc") == "shared:team:"
    explicit = AgentFuncConfig(session_id="session-1")
    assert resolve_session_key(explicit, "demo", "abc") == ""


def test_compile_generated_checks_signature_and_async() -> None:
    async def expected(a: int = 1) -> str:
        return ""

    namespace = build_module_namespace(ModuleType("compile_module"))
    code = "async def __run__(a: int = 1) -> str:\n    return str(a)"
    fn = compile_generated(namespace, code, "__run__", inspect.signature(expected))
    assert asyncio.run(fn()) == "1"

    bad = "async def __run__(b: int = 1) -> str:\n    return str(b)"
    with pytest.raises(Exception, match="signature mismatch"):
        compile_generated(namespace, bad, "__run__", inspect.signature(expected))

    sync = "def __run__(a: int = 1) -> str:\n    return str(a)"
    with pytest.raises(Exception, match="not an async def"):
        compile_generated(namespace, sync, "__run__", inspect.signature(expected))


def test_build_module_namespace_injects_ctx() -> None:
    namespace = build_module_namespace(ModuleType("namespace_module"))
    assert namespace.agent_ctx is agent_ctx
    assert namespace.emit == agent_ctx.emit


async def _stub(a: int = 1) -> str:
    return ""


def test_decorator_keeps_signature_and_refuses_sync_and_closures() -> None:
    wrapped = agent_func(_stub)
    assert inspect.iscoroutinefunction(wrapped)
    assert inspect.signature(wrapped) == inspect.signature(_stub)

    def sync_function() -> str:
        return "no"

    with pytest.raises(TypeError, match="async def"):
        agent_func(sync_function)  # type: ignore[arg-type]

    async def nested() -> str:
        return "no"

    with pytest.raises(TypeError, match="module-level"):
        agent_func(nested)
