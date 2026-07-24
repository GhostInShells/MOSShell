"""
memento_pydantic_agent factory — build an agent from an AGENT.py file.

Reads the .py source, compiles it via Compiler, sets up a two-layer Sandbox
(init with unrestricted builtins holding the compiled namespace + agent
with SANDBOX_BUILTINS blocking __import__), then wires a pydantic-ai Agent
whose only tool is `sandbox_exec` — the model writes Python, the sandbox
runs it.

Magic attrs read from the compiled module:
- `__model__`: Anthropic model name. Fallback: ANTHROPIC_MODEL env var.
- `__owner__`: (recognized but not used in v1 factory — CLI layer handles it)

Reflection: `sandbox.get_interface()` output becomes the system prompt.
The model sees the module docstring + imported types + top-level bindings —
the .py file is its instruction manual.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from anthropic.types.beta import BetaThinkingConfigDisabledParam
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.agents.memento_pydantic_agent.impl import MementoPydanticAgentImpl
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.sandbox import SANDBOX_BUILTINS, Sandbox

__all__ = ["factory"]


def factory(agent_path: str | Path) -> MementoAgent:
    """Build a MementoAgent from an AGENT.py file.

    :param agent_path: path to the .py file. Read + compiled at factory time.
    :raises FileNotFoundError: if the path does not exist.
    :raises RuntimeError: if compilation fails or __model__ / ANTHROPIC_MODEL
        neither is set.
    """
    path = Path(agent_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AGENT.py not found: {path}")

    source = path.read_text(encoding="utf-8")
    stem = path.stem.removesuffix(".agent") if path.stem.endswith(".agent") else path.stem

    # Compile with full builtins (imports + top-level side effects execute here).
    compiler = Compiler(
        source=source,
        modulename=stem,
        filename=str(path),
        compile_soon=True,
    )
    compiled = compiler.compiled

    # Resolve model.
    model_name = getattr(compiled, "__model__", None) or os.environ.get("ANTHROPIC_MODEL")
    if not model_name:
        raise RuntimeError(
            f"model not resolved for {path}: set __model__ = '...' in the "
            f"AGENT.py or export ANTHROPIC_MODEL env var."
        )

    # Two-layer sandbox: init holds compiled namespace with unrestricted builtins;
    # agent shares the namespace but with SANDBOX_BUILTINS (__import__ blocked).
    init_sandbox = Sandbox(name=f"{stem}.init", builtins=None, source=source)
    for k, v in compiled.__dict__.items():
        if not k.startswith("__"):
            init_sandbox.set(k, v)

    agent_sandbox = Sandbox(
        name=stem,
        parent=init_sandbox,
        builtins=SANDBOX_BUILTINS,
        source=source,
    )

    # Sandbox exec tool — the only tool the model gets.
    def sandbox_exec(code: str) -> str:
        """Execute Python code in the agent's sandbox.

        The sandbox namespace holds the agent's declared capabilities
        (e.g. `file_editor`, `ctx`, imported modules). State persists
        across calls — variables you assign remain in later exec calls.

        Returns the exec result as text (stdout, exception, or __result__).
        """
        result = agent_sandbox.exec(code)
        return _format_result(result)

    # Build pydantic-ai Agent.
    model = AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(),
        settings=AnthropicModelSettings(
            anthropic_thinking=BetaThinkingConfigDisabledParam(type="disabled"),
        ),
    )
    description = (compiled.__doc__ or "").strip().splitlines()[0] if compiled.__doc__ else ""
    ai_agent = Agent(
        name=stem,
        description=description,
        model=model,
        tools=[sandbox_exec],
    )

    return MementoPydanticAgentImpl(
        agent=ai_agent,
        sandbox=agent_sandbox,
        name=stem,
        description=description,
    )


def _format_result(result: Any) -> str:
    """Format sandbox.exec ExecutionResult as agent-facing text."""
    parts: list[str] = []
    std_output = getattr(result, "std_output", None)
    if std_output:
        parts.append(std_output.rstrip())
    exception = getattr(result, "exception", None)
    if exception:
        parts.append(f"Error: {exception}")
        tb = getattr(result, "traceback", None)
        if tb:
            parts.append(tb.rstrip())
    returns = getattr(result, "returns", None)
    if returns is not None:
        parts.append(f"__result__: {returns!r}")
    return "\n".join(parts) if parts else "(executed, no output)"
