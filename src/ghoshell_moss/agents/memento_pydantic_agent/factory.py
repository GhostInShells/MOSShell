"""
memento_pydantic_agent factory — build an agent from an agent .py file.

The pipeline:

1. Read source, compile via Compiler with a recording __import__ that captures
   the file's authorization surface.
2. Set up a two-layer Sandbox — init holds the compiled namespace with
   unrestricted builtins; agent shares the namespace but with SANDBOX_BUILTINS
   plus a REPLAY __import__ that idempotently re-serves recorded imports.
3. Inject `get_interface` into the sandbox as the model's on-demand reflection
   verb (pull-mode capability discovery).
4. Build a pydantic-ai Agent whose only tool is `sandbox_exec`.

Magic attrs on the compiled module:
- `__model__`: Anthropic model name. Fallback: ANTHROPIC_MODEL env var.
- `__owner__`: recognized but consumed at the CLI layer, not here.
- `__interfaces__`: instruction-assembly appendix source; also consumed at
  the impl layer (assemble_instruction reads it).

The instruction the model sees is composed on demand by impl (meta + source
+ optional __interfaces__ appendix) — the factory does not build prompt
text, it only shapes the runtime the impl will drive.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

from ghoshell_moss.depends import depend_ghost

depend_ghost()
from anthropic.types.beta import (
    BetaThinkingConfigDisabledParam,
    BetaThinkingConfigEnabledParam,
)
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import DeferredToolRequests, Tool

from ghoshell_moss.agents._imports import recording_builtins, replay_import
from ghoshell_moss.agents._instruction import reflect_element
from ghoshell_moss.agents.capabilities import CAPABILITY_FACTORIES
from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.agents.memento_pydantic_agent.impl import MementoPydanticAgentImpl
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.executor import ExecutionResult
from ghoshell_moss.core.codex.sandbox import SANDBOX_BUILTINS, Sandbox

__all__ = ["factory"]

logger = logging.getLogger("moss.memento_agent")

_CAPABILITIES_MODULE: str = "ghoshell_moss.agents.capabilities"
_THINKING_BUDGET_TOKENS: int = 2048


def factory(
    agent_path: str | Path,
    injections: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> MementoAgent:
    """Build a MementoAgent from an agent .py file.

    :param agent_path: path to the .py file. Read + compiled at factory time.
    :param injections: additional key-value pairs injected into the agent
        sandbox namespace. The capability-injection mechanism (§13.7):
        "imports are authorization" is the positive use — a library module
        exports a stub function; the factory receives the real one and
        injects it here. Default None = no extra capabilities beyond
        get_interface and auto-detected capability imports.
    :param cwd: working directory for capability implementations that need
        filesystem grounding (e.g. look_at). Defaults to the agent .py
        parent directory.
    :raises FileNotFoundError: if the path does not exist.
    :raises RuntimeError: if compilation fails or __model__ / ANTHROPIC_MODEL
        neither is set.
    """
    path = Path(agent_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"agent .py not found: {path}")

    source = path.read_text(encoding="utf-8")
    stem = path.stem.removesuffix(".agent") if path.stem.endswith(".agent") else path.stem
    resolved_cwd = cwd or path.parent

    # Compile-time: recording __import__ captures every module the file loads.
    # That set becomes the exec-time authorization whitelist.
    compile_builtins, recorded_imports = recording_builtins()
    compiler = Compiler(
        source=source,
        modulename=stem,
        filename=str(path),
        local_injections={"__builtins__": compile_builtins},
        compile_soon=True,
    )
    compiled = compiler.compiled

    # Deferred: config stays in the .py (dunders), no delimiter-split of the
    # source — dunders are configuration, and the agent's self-knowledge
    # should be as true as possible. Why: splitting needs new conventions +
    # a new failure surface for a few tokens of savings. Trigger: construct /
    # config grows into a long dict that needs externalization.
    model_name = getattr(compiled, "__model__", None) or os.environ.get("ANTHROPIC_MODEL")
    if not model_name:
        logger.error("model not resolved for %s (no __model__, no ANTHROPIC_MODEL)", path)
        raise RuntimeError(
            f"model not resolved for {path}: set __model__ = '...' in the "
            f"agent .py or export ANTHROPIC_MODEL env var."
        )
    # Thinking default ON — product quality over archival purity: thinking is
    # invisible in the folded read (window uses content only) yet raises
    # reasoning quality; post-compact everything becomes commit summaries, so
    # thinking blocks in payloads need no special handling. Per-agent off via
    # `__thinking__ = False` for cost-sensitive mechanical agents.
    thinking = bool(getattr(compiled, "__thinking__", True))
    logger.info(
        "building agent %s (owner=%s, model=%s, thinking=%s, cwd=%s)",
        path, stem, model_name, thinking, resolved_cwd,
    )

    # Two-layer sandbox: init copies the compiled namespace under safe default
    # builtins; agent shares the same __dict__ but under SANDBOX_BUILTINS with
    # replay __import__ layered on top.
    init_sandbox = Sandbox(name=f"{stem}.init", source=source)
    for k, v in compiled.__dict__.items():
        if not k.startswith("__"):
            init_sandbox.set(k, v)

    # Deferred: capture `print` at sandbox creation as supplemental context
    # messages (dynamic-info input channel). Why: v1 injects nothing dynamic;
    # the agent's input is the static composed instruction only. Trigger:
    # when runtime dynamic info must reach the agent mid-run.
    agent_sandbox = Sandbox(
        name=stem,
        parent=init_sandbox,
        builtins={
            **SANDBOX_BUILTINS,
            "__import__": replay_import(frozenset(recorded_imports)),
        },
        source=source,
    )
    # get_interface lives in the sandbox as the model's pull-mode reflection.
    # Named identically to `moss codex get-interface` on purpose — same verb,
    # same output shape, whichever seat you are in.
    agent_sandbox.set("get_interface", reflect_element)
    for key, value in (injections or {}).items():
        agent_sandbox.set(key, value)

    # Auto-detect capability imports: when the agent .py imports a name from
    # the capabilities stub module, inject the real implementation bound to
    # the working directory. The stub in capabilities.py never executes in
    # the sandbox — it is overridden here.
    injected = []
    for name, obj in compiled.__dict__.items():
        if name.startswith("_"):
            continue
        if not callable(obj):
            continue
        mod = getattr(obj, "__module__", None)
        if mod != _CAPABILITIES_MODULE:
            continue
        make_impl = CAPABILITY_FACTORIES.get(name)
        if make_impl is None:
            continue
        agent_sandbox.set(name, make_impl(resolved_cwd))
        injected.append(name)

    if injected:
        logger.info("capability injection: %s", ", ".join(sorted(injected)))

    # Freeze the authorized surface (compiled objects + tools + injections)
    # as the base. Each aexec() call copies from this snapshot, so the model's
    # code runs hermetic — no variable accumulates across calls.
    agent_sandbox.snapshot_base()

    async def sandbox_exec(code: str) -> str:
        """Execute Python code in the agent's sandbox.

        The code must define an `async def main(): ...` entry point, which is
        awaited inside the event loop. Each call runs in a fresh namespace —
        no variable carries over between calls. Returns any stdout, exception,
        or the value `main` returns.
        """
        result = await agent_sandbox.aexec(code)
        return _format_result(result)

    thinking_settings = (
        AnthropicModelSettings(
            anthropic_thinking=BetaThinkingConfigEnabledParam(
                type="enabled",
                budget_tokens=_THINKING_BUDGET_TOKENS,
            ),
        )
        if thinking
        else AnthropicModelSettings(
            anthropic_thinking=BetaThinkingConfigDisabledParam(type="disabled"),
        )
    )
    model = AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(),
        settings=thinking_settings,
    )
    description = (compiled.__doc__ or "").strip().splitlines()[0] if compiled.__doc__ else ""
    ai_agent = Agent(
        name=stem,
        description=description,
        model=model,
        tools=[sandbox_exec],
    )
    # dry run 工具面:同一个 sandbox_exec 挂 requires_approval — 模型生成的
    # 工具调用停在 ToolCallPart、不执行(零副作用)。output_type 加
    # DeferredToolRequests 让 graph 把未执行调用作为 result.output 返回。
    dry_run_agent = Agent(
        name=stem,
        description=description,
        model=model,
        output_type=str | DeferredToolRequests,
        tools=[Tool(sandbox_exec, requires_approval=True)],
    )

    return MementoPydanticAgentImpl(
        agent=ai_agent,
        dry_run_agent=dry_run_agent,
        sandbox=agent_sandbox,
        compiled_module=compiled,
        source=source,
        name=stem,
        description=description,
        model_name=model_name,
        thinking=thinking,
        tools_protocol=[_tool_protocol(sandbox_exec)],
    )


def _tool_protocol(tool_fn: Any) -> dict[str, Any]:
    """Extract a tool's protocol declaration — name + signature + description."""
    sig = inspect.signature(tool_fn)
    doc = ""
    if tool_fn.__doc__:
        doc = tool_fn.__doc__.strip().splitlines()[0]
    return {
        "name": tool_fn.__name__,
        "signature": f"{tool_fn.__name__}{sig}",
        "description": doc,
    }


def _format_result(result: ExecutionResult) -> str:
    """Format sandbox.aexec ExecutionResult as agent-facing text."""
    parts: list[str] = []
    if result.std_output:
        parts.append(result.std_output.rstrip())
    if result.exception:
        parts.append(f"Error: {result.exception}")
        if result.traceback:
            parts.append(result.traceback.rstrip())
    if result.returns is not None:
        parts.append(f"returned: {result.returns!r}")
    return "\n".join(parts) if parts else "(executed, no output)"
