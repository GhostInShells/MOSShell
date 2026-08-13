"""
MementoAgent contract — the agents/ family-level abstraction.

Layer placement: this ABC is the family-level contract for `agents/`, not
for `memento_pydantic_agent/` in particular. `memento_pydantic_agent` is one
concrete implementation family (pydantic-ai substrate); future families
(anthropic-direct, deepseek, bash-only, …) reuse this ABC and carry their
own factory configs, which are NOT part of this contract.

Filename is contract.py, not abc.py: some IDEs treat `abc.py` as special
and clash on it; contract.py aligns with the project's top-level
`contracts/` naming convention already in use.

Design lineage, distilled:

- **agent = single interaction → final answer**. How many rounds / records
  / commits happen inside is entirely the family's business. Interaction
  turns are not aligned with commit boundaries.
- **v1 has no compact**. Compact / magic hooks / spec are harness organs
  (the "no-harness" abandon trigger). Staging accumulates; humans use
  `moss memento branch commit` when they want a checkpoint. If a v2
  compact policy materializes it will grow back as a family concern first.
- **staging residue at the invoke boundary is legal** — not a crash
  remnant. Runner does not sweep it.
- **`memento=None` is the degraded baseline, not a compromise** — a pure
  in-memory single round with no storage write. The contract absorbs this
  explicitly: families branch on it at the invoke layer (record only when a
  store exists) instead of burying the check inside a helper.

Three v1 methods (tentative — cut if redundant, add if missing, no freeze):

| method              | semantics                                       |
| ------------------- | ----------------------------------------------- |
| `invoke`            | one interaction → final answer. Side effects   |
|                     | (record / whatever) are the family's business. |
| `export_context_md` | current context (system + window + recent)     |
|                     | rendered as markdown. No side effects.         |
| `describe_line`     | agent-perspective line summary. No side        |
|                     | effects.                                        |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ghoshell_moss.anchor import Anchor
from ghoshell_moss.memento.abc import Memento

__all__ = ["MementoAgent", "InvocationRecord"]


class InvocationRecord(BaseModel):
    """Structured result of one memento-agent interaction — dry run's first-class return.

    Mirrors LLMFuncResult's shape (strong BaseModel + usage fidelity + message
    serialization). ``output`` is the final answer; ``tool_calls`` are the
    unexecuted tool calls (the dry-run core); ``messages`` is the full
    trajectory. Run: tool_calls empty, output set. Dry-run paused at a tool
    call: tool_calls set, output empty. ``model_dump()`` is the ``-j`` payload.
    """

    output: str = ""
    content: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    cast: float = 0.0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)


class MementoAgent(ABC):
    """
    Memento-family agent contract. beta1: 3 tentative methods.

    Implementations are constructed by a family factory (see factory.py).
    An agent's .py file declares `memento_agent = <factory>` and optional
    `construct = {...}` factory kwargs; the CLI resolves both.

    The constructed instance is a family-specific concretization; `invoke`
    receives the per-call anchors (user_prompt / memento / line / cwd /
    metadata).
    """

    @abstractmethod
    async def invoke(
        self,
        *,
        user_prompt: str,
        memento: Memento | None = None,
        line_name: str = "",
        cwd: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        One interaction. Returns the final answer text.

        :param user_prompt: the newest user turn — pydantic-ai's `user_prompt`
            in spirit. NOT to be confused with the model's instruction, which
            is composed internally from the agent's source (meta narrative +
            source verbatim + optional __interfaces__ expansion).
        :param memento: memento index + storage. The agent holds the owner
            perspective and writes through `memento.get_line(line_name)` if
            recording is part of its family behaviour. `None` is the degraded
            baseline — a pure in-memory single round with no storage write;
            the family must branch on this at the invoke layer, not inside a
            recording helper.
        :param line_name: target line (branch). Runner chooses which line to
            bind; the agent does not select lines. Required for recording;
            ignored when `memento` is None.
        :param cwd: working directory (the ground degenerate form). Default
            is the .py file's parent; CLI --cwd overrides. Used as the
            default cwd for any tool-like injections (file_editor etc.).
        :param metadata: additional anchors. Families may extend; unknown
            keys are silently ignored.

        :return: final answer text. CLI writes this to stdout.

        Side effects — record / whatever — are family-internal. Observers
        infer commit landings via before/after `line.log()` diffs; some
        flake is tolerated.
        """
        raise NotImplementedError

    @abstractmethod
    def export_context_md(self, memento: Memento, line_name: str) -> str:
        """
        Export the agent-perspective current context as markdown.

        v1 semantics: system prompt (composed instruction) + folded window
        text + recent moments in staging. Format is the family's choice.

        Use cases: human diagnosis / external orchestrator consumption /
        cross-family portability reference. No side effects — never writes
        to memento.
        """
        raise NotImplementedError

    @abstractmethod
    def describe_line(self, memento: Memento, line_name: str) -> str:
        """
        Agent-perspective summary of a line.

        Contrast with `moss memento branch log/window`: those are the
        memento structural view (commit / moment / trailer); this is the
        agent's semantic view (what the agent thinks this line is about).

        No side effects.
        """
        raise NotImplementedError

    @abstractmethod
    async def dry_run(
        self,
        *,
        user_prompt: str,
        memento: Memento | None = None,
        line_name: str = "",
        cwd: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InvocationRecord:
        """
        Pure probe — one real generation, paused at the tool-call position.

        Shares invoke's instruction assembly (parity), but the tool is
        ``requires_approval``: the model's tool calls are NOT executed, only
        returned as ``tool_calls``. No history write, no side effects.

        ``messages`` is the full trajectory (incl. the unexecuted
        ToolCallPart); ``usage`` is token spend. ``output`` stays empty when
        the model paused at a tool call, and carries the final text only when
        the model answered directly without tools.

        :return: structured record — ``model_dump()`` is the ``-j`` payload.
        """
        raise NotImplementedError

    @abstractmethod
    def dump_anchor(
        self,
        *,
        memento: Memento | None = None,
        line_name: str = "",
        name: str = "",
        description: str = "",
    ) -> Anchor:
        """
        Dump the agent's current cognitive conditions as an anchor.

        Freeze instruction (composed, window folded in) + tool protocol +
        model config into a self-explaining anchor. The generic Anchor
        container carries the payload; its ``meta.ref`` declares the payload
        shape (family-specific). No side effects — pure snapshot production.

        :param memento / line_name: feed the window into instruction, same as
            invoke. None / empty = degraded (no window).
        :param name / description: AnchorMeta fields.
        :return: weak Anchor — payload structure resolved via ``meta.ref``.
        """
        raise NotImplementedError
