"""
memento_pydantic_agent impl — MementoAgent 的 pydantic-ai 具体实现.

v1 hello-world POC (步 5-6): invoke 调 pydantic-ai `Agent.run` 返回 output.
memento / line_name / cwd 参数接受但不消费 — 会在步 6+ 逐步接入.

三个 memento 相关方法 (compact / export_context_md / describe_line) 目前
raise NotImplementedError. 它们的实现挂在后续步骤 (步 7 是 compact 关键节点).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.agents.memento_pydantic_agent.config import MementoPydanticAgentConfig
from ghoshell_moss.memento.abc import Memento

__all__ = ["MementoPydanticAgentImpl"]


class MementoPydanticAgentImpl(MementoAgent):
    """Hello-world POC 实现. 步 5-6 起点."""

    def __init__(
        self,
        *,
        agent: Agent,
        config: MementoPydanticAgentConfig,
        name: str,
        description: str,
    ):
        self._agent = agent
        self._config = config
        self._name = name
        self._description = description

    # ── MementoAgent contract ──

    async def invoke(
        self,
        *,
        instruction: str,
        prompt: str,
        memento: Memento,
        line_name: str,
        cwd: Path,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """POC: prompt 作为 per-run instructions, instruction 作为 user message.

        memento / line_name / cwd 已接受但不消费 — 步 6+ 才接入 record/commit.
        """
        run_kwargs: dict[str, Any] = {}
        if prompt:
            run_kwargs["instructions"] = prompt
        try:
            result = await self._agent.run(instruction, **run_kwargs)
        except Exception as e:
            raise RuntimeError(f"agent {self._name!r} invoke failed: {e}") from e
        return str(result.output)

    def compact(self, memento: Memento, line_name: str) -> None:
        raise NotImplementedError("compact lands in step 7 — the key node")

    def export_context_md(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("export_context_md lands in step 6+")

    def describe_line(self, memento: Memento, line_name: str) -> str:
        raise NotImplementedError("describe_line lands in step 6+")
