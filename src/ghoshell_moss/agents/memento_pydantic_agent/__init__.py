"""
memento_pydantic_agent — `MementoAgent` 契约的 pydantic-ai 底座实现家族.

v1 是 Atom prototype 定位: 用 CLI 驱动 + AGENT.md 声明 + memento 索引契约 +
cwd (ground 退化态), 把 memento 边界画完并投入真实使用.

家族契约面在上一级 (`ghoshell_moss.agents.contract:MementoAgent`); 本目录只
承载家族内的具体实现 (config / factory / impl).

AGENT.md 通过 `memento_agent: ghoshell_moss.agents.memento_pydantic_agent:factory`
指向工厂, `construct` 字段作为 `MementoPydanticAgentConfig` 的 sink.

关键纪律 (FEATURE.md §9.2):
- agent 全权管写 (invoke 内部自调 line.record / line.commit)
- invoke ≠ commit 生命周期, staging 残留在 invoke 边界上合法
- runner 不摸 line 写侧
"""

from ghoshell_moss.agents.memento_pydantic_agent.config import MementoPydanticAgentConfig
from ghoshell_moss.agents.memento_pydantic_agent.factory import factory
from ghoshell_moss.agents.memento_pydantic_agent.impl import MementoPydanticAgentImpl

__all__ = ["factory", "MementoPydanticAgentConfig", "MementoPydanticAgentImpl"]
