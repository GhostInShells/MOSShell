"""
memento_pydantic_agent — `MementoAgent` 契约的 pydantic-ai 底座实现家族.

v1: `*.agent.py` 定义身份 + 能力面. Compiler + Sandbox 装载, 反射即 prompt,
pydantic-ai 驱动模型 loop, sandbox_exec 是唯一工具.

家族契约面在 `ghoshell_moss.agents.contract:MementoAgent`; 本目录只承载
家族内的具体实现 (factory / impl).

关键纪律:
- Sandbox 是 tool, 认知归 runner (agent 不自己写 memento)
- v1 无 compact 无 magic hook, staging 累积不 commit
- 反射天然过滤 dunder, 未来加 __*__ hooks 无副作用
"""

from ghoshell_moss.agents.memento_pydantic_agent.factory import factory
from ghoshell_moss.agents.memento_pydantic_agent.impl import MementoPydanticAgentImpl

__all__ = ["factory", "MementoPydanticAgentImpl"]
