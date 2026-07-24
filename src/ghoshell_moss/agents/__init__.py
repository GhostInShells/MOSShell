"""
ghoshell_moss.agents — agent 家族容器.

一级家族契约 (`MementoAgent`) 在 contract.py; 具体实现家族在子目录. v1 只装
memento_pydantic_agent 一家 (pydantic-ai 底座). 通用 MOSS agent 抽象 (四锚:
factory + AGENT.md + memento + ground) 保留在 FEATURE.md 文档层, 不进代码 —
等第二个 agent 家族出现再讨论提级.

见 .ai_partners/features/workstreams/2026/07/memento-cli-and-agent/FEATURE.md §9.
"""

from ghoshell_moss.agents.contract import MementoAgent

__all__ = ["MementoAgent"]
