"""
memento_pydantic_agent 工厂.

`factory(metadata, name, description) -> MementoAgent` 是 AGENT.md 的
`memento_agent` 字段指向的目标. 对齐 atom `_meta.py` 的 build_agent 逻辑:
- model=None → ANTHROPIC_MODEL 环境变量, 缺失即 raise
- provider 走 pydantic-ai 默认 `AnthropicProvider()` (读环境变量)
- 默认禁用 extended thinking (config.disable_thinking=True)
"""

from __future__ import annotations

import os
from typing import Any

from anthropic.types.beta import BetaThinkingConfigDisabledParam
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from ghoshell_moss.agents.contract import MementoAgent
from ghoshell_moss.agents.memento_pydantic_agent.config import MementoPydanticAgentConfig
from ghoshell_moss.agents.memento_pydantic_agent.impl import MementoPydanticAgentImpl

__all__ = ["factory"]


def factory(
    metadata: dict[str, Any] | None = None,
    *,
    name: str = "memento-agent",
    description: str = "",
) -> MementoAgent:
    """构造一个 MementoPydanticAgentImpl.

    :param metadata: AGENT.md `construct` 字段 dict. None 或空 dict 也合法
        (全默认). 未识别 key 静默忽略.
    :param name: agent 身份标签 (来自 AGENT.md frontmatter `name` 字段, 或
        文件 stem).
    :param description: 人类可读描述 (来自 AGENT.md frontmatter `description`
        字段).
    """
    config = MementoPydanticAgentConfig.model_validate(metadata or {})

    model_name = config.model or os.environ.get("ANTHROPIC_MODEL")
    if not model_name:
        raise RuntimeError(
            "model not set: pass construct.model in AGENT.md metadata or "
            "set ANTHROPIC_MODEL env var."
        )

    settings_kwargs: dict[str, Any] = {}
    if config.disable_thinking:
        settings_kwargs["anthropic_thinking"] = BetaThinkingConfigDisabledParam(type="disabled")

    model = AnthropicModel(
        model_name=model_name,
        provider=AnthropicProvider(),
        settings=AnthropicModelSettings(**settings_kwargs),
    )
    agent = Agent(name=name, description=description, model=model)

    return MementoPydanticAgentImpl(
        agent=agent,
        config=config,
        name=name,
        description=description,
    )
