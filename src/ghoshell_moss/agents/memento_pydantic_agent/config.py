"""
memento_pydantic_agent 家族级 config.

BaseModel 全字段默认能跑 — AGENT.md 的 `construct` 字段 dict 直接
`model_validate` 得实例. 未识别字段静默忽略 (BaseModel 默认行为).

model=None 时 fallback 到 ANTHROPIC_MODEL 环境变量 (对齐 atom `_meta.py`).
provider=None 时用 pydantic-ai 默认 `AnthropicProvider()` (读 ANTHROPIC_API_KEY /
ANTHROPIC_BASE_URL 环境变量, 无需显式传).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["MementoPydanticAgentConfig"]


class MementoPydanticAgentConfig(BaseModel):
    """家族级配置. 全字段默认."""

    model: str | None = Field(
        default=None,
        description="Anthropic model name. None 时读 ANTHROPIC_MODEL 环境变量.",
    )
    disable_thinking: bool = Field(
        default=True,
        description="是否禁用 extended thinking. 对齐 atom 默认.",
    )
