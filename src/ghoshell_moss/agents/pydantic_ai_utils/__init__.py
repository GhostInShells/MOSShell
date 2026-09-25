"""pydantic-ai 协议工具 — agents 与 llms 共用的序列化/反序列化.

依赖门控:模块级零 pydantic-ai import(照 llms/pydantic_ai_adapter/client.py
惰性模式)。architecture.py 能安全 import 本模块而不触发 pydantic-ai 加载;
真实使用点(每个函数)惰性 import。

第一批收敛物:消息序列化,从 llm funcs 倒过来提炼。锚的 CallAnchor.turns
与 memento 的 dry run messages 共用同一条 ModelMessagesTypeAdapter 链。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

__all__ = ["serialize_messages", "deserialize_messages"]


def serialize_messages(messages: list["ModelMessage"]) -> list[dict[str, Any]]:
    """pydantic-ai 标准序列化 message history — 保住 thinking/text/tool 所有 part.

    惰性 import:仅在真正序列化时才加载 pydantic-ai(照 client.py 模式)。
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


def deserialize_messages(turns: list[dict[str, Any]]) -> list["ModelMessage"]:
    """pydantic-ai 标准序列化反向 — dict 列表还原为 ModelMessage(消费锚)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.validate_python(turns)
