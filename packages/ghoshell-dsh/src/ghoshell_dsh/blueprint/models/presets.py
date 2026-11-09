"""
agentPreset 域 — dsh 的 agent 组合声明, 只读面.

镜像 ``@deepseek-ai/dsh-agent-preset-registry`` 的 ``agentPresets/list`` 与
``agentPresets/read`` 两个动词 (Typert Remote, 0.2.0 代际).

preset 是 dsh 的**组合声明**: 它决定一个 session 由哪些插件/工具/指令组成, 所以界面新建
agent 的第一步就是选一个. 它不是模型、也不是 workspace —— 那两件分别是
``session/selectModel`` 与 ``workspace/*`` 的事, 不要混进这里.

``agentPresets/select`` 给已有会话换 preset (且要求会话还没开聊), 本代际没有消费者,
不建模; 需要时再补 ``AgentPresetSelectParams``.

三个形状都是**读面**: 声明来自部署的 preset 文件, 客户端不改它.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AgentPresetDocument",
    "AgentPresetRoster",
    "AgentPresetRow",
]


class AgentPresetRow(BaseModel):
    """roster 里的一行 — 客户端看得见的声明摘要."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="稳定标识; 也是显示名的兜底.")
    isDefault: bool = Field(
        default=False,
        description="没有指名 preset 的会话是否组合这一个. 一个部署通常只有一个为真.",
    )
    name: str | None = Field(default=None, description="preset 自己发布的显示名.")
    description: str | None = Field(default=None, description="一句话说明它适合什么.")
    broken: str | None = Field(
        default=None,
        description=(
            "为什么它组合不出会话; 能组合时为 None. 坏声明照样留在 roster 里 —— "
            "「消失」和「坏掉」是两回事, 客户端要能分辨."
        ),
    )


class AgentPresetRoster(BaseModel):
    """一个部署当前提供的全部声明, **包括激活失败的**."""

    model_config = ConfigDict(extra="allow")

    presets: list[AgentPresetRow] = Field(
        default_factory=list, description="按部署声明顺序."
    )


class AgentPresetDocument(BaseModel):
    """一个 preset 的组合内容, 已渲染成可读文本."""

    model_config = ConfigDict(extra="allow")

    agentPreset: str = Field(default="", description="这份组合属于哪个 preset.")
    content: str = Field(
        default="",
        description="声明的子插件清单, 以 entry-list YAML 呈现 (``!!js`` 表达式原样保留).",
    )
    name: str | None = Field(default=None, description="显示名.")
    description: str | None = Field(default=None, description="一句话说明.")
