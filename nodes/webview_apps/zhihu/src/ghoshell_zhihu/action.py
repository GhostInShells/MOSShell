"""Action 是模型签发的意图，也是 human surface 的卡片结构。

模型通过 channel 的 ``action`` 命令签发一个 Action JSON（text__ 参数），node 物化它、
路由到 human surface 做授权、跑 CLI、把结果写回来。Action 本身是声明：读哪个命令族、
带什么参数、可选地把返回值加工/渲染成界面元素。

四个字段里 ``transform`` / ``render`` 可选；``type`` / ``args`` 的合法性由 zhihu-cli 校验，
node 不做 per-type 校验 —— 这是 node 保持薄的关键。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    """模型签发的一个知乎读数据意图。

    - ``type``: 命令族名，与 ``capabilities`` 返回的 ``name`` 对齐（如 "me stats"）。
    - ``args``: skill 参数平铺 dict，参数名见 ``help <type>``。
    - ``transform``: 可选 TS 源码 ``(data, deps) => processed``，加工返回值。
    - ``render``: 可选 TS 源码 ``(processed, deps) => element``，渲染成界面元素。
    """

    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    transform: str | None = None
    render: str | None = None


class ActionResult(BaseModel):
    """一次 action 取回的数据，带结构与样本（模型据此写 transform/render）。"""

    type: str = ""
    count: int | None = None
    fields: list[str] = Field(default_factory=list)
    sample: Any = None
    data_ref: str | None = None
    raw: Any = None


def extract(data: Any) -> ActionResult:
    """从 zhihu-cli 的 ``Data`` 提取结构与样本。

    列表类（me contents / search / comments）抽 Items；dict 类（stats / content
    detail）直接给顶层键 + 一个缩略样本。全量留在 ``raw``，超阈值时由调用方落盘。
    """
    if isinstance(data, list):
        return ActionResult(
            count=len(data),
            fields=list(data[0].keys()) if data else [],
            sample=data[:2],
            raw=data,
        )
    if isinstance(data, dict):
        items = data.get("Items")
        if isinstance(items, list):
            return ActionResult(
                count=len(items),
                fields=list(items[0].keys()) if items else [],
                sample=items[:2],
                raw=data,
            )
        return ActionResult(
            count=None,
            fields=list(data.keys()),
            sample=_shrink(data),
            raw=data,
        )
    return ActionResult(sample=data, raw=data)


def _shrink(value: Any, depth: int = 0) -> Any:
    """把深结构缩到可读样本，避免 sample 本身吃掉 token。"""
    if depth >= 2:
        if isinstance(value, dict):
            return {"…": len(value)}
        if isinstance(value, list):
            return f"[{len(value)} items]"
        return value
    if isinstance(value, dict):
        return {k: _shrink(v, depth + 1) for k, v in list(value.items())[:8]}
    if isinstance(value, list):
        return [_shrink(v, depth + 1) for v in value[:3]]
    return value
