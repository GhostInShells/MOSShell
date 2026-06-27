import asyncio
from dataclasses import dataclass, field as dc_field
from typing import Any, ClassVar

from ghoshell_common.helpers import uuid


class VideoLocator(str):
    """视频资源定位符。继承 str，可被 Reflex/pydantic 透明序列化。
    布局中用 list[VideoLocator] 标注，build() 据此生成 locator→URL 命令。"""
    pass


@dataclass(kw_only=True)
class EventModel:
    event_type: ClassVar[str] = ""

    event_id: str = dc_field(default_factory=uuid)
    future: asyncio.Future | None = dc_field(default=None, repr=False)


@dataclass(kw_only=True)
class LayoutEvent(EventModel):
    event_type: ClassVar[str] = "set_layout"

    layout: str = ""


@dataclass(kw_only=True)
class StreamEvent(EventModel):
    event_type: ClassVar[str] = "stream"

    field: str = ""
    chunk: str = ""


@dataclass(kw_only=True)
class SetEvent(EventModel):
    event_type: ClassVar[str] = "set"

    field: str = ""
    data: Any = None


@dataclass(kw_only=True)
class AppendEvent(EventModel):
    event_type: ClassVar[str] = "append"

    field: str = ""
    data: Any = None


@dataclass(kw_only=True)
class UpdateEvent(EventModel):
    event_type: ClassVar[str] = "update"

    field: str = ""
    index: int = 0
    data: Any = None


@dataclass(kw_only=True)
class PopEvent(EventModel):
    event_type: ClassVar[str] = "pop"

    field: str = ""


@dataclass(kw_only=True)
class ClearEvent(EventModel):
    event_type: ClassVar[str] = "clear"

    field: str = ""
