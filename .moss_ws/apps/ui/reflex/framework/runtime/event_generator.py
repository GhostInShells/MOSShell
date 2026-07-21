import asyncio
import inspect
import json
import typing

import pydantic
import reflex as rx
from PIL import Image
from ghoshell_moss import CommandError, CommandErrorCode, new_channel_state
from ghoshell_moss.contracts import ResourceRegistry
from ghoshell_moss.core import PyCommand, ChannelCtx

from framework.events import StreamEvent, ClearEvent, AppendEvent, PopEvent, UpdateEvent, SetEvent, LayoutEvent, EventModel, VideoLocator

_CMD_TIMEOUT = 5.0  # 非流式命令等待 UI 处理的超时秒数
_RESOURCE_BASE = "http://127.0.0.1:20880/resources"


def _locator_to_url(locator: str) -> str:
    """locator → HTTP URL: local-video://host/path → http://127.0.0.1:20880/resources/local-video/host/path"""
    if "://" not in locator:
        raise CommandError(code=CommandErrorCode.FAILED, message=f"Invalid locator: {locator}")
    scheme, rest = locator.split("://", 1)
    return f"{_RESOURCE_BASE}/{scheme}/{rest}"


async def _await_event(queue: asyncio.Queue, event: EventModel, timeout: float = _CMD_TIMEOUT) -> None:
    """塞 future 进事件，入队后同步等待 UI 侧 resolve/reject。"""
    fut = asyncio.get_event_loop().create_future()
    event.future = fut
    await queue.put(event)
    try:
        await asyncio.wait_for(fut, timeout=timeout)
    except TimeoutError:
        raise CommandError(
            code=CommandErrorCode.FAILED,
            message="UI 处理超时",
        )
    except Exception as e:
        raise CommandError(
            code=CommandErrorCode.FAILED,
            message=f"UI 处理失败: {e}",
        ) from e


def build(state_name, state_class: type[rx.State], layout_state_class: type[rx.ComponentState], queue: asyncio.Queue):
    commands = []
    annotations = state_class.__annotations__
    for name, type_ in annotations.items():
        ori_type = typing.get_origin(type_)
        if ori_type is list and typing.get_args(type_)[0] is VideoLocator:
            commands.extend(
                generate_video_list_command(name, state_class, queue)
            )
        elif type_ is VideoLocator:
            commands.extend(
                generate_video_command(name, state_class, queue)
            )
        elif str in [type_, ori_type]:
            commands.extend(
                generate_stream_command(name, state_class, queue)
            )
        elif list in [type_, ori_type]:
            elem_type = typing.get_args(type_)[0]
            commands.extend(
                generate_list_command(name, state_class, elem_type, queue)
            )
        elif issubclass(type_, pydantic.BaseModel):
            commands.extend(
                generate_pydantic_model_command(name, state_class, type_, queue)
            )
        elif Image.Image in [type_, ori_type]:
            commands.extend(
                generate_image_command(name, state_class, queue)
            )

    layout_doc = inspect.getdoc(layout_state_class) or ""
    state = new_channel_state(name=state_name, description=layout_doc)

    async def state_startup():
        await queue.put(LayoutEvent(layout=state_name))

    state.startup(state_startup)
    for command in commands:
        state.add_command(command)
    return state


def generate_stream_command(name: str, state_class: type[rx.State], queue: asyncio.Queue):
    # 1. 定义函数体逻辑
    async def stream(state: state_class, t: str):  # type: ignore[valid-type]
        # 动态给 state 的属性赋值
        setattr(state, name, getattr(state, name) + t)

    async def clear(state: state_class):  # type: ignore[valid-type]
        setattr(state, name, "")

    # 2. 动态修改函数名（符合 stream_{title} 格式）
    stream.__name__ = stream.__qualname__  = f"stream_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"

    # 3. 用 rx.event 装饰并返回
    stream_decorated = rx.event(stream)
    clear_decorated = rx.event(clear)
    setattr(state_class,  f"stream_{name}", stream_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    async def stream_command(chunks__):
        async for chunk in chunks__:
            await queue.put(StreamEvent(field=name, chunk=chunk))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    return [
        PyCommand(
            func=stream_command,
            name=f"stream_{name}",
            doc=f"流式渲染{name}字段",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}字段"
        )
    ]


def generate_list_command(name: str, state_class: type[rx.State], field_type, queue: asyncio.Queue):
    async def append(state: state_class, t: field_type):  # type: ignore[valid-type]
        if issubclass(field_type, pydantic.BaseModel):
            t = field_type.model_validate_json(t)
        if issubclass(field_type, dict):
            t = json.loads(t)
        getattr(state, name).append(t)

    async def pop(state: state_class):  # type: ignore[valid-type]
        ll = getattr(state, name)
        if len(ll) == 0:
            return
        getattr(state, name).pop()

    async def clear(state: state_class):  # type: ignore[valid-type]
        getattr(state, name).clear()

    async def stream_append(state: state_class, t: str): # type: ignore[valid-type]
        l = getattr(state, name)
        l[-1] += t

    async def update(state: state_class, index: int, t: field_type):  # type: ignore[valid-type]
        if issubclass(field_type, pydantic.BaseModel):
            t = field_type.model_validate_json(t)
        if issubclass(field_type, dict):
            t = json.loads(t)
        l = getattr(state, name)
        if index >= len(l):
            l.append(t)
        else:
            l[index] = t

    # 2. 动态修改函数名（符合 stream_{title} 格式）
    append.__name__ = append.__qualname__  = f"append_{name}"
    pop.__name__ = pop.__qualname__  = f"pop_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"
    stream_append.__name__ = stream_append.__qualname__ = f"stream_{name}"
    update.__name__ = update.__qualname__ = f"update_{name}"

    append_decorated = rx.event(append)
    pop_decorated = rx.event(pop)
    clear_decorated = rx.event(clear)
    stream_last_decorated = rx.event(stream_append)
    update_decorated = rx.event(update)

    setattr(state_class, f"append_{name}", append_decorated)
    setattr(state_class, f"pop_{name}", pop_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    if field_type == str:
        setattr(state_class, f"stream_{name}", stream_last_decorated)

    # update 仅对结构化元素（BaseModel / dict）有意义，str 列表无 id 概念
    if issubclass(field_type, (pydantic.BaseModel, dict)):
        setattr(state_class, f"update_{name}", update_decorated)

    async def append_command(text__):
        # 数据结构校验
        try:
            json.loads(text__)
        except json.decoder.JSONDecodeError as e:
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=e.msg,
            )
        await _await_event(queue, AppendEvent(field=name, data=text__))

    async def pop_command():
        await _await_event(queue, PopEvent(field=name))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    async def update_command(index: int, text__):
        try:
            json.loads(text__)
        except json.decoder.JSONDecodeError as e:
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=e.msg,
            )
        await _await_event(queue, UpdateEvent(field=name, index=index, data=text__))

    async def stream_command(chunks__):
        # 先创建一个最新的元素
        await queue.put(AppendEvent(field=name, data=""))

        # 流式追加内容到最新的元素
        async for chunk in chunks__:
            await queue.put(StreamEvent(field=name, chunk=chunk))

    async def append_image_command(locator: str):
        registry = ChannelCtx.container().force_fetch(ResourceRegistry)
        item = await registry.get(locator)
        if not item:
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=f"{locator} not found",
            )
        image = await item.get()
        if not isinstance(image, Image.Image):
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=f"{locator} is not an image",
            )

        await _await_event(queue, AppendEvent(field=name, data=image))

    commands = [
        PyCommand(
            func=pop_command,
            name=f"pop_{name}",
            doc=f"弹出{name}列表最后一个元素",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}列表",
        ),
    ]

    if field_type == str:
        commands.append(PyCommand(
            func=stream_command,
            name=f"stream_{name}",
            doc=f"追加一个元素到{name}，并流式渲染",
        ))
    if issubclass(field_type, pydantic.BaseModel):
        commands.extend([
            PyCommand(
                func=update_command,
                name=f"update_{name}",
                doc=f"更新{name}列表中指定索引的元素，字段格式为{field_type.model_json_schema()}",
            ),
            PyCommand(
                func=append_command,
                name=f"append_{name}",
                doc=f"追加元素到{name}列表，字段格式为{field_type.model_json_schema()}",
            ),
        ])
    if field_type == dict:
        commands.extend([
            PyCommand(
                func=update_command,
                name=f"update_{name}",
                doc=f"更新{name}列表中指定索引的元素",
            ),
            PyCommand(
                func=append_command,
                name=f"append_{name}",
                doc=f"追加元素到{name}列表",
            ),
        ])
    if field_type == Image.Image:
        commands.extend([
            PyCommand(
                func=append_image_command,
                name=f"append_{name}",
                doc=f"追加渲染一张图片到{name}列表，参数 locator 为资源定位符，必须作为 XML 属性传入。例如 <append_{name} locator=\"pil-image://path\" />"
            )
        ])

    return commands


def generate_pydantic_model_command(name: str, state_class: type[rx.State], field_type: type[pydantic.BaseModel], queue: asyncio.Queue):
    # 1. 定义函数体逻辑
    async def set_(state: state_class, t: str):  # type: ignore[valid-type]
        # 动态给 state 的属性赋值
        model = field_type.model_validate_json(t)
        setattr(state, name, model)

    async def clear(state: state_class):  # type: ignore[valid-type]
        setattr(state, name, field_type())

    # 2. 动态修改函数名（符合 stream_{title} 格式）
    set_.__name__ = set_.__qualname__  = f"set_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"

    # 3. 用 rx.event 装饰并返回
    set_decorated = rx.event(set_)
    clear_decorated = rx.event(clear)
    setattr(state_class,  f"set_{name}", set_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    async def set_command(t: str):
        model = field_type.model_validate_json(t)
        await _await_event(queue, SetEvent(field=name, data=model))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    return [
        PyCommand(
            func=set_command,
            name=f"set_{name}",
            doc=f"设置{name}字段，字段格式为{field_type.model_json_schema()}",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}字段"
        )
    ]


def generate_image_command(name: str, state_class: type[rx.State], queue: asyncio.Queue):
    async def set_(state: state_class, t: Image.Image):  # type: ignore[valid-type]
        setattr(state, name, t)

    async def clear(state: state_class):  # type: ignore[valid-type]
        setattr(state, name, None)

    set_.__name__ = set_.__qualname__ = f"set_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"

    set_decorated = rx.event(set_)
    clear_decorated = rx.event(clear)
    setattr(state_class, f"set_{name}", set_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    async def set_command(locator: str):
        registry = ChannelCtx.container().force_fetch(ResourceRegistry)
        item = await registry.get(locator)
        if not item:
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=f"{locator} not found",
            )
        image = await item.get()
        if not isinstance(image, Image.Image):
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=f"{locator} is not an image",
            )

        await _await_event(queue, SetEvent(field=name, data=image))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    return [
        PyCommand(
            func=set_command,
            name=f"set_{name}",
            doc=f"渲染一张图片到{name}字段，参数 locator 为资源定位符，必须作为 XML 属性传入。例如 <set_{name} locator=\"pil-image://path\" />",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}字段"
        )
    ]


def generate_video_command(name: str, state_class: type[rx.State], queue: asyncio.Queue):
    async def set_(state: state_class, t: str):  # type: ignore[valid-type]
        setattr(state, name, t)

    async def clear(state: state_class):  # type: ignore[valid-type]
        setattr(state, name, "")

    set_.__name__ = set_.__qualname__ = f"set_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"

    set_decorated = rx.event(set_)
    clear_decorated = rx.event(clear)
    setattr(state_class, f"set_{name}", set_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    async def set_command(locator: str):
        url = _locator_to_url(locator)
        await _await_event(queue, SetEvent(field=name, data=url))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    return [
        PyCommand(
            func=set_command,
            name=f"set_{name}",
            doc=f"设置{name}视频，参数 locator 为资源定位符。例如 <set_{name} locator=\"local-video://host/path\" />",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}字段",
        )
    ]


def generate_video_list_command(name: str, state_class: type[rx.State], queue: asyncio.Queue):
    async def append(state: state_class, t: str):  # type: ignore[valid-type]
        getattr(state, name).append(t)

    async def pop(state: state_class):  # type: ignore[valid-type]
        ll = getattr(state, name)
        if ll:
            ll.pop()

    async def clear(state: state_class):  # type: ignore[valid-type]
        getattr(state, name).clear()

    append.__name__ = append.__qualname__ = f"append_{name}"
    pop.__name__ = pop.__qualname__ = f"pop_{name}"
    clear.__name__ = clear.__qualname__ = f"clear_{name}"

    append_decorated = rx.event(append)
    pop_decorated = rx.event(pop)
    clear_decorated = rx.event(clear)
    setattr(state_class, f"append_{name}", append_decorated)
    setattr(state_class, f"pop_{name}", pop_decorated)
    setattr(state_class, f"clear_{name}", clear_decorated)

    async def append_command(locator: str):
        url = _locator_to_url(locator)
        await _await_event(queue, AppendEvent(field=name, data=url))

    async def pop_command():
        await _await_event(queue, PopEvent(field=name))

    async def clear_command():
        await _await_event(queue, ClearEvent(field=name))

    return [
        PyCommand(
            func=append_command,
            name=f"append_{name}",
            doc=f"追加视频到{name}列表，参数 locator 为资源定位符。例如 <append_{name} locator=\"local-video://host/path\" />",
        ),
        PyCommand(
            func=pop_command,
            name=f"pop_{name}",
            doc=f"弹出{name}列表最后一个元素",
        ),
        PyCommand(
            func=clear_command,
            name=f"clear_{name}",
            doc=f"清空{name}列表",
        ),
    ]
