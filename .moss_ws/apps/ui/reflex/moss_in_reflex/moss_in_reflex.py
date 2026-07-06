import asyncio
import importlib
import inspect
import json
import logging
import os
import signal
import typing
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pydantic
import reflex as rx
from PIL import Image
from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, DefaultFileStorage
from ghoshell_moss import PyChannel, Message, Text, Matrix, Observe
from ghoshell_moss.contracts import ResourceRegistry
from ghoshell_moss.core import ChannelCtx, PyCommand
from ghoshell_moss.core.concepts.channel import ChannelRuntime
from pydantic import Field

from framework.events import EventModel, LayoutEvent, StreamEvent, SetEvent, AppendEvent, UpdateEvent, PopEvent, ClearEvent
from framework.helpers.layout_snapshot import LayoutSnapshot
from framework.runtime.event_generator import build

# =========================== System ===========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Config(YamlConfig):
    relative_path = "config.yaml"

    layouts: list[str] = Field(default_factory=list, description="List of layout names")


def _load_config() -> Config:
    config_dir = Path(__file__).parent.absolute()
    store = WorkspaceConfigs(DefaultFileStorage(dir_=str(config_dir)))

    mode_name = os.environ.get("MOSS_MODE_NAME")
    if mode_name and (config_dir / f"config.{mode_name}.yaml").exists():
        Config.relative_path = f"config.{mode_name}.yaml"

    return store.get_or_create(Config())


CONFIG = _load_config()

def load_layouts() -> list:
    """从 layouts.toml 动态加载 Layout 类。"""
    layouts = []
    for module_path in CONFIG.layouts:
        module_name, class_name = module_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        layouts.append(cls)
        logger.info(f"Loaded layout: {module_path}")
    return layouts

# 外部系统和Reflex系统通信媒介
QUEUE: asyncio.Queue[EventModel] = asyncio.Queue()

# 注册可用的LAYOUT（从 layouts.toml 动态加载）
LAYOUTS = load_layouts()

# 动态生成LAYOUT运行时
LAYOUT_COMPONENTS = [(l.name(), l.create()) for l in LAYOUTS]
LAYOUT_CHANNEL_STATES = []

for i, _l in enumerate(LAYOUTS):
    _n, _lc = LAYOUT_COMPONENTS[i]
    LAYOUT_CHANNEL_STATES.append(build(_n, _lc.State, _l, QUEUE))

# Layout 状态镜像 — Reflex State 的模块级只读缓存
# self.layout 是权威数据源（持久化，hot-reload 存活），_LAYOUT 是模块级镜像
# moss_listener 同步写，context_messages() 只读
@dataclass
class _LayoutMirror:
    name: str = LAYOUTS[0].name()

    def get_component(self) -> rx.Component:
        for n, c in LAYOUT_COMPONENTS:
            if n == self.name:
                return c
        return LAYOUT_COMPONENTS[0][1]

_LAYOUT = _LayoutMirror()

# Layout 状态快照 — moss_listener 循环末尾一把读，context_messages() 消费
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
_SNAPSHOTS: dict[str, LayoutSnapshot] = {
    name: LayoutSnapshot(name, component.State, _SNAPSHOT_DIR)
    for name, component in LAYOUT_COMPONENTS
}

# =========================== System ===========================

# =========================== Reflex ===========================
def _apply_event_to_state(substate, event: EventModel, state_class: type[rx.State]) -> object:
    """在 async with self 内直接操作 substate，返回变更后的新值用于增量 snapshot 更新。"""
    field = event.field
    type_hint = state_class.__annotations__.get(field)
    if type_hint is None:
        return None
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)
    elem_type = args[0] if args else None

    if isinstance(event, ClearEvent):
        if isinstance(type_hint, type) and issubclass(type_hint, str):
            setattr(substate, field, "")
            return ""
        elif origin is list:
            lst = getattr(substate, field)
            lst.clear()
            return list(lst)
        elif type_hint is Image.Image:
            setattr(substate, field, None)
            return None
        elif isinstance(type_hint, type) and issubclass(type_hint, pydantic.BaseModel):
            empty = type_hint()
            setattr(substate, field, empty)
            return empty
        else:
            setattr(substate, field, "")
            return ""

    elif isinstance(event, StreamEvent):
        if isinstance(type_hint, type) and issubclass(type_hint, str):
            val = getattr(substate, field)
            new_val = val + event.chunk
            setattr(substate, field, new_val)
            logger.info("StreamEvent str field=%r old=%r chunk=%r new=%r", field, val, event.chunk, new_val)
            return new_val
        elif origin is list and isinstance(elem_type, type) and issubclass(elem_type, str):
            lst = getattr(substate, field)
            if lst:
                lst[-1] += event.chunk
                return list(lst)

    elif isinstance(event, SetEvent):
        setattr(substate, field, event.data)
        return event.data

    elif isinstance(event, AppendEvent):
        if origin is list:
            lst = getattr(substate, field)
            if isinstance(elem_type, type) and issubclass(elem_type, str):
                lst.append(event.data)
            elif isinstance(elem_type, type) and issubclass(elem_type, pydantic.BaseModel):
                parsed = elem_type.model_validate_json(event.data) if isinstance(event.data, str) else event.data
                lst.append(parsed)
            elif elem_type is dict:
                parsed = json.loads(event.data) if isinstance(event.data, str) else event.data
                lst.append(parsed)
            elif elem_type is Image.Image:
                lst.append(event.data)
            return list(lst)

    elif isinstance(event, UpdateEvent):
        if origin is list:
            lst = getattr(substate, field)
            if isinstance(elem_type, type) and issubclass(elem_type, pydantic.BaseModel):
                parsed = elem_type.model_validate_json(event.data) if isinstance(event.data, str) else event.data
            elif elem_type is dict:
                parsed = json.loads(event.data) if isinstance(event.data, str) else event.data
            else:
                parsed = event.data

            if event.index >= len(lst):
                lst.append(parsed)
            else:
                lst[event.index] = parsed
            return list(lst)

    elif isinstance(event, PopEvent):
        if origin is list:
            lst = getattr(substate, field)
            if lst:
                lst.pop()
            return list(lst)

    return None


class State(rx.State):
    """The app state."""

    layout: str = LAYOUTS[0].name()

    @rx.event(background=True)
    async def moss_listener(self):
        # Yield immediately so Reflex can finish on_load and render the page
        # yield

        # hot-reload 后模块全局变量被重置，从 Reflex State 权威恢复
        # self.layout 存储在 Reflex StateManager 中，不随模块 reload 丢失
        if self.layout != _LAYOUT.name:
            _LAYOUT.name = self.layout
            logger.info("Layout 从 Reflex State 恢复: %s", self.layout)

        while True:
            # 先取事件；若 background task 被取消，get() 抛 CancelledError，
            # 此时没有取到 item，直接退出，绝不能调用 task_done()（否则超调）。
            try:
                event = await QUEUE.get()
            except asyncio.CancelledError:
                return

            # 取到 item 后，无论处理成功与否，finally 都恰好 task_done() 一次，
            # 与 get() 一一对应。
            fut = event.future  # 命令侧通过 _await_event 塞入，否则为 None

            try:
                logger.info(f"moss_listener {event}")

                if isinstance(event, LayoutEvent):
                    _LAYOUT.name = event.layout
                    async with self:
                        self.layout = event.layout

                current = _LAYOUT.get_component()
                if not current:
                    logger.warning("Current layout is None")
                    if fut and not fut.done():
                        fut.set_exception(RuntimeError("no active layout"))
                    continue

                if isinstance(event, (StreamEvent, SetEvent, AppendEvent, UpdateEvent, PopEvent, ClearEvent)):
                    if hasattr(current.State, event.field):
                        async with self:
                            substate = await self.get_state(current.State)
                            new_val = _apply_event_to_state(substate, event, current.State)
                            if new_val is not None:
                                _SNAPSHOTS[_LAYOUT.name].update_field(event.field, new_val)
                    else:
                        logger.warning("Layout %r has no field %r", _LAYOUT.name, event.field)
                        if fut and not fut.done():
                            fut.set_exception(
                                RuntimeError(f"field {event.field} not found on layout {_LAYOUT.name}")
                            )

            except asyncio.CancelledError:
                if fut and not fut.done():
                    fut.cancel()
                return
            except Exception as ex:
                logger.error(f"Exception: {ex}")
                if fut and not fut.done():
                    fut.set_exception(ex)
            else:
                if fut and not fut.done():
                    fut.set_result(None)
            finally:
                QUEUE.task_done()


def index() -> rx.Component:
    return rx.box(
        rx.match(
            State.layout,
            *LAYOUT_COMPONENTS,
            rx.text("default")
        ),
    )
# =========================== Reflex ===========================

# =========================== MOSS ===========================
async def context_messages():
    messages = []

    for l in LAYOUTS:
        if l.name() == _LAYOUT.name:
            module = inspect.getmodule(l)
            source_code = inspect.getsource(module)
            messages.append(
                Message.new(tag="layout-source-code").with_content(
                    Text(text=f"当前layout为name: {l.name()}，reflex源代码如下\n"),
                    Text(text=source_code),
                )
            )

    snap = _SNAPSHOTS.get(_LAYOUT.name)
    if snap is not None:
        data = snap.get()
        if data:
            state_lines = [f"{k}: {v}" for k, v in data.items()]
            messages.append(
                Message.new(tag="current-state").with_content(
                    Text(text="\n".join(state_lines))
                )
            )

    registry = ChannelCtx.container().force_fetch(ResourceRegistry)
    resource_msg = Message.new(tag="resources")

    image_infos = await registry.list_infos(scheme="pil-image")
    for info in image_infos:
        resource_msg.with_content(
            f"locator: {info.locator} description: {info.description}\n"
        )

    video_infos = await registry.list_infos(scheme="local-video")
    for info in video_infos:
        resource_msg.with_content(
            f"locator: {info.locator} description: {info.description}\n"
        )

    webm_infos = await registry.list_infos(scheme="local-webm")
    for info in webm_infos:
        resource_msg.with_content(
            f"locator: {info.locator} description: {info.description}\n"
        )

    messages.append(resource_msg)
    return messages


def _frontend_url():
    host, port = "localhost", 3000
    try:
        from reflex.config import get_config
        cfg = get_config()
        port = cfg.frontend_port or port
    except Exception:
        pass
    return f"当前前端页面地址为 http://{host}:{port}"


async def switch_layout(layout_name: str) -> Observe:
    """切换当前布局和 ChannelState，并强制模型观察新上下文，调用当前command后，需要等待返回值才能继续渲染"""
    valid = {l.name() for l in LAYOUTS}
    if layout_name not in valid:
        return Observe.new(
            f"Unknown layout '{layout_name}'. Available: {', '.join(sorted(valid))}"
        )

    # 通过 kernel 的 switch_state 切换 ChannelState
    # 内部 on_startup 会发送 LayoutEvent 触发 Reflex UI 切换
    runtime = ChannelCtx.runtime()
    if runtime is not None and hasattr(runtime, "switch_state"):
        await runtime.switch_state(layout_name)

    logger.info("switch_layout: %s", layout_name)
    return Observe.new(f"Switched to layout: {layout_name}")


async def moss():
    chan = PyChannel(name="reflex", description="提供基于Reflex框架的流式GUI页面，用于AI实时渲染")
    chan.build.instruction(_frontend_url)
    chan.build.context_messages(context_messages)

    first = True
    for state in LAYOUT_CHANNEL_STATES:
        if first:
            chan.with_state(state, is_default=True)
            first = False
            continue

        chan.with_state(state)

    chan.build.add_command(PyCommand(
        func = switch_layout,
    ))

    matrix = Matrix.discover()
    async with matrix:
        await matrix.provide_channel(chan)

# =========================== MOSS ===========================

# =========================== Bootstrap ===========================
@asynccontextmanager
async def lifespan():
    task = asyncio.create_task(moss())
    yield  # ← 在 yield 之前是 startup，之后是 shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = rx.App()
app.register_lifespan_task(lifespan)
app.add_page(index, on_load=State.moss_listener)
# =========================== Bootstrap ===========================
