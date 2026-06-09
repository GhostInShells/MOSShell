import asyncio
import importlib
import inspect
import logging
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import reflex as rx
from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, DefaultFileStorage
from ghoshell_moss import PyChannel, Message, Text, Matrix
from ghoshell_moss.contracts import ResourceRegistry
from ghoshell_moss.core import ChannelCtx
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

CONFIG = WorkspaceConfigs(
    DefaultFileStorage(dir_=str(Path(__file__).parent.absolute()))
).get_or_create(
    Config()
)

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
    name: str = "simple"

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
class State(rx.State):
    """The app state."""

    layout: str = "simple"

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

                handler_missing = None
                if isinstance(event, StreamEvent):
                    handler = f"stream_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)(event.chunk)
                    else:
                        handler_missing = handler
                if isinstance(event, SetEvent):
                    handler = f"set_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)(event.data)
                    else:
                        handler_missing = handler
                if isinstance(event, AppendEvent):
                    handler = f"append_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)(event.data)
                    else:
                        handler_missing = handler
                if isinstance(event, UpdateEvent):
                    handler = f"update_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)(event.index, event.data)
                    else:
                        handler_missing = handler
                if isinstance(event, PopEvent):
                    handler = f"pop_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)()
                    else:
                        handler_missing = handler
                if isinstance(event, ClearEvent):
                    handler = f"clear_{event.field}"
                    if hasattr(current.State, handler):
                        yield getattr(current.State, handler)()
                    else:
                        handler_missing = handler

                if handler_missing:
                    logger.warning("Layout %r has no handler %r", _LAYOUT.name, handler_missing)
                    if fut and not fut.done():
                        fut.set_exception(
                            RuntimeError(f"handler {handler_missing} not found on layout {_LAYOUT.name}")
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
                if handler_missing is None:
                    async with self:
                        await _SNAPSHOTS[_LAYOUT.name].refresh(self)
                if fut and not fut.done():
                    fut.set_result(None)
            finally:
                QUEUE.task_done()


def index() -> rx.Component:
    return rx.container(
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
    infos =  await registry.list_infos(scheme="pil-image")
    resource_msg = Message.new(tag="resources")
    for info in infos:
        resource_msg.with_content(
            f"locator: {info.locator} description: {info.description}\n"
        )
    messages.append(resource_msg)
    return messages


async def moss():
    chan = PyChannel(name="reflex", description="提供基于Reflex框架的流式GUI页面，用于AI实时渲染")
    chan.build.context_messages(context_messages)

    first = True
    for state in LAYOUT_CHANNEL_STATES:
        if first:
            chan.with_state(state, is_default=True)
            first = False
            continue

        chan.with_state(state)

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
