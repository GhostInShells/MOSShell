import asyncio
import importlib
import inspect
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web as aiohttp_web
import reflex as rx
from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, DefaultFileStorage
from ghoshell_moss import PyChannel, Message, Text, Matrix
from ghoshell_moss.contracts import ResourceRegistry
from ghoshell_moss.core import ChannelCtx
from ghoshell_moss.core.concepts.channel import ChannelRuntime
from pydantic import Field

from framework.events import EventModel, LayoutEvent, LoadChapterEvent, StreamEvent, SetEvent, AppendEvent, UpdateEvent, PopEvent, ClearEvent
from framework.helpers.layout_snapshot import LayoutSnapshot
from framework.runtime.event_generator import build, _await_event
from moss_in_reflex.course_manager import CourseManager
from moss_in_reflex.course_storage import CourseStorage
from moss_in_reflex.context_messages import idle, discussing, preparing, teaching
from ghoshell_moss.contracts.workspace import Workspace

from moss_in_reflex.course_storage import CourseStorageMeta
from ghoshell_moss.contracts.resource import ResourceStorageFactoryBootstrapper
from ghoshell_moss.core.blueprint.mindflow import InputSignal

# =========================== System ===========================
logger = logging.getLogger("moss")
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

# ── 课程持久化状态 ──
_COURSE_MGR = CourseManager()

# ── SSE 聊天推送 ──
_SSE_EVENT: asyncio.Event = asyncio.Event()
_SSE_QUEUE: list[dict] = []          # [{"role":"ai","text":"..."}]
_SSE_LOCK: asyncio.Lock = asyncio.Lock()

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
                if isinstance(event, LoadChapterEvent):
                    ch = event.chapter_data

                    # 清空全部字段
                    for field in ("title", "sub_title", "image", "main_text", "annotations", "appreciation"):
                        handler = f"clear_{field}"
                        if hasattr(current.State, handler):
                            yield getattr(current.State, handler)()

                    # 逐字段填充
                    if ch.get("title"):
                        yield current.State.stream_title(ch["title"])
                    if ch.get("sub_title"):
                        yield current.State.stream_sub_title(ch["sub_title"])
                    if ch.get("main_text"):
                        yield current.State.stream_main_text(ch["main_text"])
                    for ann in ch.get("annotations", []):
                        yield current.State.append_annotations(ann)
                    if ch.get("appreciation"):
                        yield current.State.stream_appreciation(ch["appreciation"])
                    for img in ch.get("images", []):
                        yield current.State.append_image(img)

                    handler_missing = None  # 已处理，不需要单独的 callback

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
    logger.info("[context_messages] >>> 被调用, mode=%s", _COURSE_MGR.mode)
    messages = []

    # ── 情境感知：调模板函数获取文案 ──

    if _COURSE_MGR.mode == "discussing":
        messages.append(
            Message.new(tag="context-mode").with_content(
                Text(text=discussing(_COURSE_MGR.name or "未命名"))
            )
        )
    elif _COURSE_MGR.mode == "preparing" and _COURSE_MGR.outline:
        messages.append(
            Message.new(tag="context-mode").with_content(
                Text(text=preparing(list(_COURSE_MGR.outline), list(_COURSE_MGR.saved_chapters)))
            )
        )
    elif _COURSE_MGR.mode == "teaching" and _COURSE_MGR.is_loaded:
        messages.append(
            Message.new(tag="context-mode").with_content(
                Text(text=teaching(_COURSE_MGR.name, list(_COURSE_MGR.outline), _COURSE_MGR.chapter_index))
            )
        )
    else:
        messages.append(
            Message.new(tag="context-mode").with_content(Text(text=idle()))
        )

    # ── 布局源码 + State 快照（仅备课/讨论模式，按模式信息隔离） ──
    if _COURSE_MGR.mode in ("discussing", "preparing"):
        for l in LAYOUTS:
            if l.name() == _LAYOUT.name:
                module = inspect.getmodule(l)
                source_code = inspect.getsource(module)
                messages.append(
                    Message.new(tag="layout-source-code").with_content(
                        Text(text=f"当前 layout: {l.name()}，reflex 源码如下\n"),
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

    # ── 可用资源（图片 + 课程） ──
    registry = ChannelCtx.container().force_fetch(ResourceRegistry)
    resource_msg = Message.new(tag="resources")

    pil_infos = await registry.list_infos(scheme="pil-image")
    for info in pil_infos:
        resource_msg.with_content(
            f"locator: {info.locator} description: {info.description}\n"
        )

    course_infos = await registry.list_infos(scheme="course")
    for info in course_infos:
        course_name = info.path.rsplit("/", 1)[0] if "/" in info.path else info.path
        resource_msg.with_content(
            f"locator: {info.locator} course: {course_name} title: {info.title} 章节数: {info.chapters_count}\n"
        )

    if pil_infos or course_infos:
        messages.append(resource_msg)

    # ── 聊天指令 ──
    messages.append(
        Message.new(tag="chat-instruction").with_content(
            Text(text="\n".join([
                "## 聊天功能",
                "你正在和用户进行实时文字对话。当收到用户消息（Signal name='input'）时，",
                "请使用 <moshi:chat_reply> 命令回复用户。",
                "你的回复应该是自然的中文对话，直接写回复内容即可，不需要任何标记或前缀。",
            ]))
        )
    )

    return messages


async def moss():
    chan = PyChannel(name="moshi", description="魔师 Moshi — AI 内容传递协作助手。提供课程管理、聊天交互、白板渲染能力")
    chan.build.context_messages(context_messages)

    # -- 课程持久化命令 --

    @chan.build.command(
        name="start_prepare",
        doc="进入讨论大纲阶段。带 course 参数=开始备课，不带参数=重置回空闲",
        timeout=5.0,
        blocking=True,
    )
    async def start_prepare(course: str = "") -> str:
        return _COURSE_MGR.start_prepare(course)

    @chan.build.command(
        name="set_outline",
        doc="锁定课程大纲。course=课程名, title=课程标题, chapters=逗号分隔的章节标识列表",
        timeout=5.0,
        blocking=True,
    )
    async def set_outline(course: str, title: str, chapters: str) -> str:
        storage = ChannelCtx.container().force_fetch(CourseStorage)
        return await _COURSE_MGR.set_outline(course, title, chapters, storage)

    @chan.build.command(
        name="save_chapter",
        doc="存档当前章节。chapter=章节标识, locators=逗号分隔的图片 locator。",
        timeout=10.0,
        blocking=True,
    )
    async def save_chapter(chapter: str, locators: str = "") -> str:
        snap = _SNAPSHOTS.get(_LAYOUT.name)
        if snap is None:
            raise RuntimeError("no active layout snapshot")

        full = snap.get_full()
        locator_list = CourseManager.parse_locators(locators)

        # 校验 locator 数量（需要 snapshot，留在命令侧）
        image_count = full.get("image", 0)
        if not isinstance(image_count, int):
            image_count = 0
        if len(locator_list) != image_count:
            from ghoshell_moss import CommandError, CommandErrorCode
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=(
                    f"locator count mismatch: Ghost 传了 {len(locator_list)} 个，"
                    f"页面有 {image_count} 张图"
                ),
            )

        storage = ChannelCtx.container().force_fetch(CourseStorage)
        return await _COURSE_MGR.save_chapter(chapter, locator_list, full, storage)

    # -- 课程加载 / 讲课翻页 --

    async def _switch_to_chapter(n: int) -> None:
        """从 _COURSE_MGR 取章节数据，解析图片，通过 LoadChapterEvent 上屏。"""
        chapter_data = _COURSE_MGR.get_chapter_data(n)

        # 解析图片 locator → PIL 对象
        registry = ChannelCtx.container().force_fetch(ResourceRegistry)
        images = []
        for loc in chapter_data.get("images", []):
            item = await registry.get(loc)
            if item:
                images.append(await item.get())
        chapter_data["images"] = images

        await _await_event(QUEUE, LoadChapterEvent(chapter_data=chapter_data))

        _COURSE_MGR.set_chapter_index(n)

    @chan.build.command(
        name="load_course",
        doc="加载指定课程并继续备课。course=课程名",
        timeout=15.0,
        blocking=True,
    )
    async def load_course(course: str) -> str:
        storage = ChannelCtx.container().force_fetch(CourseStorage)
        return await _COURSE_MGR.load_course(course, storage)

    @chan.build.command(
        name="start_teaching",
        doc="讲课模式，开讲课程。course=课程名（未加载时必传），start_chapter=起始章节序号（默认 0）",
        timeout=15.0,
        blocking=True,
    )
    async def start_teaching(course: str = "", start_chapter: int = 0) -> str:
        if not _COURSE_MGR.is_loaded:
            if not course:
                raise RuntimeError("no course loaded — pass course name or call load_course first")
            storage = ChannelCtx.container().force_fetch(CourseStorage)
            await _COURSE_MGR.load_course(course, storage)

        result = _COURSE_MGR.start_teaching(start_chapter)
        await _switch_to_chapter(start_chapter)
        return result

    @chan.build.command(
        name="switch_chapter",
        doc="切换到指定章节序号（从 0 开始）",
        timeout=10.0,
        blocking=True,
    )
    async def switch_chapter(index: int = 0) -> str:
        if not _COURSE_MGR.is_loaded:
            raise RuntimeError("no course loaded — call load_course first")
        if index < 0 or index >= len(_COURSE_MGR.outline):
            raise RuntimeError(
                f"chapter index {index} out of range (0-{len(_COURSE_MGR.outline) - 1})"
            )

        await _switch_to_chapter(index)
        return f"chapter {index}: {_COURSE_MGR.outline[index]}"

    @chan.build.command(
        name="chat_reply",
        doc="回复聊天消息。text=回复正文",
        timeout=5.0,
        blocking=False,
    )
    async def chat_reply(text: str) -> str:
        msg = {"role": "ai", "text": text}
        async with _SSE_LOCK:
            _SSE_QUEUE.append(msg)
        _SSE_EVENT.set()
        logger.info("[chat_reply] SSE 推送 role=ai text=%.80s", text)
        return "ok"

    first = True
    for state in LAYOUT_CHANNEL_STATES:
        if first:
            chan.with_state(state, is_default=True)
            first = False
            continue

        chan.with_state(state)

    matrix = Matrix.discover()
    _internal_runner = None
    async with matrix:

        meta = CourseStorageMeta()
        ResourceStorageFactoryBootstrapper(meta).bootstrap(matrix.container)
        # ResourceStorageFactoryBootstrapper 只做了 registry.register，
        # container.set 还是要手动做（channel 方法需要 force_fetch(CourseStorage)）
        matrix.container.set(CourseStorage, meta.factory(matrix.container))

        # 内部 HTTP 桥接 (:9733) — 供 main.py 发 Signal + 收 SSE
        internal_app = aiohttp_web.Application()

        async def _internal_chat_in(request: aiohttp_web.Request):
            """main.py 转发用户消息 → 发 Signal 唤醒 Ghost。"""
            data = await request.json()
            text = (data.get("text") or "").strip()
            mode = data.get("mode", "")
            if not text:
                return aiohttp_web.json_response({"error": "empty text"}, status=400)
            signal = InputSignal().to_signal(
                Message.new(tag="chat-input").with_content(Text(text=text)),
                description=f"聊天: {text[:50]}",
            )
            matrix.session.add_signal(signal)
            logger.info("[chat_in] Signal 已发送, text=%.60s", text)
            return aiohttp_web.json_response({"ok": True})

        async def _internal_chat_stream(request: aiohttp_web.Request):
            """SSE：监听 asyncio.Event，推送 AI 回复给 main.py。"""
            resp = aiohttp_web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await resp.prepare(request)
            logger.info("[chat_stream] SSE 客户端连接")
            await resp.write(b": connected\n\n")
            sent = 0
            try:
                while True:
                    await asyncio.wait_for(
                        _SSE_EVENT.wait(), timeout=30
                    )
                    _SSE_EVENT.clear()
                    async with _SSE_LOCK:
                        while _SSE_QUEUE:
                            msg = _SSE_QUEUE.pop(0)
                            payload = json.dumps(msg, ensure_ascii=False)
                            await resp.write(f"data: {payload}\n\n".encode())
                            sent += 1
                            logger.info("[chat_stream] 推送给前端 role=%s text=%.60s", msg.get("role"), msg.get("text"))
            except asyncio.TimeoutError:
                pass
            except (ConnectionResetError, ConnectionAbortedError):
                logger.info("[chat_stream] SSE 客户端断开, 共推送 %d 条", sent)
            except Exception:
                logger.exception("[chat_stream] 内部错误")
            return resp

        internal_app.router.add_post("/_internal/chat_in", _internal_chat_in)
        internal_app.router.add_get("/_internal/chat_stream", _internal_chat_stream)

        _internal_runner = aiohttp_web.AppRunner(internal_app)
        await _internal_runner.setup()
        _internal_site = aiohttp_web.TCPSite(_internal_runner, "127.0.0.1", 9733)
        await _internal_site.start()
        logger.info("内部 HTTP 桥接 → http://127.0.0.1:9733")

        await matrix.provide_channel(chan)

    # moss() 退出时清理内部 HTTP 桥接
    if _internal_runner is not None:
        try:
            await _internal_runner.cleanup()
        except Exception:
            pass


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
