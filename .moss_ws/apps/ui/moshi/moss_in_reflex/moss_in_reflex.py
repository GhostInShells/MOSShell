import asyncio
import inspect
from contextlib import asynccontextmanager

from aiohttp import web as aiohttp_web
import reflex as rx
from ghoshell_moss import PyChannel, Message, Text, Matrix
from ghoshell_moss.contracts import ResourceRegistry
from ghoshell_moss.core import ChannelCtx

from framework.events import EventModel, LayoutEvent, LoadChapterEvent, StreamEvent, SetEvent, AppendEvent, UpdateEvent, PopEvent, ClearEvent
from moss_in_reflex.course_storage import CourseStorage, CourseStorageMeta
from moss_in_reflex.context_messages import idle, discussing, preparing, teaching
from moss_in_reflex.lecture_brain import (
    POINT_ADVANCED, CHAPTER_ADVANCED, LECTURE_ENDED,
    LOADING, LECTURING, PAUSED, ENDED,
)
from ghoshell_moss.topics.audio import AudioRuntimeTopic
from ghoshell_moss.contracts.resource import ResourceStorageFactoryBootstrapper
from ghoshell_moss.contracts.configs import ConfigStore

# ── 子模块 ──
from moss_in_reflex.subtitle import setup_subtitle
from moss_in_reflex.channel_commands import MossRuntime, register_commands
from moss_in_reflex.http_endpoints import create_internal_app

# ── 模块级全局状态（state.py）──
from moss_in_reflex.state import (
    logger,
    CONFIG, LAYOUTS, LAYOUT_COMPONENTS, LAYOUT_CHANNEL_STATES,
    QUEUE,
    _LAYOUT, _SNAPSHOTS,
    _COURSE_MGR, _LECTURE_BRAIN,
    _SSE_EVENT, _SSE_QUEUE, _SSE_LOCK,
    _SUBTITLE_EVENT, _SUBTITLE_QUEUE, _SUBTITLE_LOCK,
    _DANMAKU_EVENT, _DANMAKU_QUEUE, _DANMAKU_LOCK,
)

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
        sn = _COURSE_MGR.get_speaker_notes(_COURSE_MGR.chapter_index)
        # 将 LectureBrain 实时状态合并到 talking_points，避免与持久化数据矛盾
        if _LECTURE_BRAIN.points:
            lb_status = {p.get("id"): p.get("status") for p in _LECTURE_BRAIN.points}
            merged_points = []
            for tp in sn.get("talking_points", []):
                tp_copy = dict(tp)
                tp_id = tp_copy.get("id", "")
                if tp_id in lb_status:
                    tp_copy["status"] = lb_status[tp_id]
                merged_points.append(tp_copy)
            sn = {**sn, "talking_points": merged_points}
        messages.append(
            Message.new(tag="context-mode").with_content(
                Text(text=teaching(_COURSE_MGR.name, list(_COURSE_MGR.outline),
                                  _COURSE_MGR.chapter_index, sn))
            )
        )
        # 附加 LectureBrain 实时状态
        if _LECTURE_BRAIN.is_active:
            lb_lines = [
                f"### 讲课实时状态: {_LECTURE_BRAIN.status}",
                f"章节: {_LECTURE_BRAIN.current_chapter + 1}/{_LECTURE_BRAIN.total_chapters}",
                f"段落进度: {_LECTURE_BRAIN.progress_summary}",
            ]
            if _LECTURE_BRAIN.points:
                lb_lines.append("")
                for tp in _LECTURE_BRAIN.points:
                    mark = {"done": "✓", "active": "→", "pending": "…"}.get(
                        tp.get("status", ""), "…"
                    )
                    lb_lines.append(f"  {mark} {tp.get('text', '')}")
            lb_lines.append("")
            lb_lines.append(
                "讲完当前 → 要点后调 <apps.ui_moshi:advance_point />，"
                "调完后**停止输出**，等待系统 Signal 唤醒后再继续下一段。"
                "段落间隙调 <apps.ui_moshi:check_messages />。"
            )
            if _LECTURE_BRAIN.status == PAUSED:
                lb_lines.append("⚠ 当前已暂停，回答完问题后调 <apps.ui_moshi:resume_lecture /> 恢复。")
            messages.append(
                Message.new(tag="lecture-state").with_content(
                    Text(text="\n".join(lb_lines))
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

    # ── 聊天指令（按模式隔离：teaching 模式不开放聊天功能）──
    if _COURSE_MGR.mode != "teaching":
        messages.append(
            Message.new(tag="chat-instruction").with_content(
                Text(text="\n".join([
                    "## 聊天功能",
                    "你正在和用户进行实时文字对话。当收到用户消息（Signal name='input'）时，",
                    "请使用 <apps.ui_moshi:chat_reply> 命令回复用户。",
                    "你的回复应该是自然的中文对话，直接写回复内容即可，不需要任何标记或前缀。",
                ]))
            )
        )

    return messages


async def moss():
    chan = PyChannel(name="moshi", description="魔师 Moshi — AI 内容传递协作助手。提供课程管理、聊天交互、白板渲染能力")
    chan.build.context_messages(context_messages)

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
        matrix.container.set(CourseStorage, meta.factory(matrix.container))
        _course_storage = meta.factory(matrix.container)
        _registry = matrix.container.force_fetch(ResourceRegistry)

        # ── TTS 门控：订阅 AudioRuntimeTopic 窗口 ──
        _runtime_window = matrix.session.topics.create_window_for(
            AudioRuntimeTopic, max_size=10
        )
        await _runtime_window.wait_started()
        logger.info("[moss] AudioRuntimeTopic 窗口已就绪")

        # ── 字幕订阅：Topic 总线 或 HTTP 旁路 ──
        await setup_subtitle(matrix, matrix.container.force_fetch(ConfigStore))

        # ── 注册 Channel 命令 ──
        runtime = MossRuntime(
            matrix=matrix,
            registry=_registry,
            course_storage=_course_storage,
            runtime_window=_runtime_window,
        )
        register_commands(chan, runtime)

        # ── 内部 HTTP 桥接 (:9733) — 供 main.py 发 Signal + 收 SSE ──
        internal_app = create_internal_app(matrix, _course_storage, runtime)

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
