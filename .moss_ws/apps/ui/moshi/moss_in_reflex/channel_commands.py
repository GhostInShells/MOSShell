"""Channel 命令定义——所有 MOSS channel 命令的独立实现。

每个命令是模块级 async 函数，接受 MossRuntime 获取运行时依赖。
register_commands() 将它们注册到 PyChannel。
"""

import asyncio as _asyncio
import json
from dataclasses import dataclass

from ghoshell_moss import CommandError, CommandErrorCode, Message, Text
from ghoshell_moss.core import ChannelCtx
from ghoshell_moss.core.blueprint.mindflow import InputSignal, Priority

from framework.events import LoadChapterEvent
from framework.runtime.event_generator import _await_event

from moss_in_reflex.course_manager import CourseManager
from moss_in_reflex.course_storage import CourseStorage
from moss_in_reflex.lecture_brain import (
    POINT_ADVANCED, CHAPTER_ADVANCED, LECTURE_ENDED,
    LOADING, LECTURING, PAUSED, ENDED,
)

from moss_in_reflex.state import (
    logger,
    QUEUE,
    _LAYOUT, _SNAPSHOTS,
    _COURSE_MGR, _LECTURE_BRAIN,
    _SSE_EVENT, _SSE_QUEUE, _SSE_LOCK,
)


# =========================== Runtime Context ===========================

@dataclass
class MossRuntime:
    """运行时依赖——在 Matrix 初始化后创建，传递给需要 IoC/Matrix 引用的命令。"""
    matrix: object           # Matrix instance
    registry: object         # ResourceRegistry
    course_storage: object   # CourseStorage
    runtime_window: object   # TopicWindow[AudioRuntimeTopic]


# =========================== Helpers ===========================

async def _switch_to_chapter(n: int, runtime: MossRuntime) -> None:
    """从 _COURSE_MGR 取章节数据，解析图片，通过 LoadChapterEvent 上屏。"""
    chapter_data = _COURSE_MGR.get_chapter_data(n)

    # 解析图片 locator → PIL 对象
    images = []
    for loc in chapter_data.get("images", []):
        item = await runtime.registry.get(loc)
        if item:
            images.append(await item.get())
    chapter_data["images"] = images

    await _await_event(QUEUE, LoadChapterEvent(chapter_data=chapter_data))

    _COURSE_MGR.set_chapter_index(n)
    _LECTURE_BRAIN.current_chapter = n


async def _wait_tts_done(runtime: MossRuntime):
    """等待 TTS speaker 播放完成（150ms 轮询）。

    无 speaker topic 时最多等 ~450ms（3 次轮询），
    之后视为无 TTS 运行，立即返回以免死等。
    """
    empty_count = 0
    while True:
        for topic in reversed(runtime.runtime_window.values()):
            if topic.device_name == "speaker":
                if topic.running:
                    break  # 仍在播放，继续等
                else:
                    return  # 播放完成
        else:
            empty_count += 1
            if empty_count > 3:
                return
        await _asyncio.sleep(0.15)


# =========================== 命令函数 ===========================

async def cmd_start_prepare(course: str = "") -> str:
    return _COURSE_MGR.start_prepare(course)


async def cmd_set_outline(course: str, title: str, chapters: str) -> str:
    storage = ChannelCtx.container().force_fetch(CourseStorage)
    return await _COURSE_MGR.set_outline(course, title, chapters, storage)


async def cmd_save_chapter(chapter: str, locators: str = "", speaker_notes: str = "") -> str:
    snap = _SNAPSHOTS.get(_LAYOUT.name)
    if snap is None:
        raise RuntimeError("no active layout snapshot")

    full = snap.get_full()
    locator_list = CourseManager.parse_locators(locators)

    sn: dict | None = None
    if speaker_notes.strip():
        try:
            sn = json.loads(speaker_notes)
        except json.JSONDecodeError as e:
            raise CommandError(
                code=CommandErrorCode.FAILED,
                message=f"speaker_notes JSON 解析失败: {e}",
            )

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
    return await _COURSE_MGR.save_chapter(chapter, locator_list, full, storage, speaker_notes=sn)


async def cmd_load_course(course: str) -> str:
    storage = ChannelCtx.container().force_fetch(CourseStorage)
    return await _COURSE_MGR.load_course(course, storage)


async def cmd_start_teaching(course: str = "", start_chapter: int = 0, runtime: MossRuntime = None) -> str:
    if not _COURSE_MGR.is_loaded:
        if not course:
            raise RuntimeError("no course loaded — pass course name or call load_course first")
        storage = ChannelCtx.container().force_fetch(CourseStorage)
        await _COURSE_MGR.load_course(course, storage)

    result = _COURSE_MGR.start_teaching(start_chapter)
    await _switch_to_chapter(start_chapter, runtime)

    sn = _COURSE_MGR.get_speaker_notes(start_chapter)
    _LECTURE_BRAIN.start_loading(
        course=_COURSE_MGR.name,
        chapter=start_chapter,
        total_chapters=len(_COURSE_MGR.outline),
        talking_points=sn.get("talking_points", []),
        transitions=sn.get("transitions", []),
        key_data=sn.get("key_data", []),
        estimated_duration=sn.get("estimated_duration", 0),
    )
    _LECTURE_BRAIN.start_lecturing()
    logger.info("[start_teaching] LectureBrain → lecturing, chapter=%d, points=%d",
                start_chapter, len(_LECTURE_BRAIN.points))

    return result


async def cmd_switch_chapter(index: int = 0, runtime: MossRuntime = None) -> str:
    if not _COURSE_MGR.is_loaded:
        raise RuntimeError("no course loaded — call load_course first")
    if index < 0 or index >= len(_COURSE_MGR.outline):
        raise RuntimeError(
            f"chapter index {index} out of range (0-{len(_COURSE_MGR.outline) - 1})"
        )
    await _switch_to_chapter(index, runtime)
    return f"chapter {index}: {_COURSE_MGR.outline[index]}"


async def cmd_advance_point(runtime: MossRuntime) -> str:
    if _LECTURE_BRAIN.status == ENDED:
        return LECTURE_ENDED

    result = _LECTURE_BRAIN.advance_point()
    logger.info("[advance_point] → %s, progress=%s", result, _LECTURE_BRAIN.progress_summary)

    if result == CHAPTER_ADVANCED:
        await _wait_tts_done(runtime)
        logger.info("[advance_point] TTS 播放完成，开始翻章")

        next_ch = _COURSE_MGR.chapter_index + 1
        if next_ch < len(_COURSE_MGR.outline):
            await _switch_to_chapter(next_ch, runtime)
            sn = _COURSE_MGR.get_speaker_notes(next_ch)
            _LECTURE_BRAIN.advance_chapter(
                chapter=next_ch,
                talking_points=sn.get("talking_points", []),
                transitions=sn.get("transitions", []),
                key_data=sn.get("key_data", []),
                estimated_duration=sn.get("estimated_duration", 0),
            )
            logger.info("[advance_point] 翻页 → chapter %d: %s", next_ch, _COURSE_MGR.outline[next_ch])
            result = f"chapter_advanced → 第 {next_ch + 1} 章: {_COURSE_MGR.outline[next_ch]}"
        else:
            _LECTURE_BRAIN.end()
            logger.info("[advance_point] 全部讲完 → ended")
            return LECTURE_ENDED
    else:
        await _wait_tts_done(runtime)
        logger.info("[advance_point] TTS 播放完成，发送 Signal")

    active = _LECTURE_BRAIN.active_point
    next_text = active.get("text", "") if active else ""
    hint = f"advance_point → {result}。当前段落：{next_text}。请继续叙述。"
    signal = InputSignal().to_signal(
        priority=Priority.NOTICE,
        description=f"讲课推进: {_COURSE_MGR.name}",
        hint=hint,
    )
    runtime.matrix.session.add_signal(signal)
    logger.info("[advance_point] Signal 已发送, hint=%.80s", hint)
    return result


async def cmd_check_messages() -> str:
    return "0 messages (feishu not connected)"


async def cmd_resume_lecture() -> str:
    _LECTURE_BRAIN.resume()
    logger.info("[resume_lecture] → lecturing, chapter=%d", _LECTURE_BRAIN.current_chapter)
    return f"resumed at chapter {_LECTURE_BRAIN.current_chapter}"


async def cmd_chat_reply(text: str) -> str:
    msg = {"role": "ai", "text": text}
    async with _SSE_LOCK:
        _SSE_QUEUE.append(msg)
    _SSE_EVENT.set()
    logger.info("[chat_reply] SSE 推送 role=ai text=%.80s", text)
    return "ok"


# =========================== 注册 ===========================

def register_commands(chan, runtime: MossRuntime):
    """将所有命令注册到 PyChannel。"""

    @chan.build.command(
        name="start_prepare",
        doc="进入讨论大纲阶段。带 course 参数=开始备课，不带参数=重置回空闲",
        timeout=5.0, blocking=True,
    )
    async def _start_prepare(course: str = "") -> str:
        return await cmd_start_prepare(course)

    @chan.build.command(
        name="set_outline",
        doc="锁定课程大纲。course=课程名, title=课程标题, chapters=逗号分隔的章节标识列表",
        timeout=5.0, blocking=True,
    )
    async def _set_outline(course: str, title: str, chapters: str) -> str:
        return await cmd_set_outline(course, title, chapters)

    @chan.build.command(
        name="save_chapter",
        doc="存档当前章节。chapter=章节标识, locators=逗号分隔的图片 locator, speaker_notes=演讲者笔记 JSON。",
        timeout=10.0, blocking=True,
    )
    async def _save_chapter(chapter: str, locators: str = "", speaker_notes: str = "") -> str:
        return await cmd_save_chapter(chapter, locators, speaker_notes)

    @chan.build.command(
        name="load_course",
        doc="加载指定课程并继续备课。course=课程名",
        timeout=15.0, blocking=True,
    )
    async def _load_course(course: str) -> str:
        return await cmd_load_course(course)

    @chan.build.command(
        name="start_teaching",
        doc="讲课模式，开讲课程。course=课程名（未加载时必传），start_chapter=起始章节序号（默认 0）",
        timeout=15.0, blocking=True,
    )
    async def _start_teaching(course: str = "", start_chapter: int = 0) -> str:
        return await cmd_start_teaching(course, start_chapter, runtime)

    @chan.build.command(
        name="switch_chapter",
        doc="切换到指定章节序号（从 0 开始）",
        timeout=10.0, blocking=True,
    )
    async def _switch_chapter(index: int = 0) -> str:
        return await cmd_switch_chapter(index, runtime)

    @chan.build.command(
        name="advance_point",
        doc="标记当前讲完的要点为 done，推进到下一个。Ghost 在每段讲完后调用。"
            "等待 TTS 播放完成后发 Signal 唤醒 Ghost，确保音频不重叠。",
        timeout=120.0, blocking=True,
    )
    async def _advance_point() -> str:
        return await cmd_advance_point(runtime)

    @chan.build.command(
        name="check_messages",
        doc="检查飞书群消息并回复。Phase 1-2 为桩，返回无消息。",
        timeout=5.0, blocking=True,
    )
    async def _check_messages() -> str:
        return await cmd_check_messages()

    @chan.build.command(
        name="resume_lecture",
        doc="暂停后恢复讲课。Ghost 回答完打断问题后调用。",
        timeout=5.0, blocking=True,
    )
    async def _resume_lecture() -> str:
        return await cmd_resume_lecture()

    @chan.build.command(
        name="chat_reply",
        doc="回复聊天消息。text=回复正文",
        timeout=5.0, blocking=False,
    )
    async def _chat_reply(text: str) -> str:
        return await cmd_chat_reply(text)
