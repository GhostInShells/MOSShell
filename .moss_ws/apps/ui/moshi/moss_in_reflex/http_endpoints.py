"""内部 HTTP 桥接端点 — 供 main.py 发 Signal + 收 SSE。

create_internal_app() 创建 aiohttp Application 并注册所有路由。
"""

import asyncio
import json

from aiohttp import web as aiohttp_web

from ghoshell_moss import Message, Text
from ghoshell_moss.core.blueprint.mindflow import InputSignal

from framework.events import LayoutEvent
from framework.runtime.event_generator import _await_event

from moss_in_reflex.channel_commands import _switch_to_chapter
from moss_in_reflex.state import (
    logger,
    QUEUE,
    _COURSE_MGR, _LECTURE_BRAIN,
    _SSE_EVENT, _SSE_QUEUE, _SSE_LOCK,
    _SUBTITLE_EVENT, _SUBTITLE_QUEUE, _SUBTITLE_LOCK,
    _DANMAKU_EVENT, _DANMAKU_QUEUE, _DANMAKU_LOCK,
)


# =========================== SSE 流端点 ===========================

async def _chat_stream(request: aiohttp_web.Request):
    """SSE：监听 asyncio.Event，推送 AI 回复。"""
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
            await asyncio.wait_for(_SSE_EVENT.wait(), timeout=30)
            _SSE_EVENT.clear()
            async with _SSE_LOCK:
                while _SSE_QUEUE:
                    msg = _SSE_QUEUE.pop(0)
                    payload = json.dumps(msg, ensure_ascii=False)
                    await resp.write(f"data: {payload}\n\n".encode())
                    sent += 1
                    logger.info("[chat_stream] 推送给前端 role=%s text=%.60s",
                                msg.get("role"), msg.get("text"))
    except asyncio.TimeoutError:
        pass
    except (ConnectionResetError, ConnectionAbortedError):
        logger.info("[chat_stream] SSE 客户端断开, 共推送 %d 条", sent)
    except Exception:
        logger.exception("[chat_stream] 内部错误")
    return resp


async def _subtitle_stream(request: aiohttp_web.Request):
    """SSE: 推送字幕流。"""
    resp = aiohttp_web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)
    logger.info("[subtitle_stream] SSE 客户端连接")
    await resp.write(b": connected\n\n")
    try:
        while True:
            await asyncio.wait_for(_SUBTITLE_EVENT.wait(), timeout=30)
            _SUBTITLE_EVENT.clear()
            async with _SUBTITLE_LOCK:
                while _SUBTITLE_QUEUE:
                    msg = _SUBTITLE_QUEUE.pop(0)
                    payload = json.dumps(msg, ensure_ascii=False)
                    await resp.write(f"data: {payload}\n\n".encode())
    except asyncio.TimeoutError:
        pass
    except (ConnectionResetError, ConnectionAbortedError):
        logger.info("[subtitle_stream] SSE 客户端断开")
    except Exception:
        logger.exception("[subtitle_stream] 内部错误")
    return resp


async def _danmaku_stream(request: aiohttp_web.Request):
    """SSE: 推送弹幕流。"""
    resp = aiohttp_web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)
    logger.info("[danmaku_stream] SSE 客户端连接")
    await resp.write(b": connected\n\n")
    try:
        while True:
            await asyncio.wait_for(_DANMAKU_EVENT.wait(), timeout=30)
            _DANMAKU_EVENT.clear()
            async with _DANMAKU_LOCK:
                while _DANMAKU_QUEUE:
                    msg = _DANMAKU_QUEUE.pop(0)
                    payload = json.dumps(msg, ensure_ascii=False)
                    await resp.write(f"data: {payload}\n\n".encode())
    except asyncio.TimeoutError:
        pass
    except (ConnectionResetError, ConnectionAbortedError):
        logger.info("[danmaku_stream] SSE 客户端断开")
    except Exception:
        logger.exception("[danmaku_stream] 内部错误")
    return resp


async def _subtitle_in(request: aiohttp_web.Request):
    """接收来自 Speech 进程的字幕数据（HTTP 旁路），推入 SSE 队列。"""
    try:
        data = await request.json()
        text = (data.get("text") or "").strip()
        is_final = data.get("is_final", False)
        if text or is_final:
            async with _SUBTITLE_LOCK:
                _SUBTITLE_QUEUE.append(
                    {"type": "full" if is_final else "chunk", "text": text}
                )
            _SUBTITLE_EVENT.set()
    except Exception:
        pass  # 字幕非关键，静默吞错
    return aiohttp_web.json_response({"ok": True})


# =========================== Runtime-dependent 端点 ===========================

def _create_chat_in(matrix):
    """main.py 转发用户消息 → 发 Signal 唤醒 Ghost。"""
    async def handler(request: aiohttp_web.Request):
        data = await request.json()
        text = (data.get("text") or "").strip()
        if not text:
            return aiohttp_web.json_response({"error": "empty text"}, status=400)
        signal = InputSignal().to_signal(
            Message.new(tag="chat-input").with_content(Text(text=text)),
            description=f"聊天: {text[:50]}",
        )
        matrix.session.add_signal(signal)
        logger.info("[chat_in] Signal 已发送, text=%.60s", text)
        return aiohttp_web.json_response({"ok": True})
    return handler


def _create_courses(course_storage):
    """返回课程列表，供 main.py 代理。"""
    async def handler(request: aiohttp_web.Request):
        infos = await course_storage.list_infos()
        courses = [
            {
                "name": info.path.rsplit("/", 1)[0] if "/" in info.path else info.path,
                "title": info.title,
                "chapters_count": info.chapters_count,
                "locator": info.locator,
            }
            for info in infos
        ]
        return aiohttp_web.json_response(courses)
    return handler


def _create_lecture_start(matrix, course_storage, runtime):
    """原子编排：加载+布局+渲染+状态机+唤醒Ghost。"""
    async def handler(request: aiohttp_web.Request):
        data = await request.json()
        course = (data.get("course") or "").strip()
        if not course:
            return aiohttp_web.json_response({"error": "course is required"}, status=400)

        logger.info("[lecture_start] 开始编排课程 %s", course)

        # 1. 加载课程
        if not _COURSE_MGR.is_loaded or _COURSE_MGR.name != course:
            await _COURSE_MGR.load_course(course, course_storage)

        # 2. 强制切到 lesson 布局
        await _await_event(QUEUE, LayoutEvent(layout="lesson"))
        logger.info("[lecture_start] 布局 → lesson")

        # 3. 进入讲课模式 + 渲染首章
        start_chapter = data.get("start_chapter", 0)
        _COURSE_MGR.start_teaching(start_chapter)
        await _switch_to_chapter(start_chapter, runtime)

        # 4. 初始化讲课状态机
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

        # 5. 发 Signal 唤醒 Ghost
        ready_msg = (
            f"课程《{_COURSE_MGR.name}》已就绪。"
            f"当前第 1/{len(_COURSE_MGR.outline)} 章：{_COURSE_MGR.outline[0]}。"
            f"布局已自动设为 lesson。请按 speaker_notes 开始叙述。"
        )
        signal = InputSignal().to_signal(
            Message.new(tag="lecture-ready").with_content(Text(text=ready_msg)),
            description=f"讲课就绪: {_COURSE_MGR.name}",
        )
        matrix.session.add_signal(signal)

        logger.info("[lecture_start] Signal 已发送, course=%s, chapter=%d, points=%d",
                    _COURSE_MGR.name, start_chapter, len(_LECTURE_BRAIN.points))

        return aiohttp_web.json_response({
            "ok": True,
            "course": _COURSE_MGR.name,
            "title": _COURSE_MGR.title or _COURSE_MGR.name,
            "chapter": start_chapter,
            "total_chapters": len(_COURSE_MGR.outline),
            "chapter_name": _COURSE_MGR.outline[start_chapter] if _COURSE_MGR.outline else "",
        })
    return handler


# =========================== App 创建 ===========================

def create_internal_app(matrix, course_storage, runtime) -> aiohttp_web.Application:
    """创建内部 HTTP 桥接应用，注册所有路由。

    返回已注册好全部端点的 aiohttp_web.Application（尚未启动），
    由调用方负责 AppRunner + TCPSite 生命周期。
    """
    app = aiohttp_web.Application()

    # Signal 输入
    app.router.add_post("/_internal/chat_in", _create_chat_in(matrix))

    # SSE 流（纯模块级状态，无需闭包）
    app.router.add_get("/_internal/chat_stream", _chat_stream)
    app.router.add_get("/_internal/subtitle_stream", _subtitle_stream)
    app.router.add_get("/_internal/danmaku_stream", _danmaku_stream)

    # HTTP 旁路字幕输入
    app.router.add_post("/_internal/subtitle_in", _subtitle_in)

    # 课程 & 讲课
    app.router.add_get("/_internal/courses", _create_courses(course_storage))
    app.router.add_post("/_internal/lecture/start",
                        _create_lecture_start(matrix, course_storage, runtime))

    return app
