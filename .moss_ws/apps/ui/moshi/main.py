"""Moshi — AI 辅助内容交付产品原型。

Web 服务层：静态文件 + REST API。
不直接连接 Matrix，而是通过 moss() 的内部 HTTP 桥接 (:9733) 转发聊天消息。
"""

import json
import logging
from pathlib import Path

import aiohttp
from aiohttp import web

logger = logging.getLogger("moshi")

HOST = "127.0.0.1"
PORT = 9731
REFLEX_URL = "http://localhost:3000"
INTERNAL_BRIDGE = "http://127.0.0.1:9733"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ── API Handlers ──────────────────────────────────────────────────────────

async def handle_chat(request: web.Request) -> web.Response:
    """转发用户消息到 moss() 内部桥接 → 发 Signal 给 Ghost。"""
    data = await request.json()
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "idle")
    if not text:
        return web.json_response({"error": "empty text"}, status=400)

    logger.info("[proxy] 转发 chat_in → 内部桥接, text=%.60s", text)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{INTERNAL_BRIDGE}/_internal/chat_in",
                json={"text": text, "mode": mode},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[proxy] chat_in 失败: %s", body)
                    return web.json_response({"error": body}, status=502)
                logger.info("[proxy] chat_in 转发成功")
                return web.json_response({"ok": True})
    except aiohttp.ClientError as e:
        logger.error("[proxy] chat_in 连接失败: %s", e)
        return web.json_response({"error": "内部桥接不可用"}, status=503)


async def handle_chat_stream(request: web.Request) -> web.StreamResponse:
    """转发 moss() 内部桥接的 SSE 流到浏览器。"""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    logger.info("[proxy] SSE pipe 已连接 → 内部桥接")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{INTERNAL_BRIDGE}/_internal/chat_stream",
                timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
            ) as upstream:
                async for line in upstream.content:
                    await resp.write(line)
    except aiohttp.ClientError as e:
        logger.info("[proxy] SSE pipe 断开: %s", e)
    except (ConnectionResetError, ConnectionAbortedError):
        pass

    return resp


async def handle_subtitle_stream(request: web.Request) -> web.StreamResponse:
    """转发 moss() 内部桥接的字幕 SSE 流到浏览器。"""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    logger.info("[proxy] 字幕 SSE pipe 已连接 → 内部桥接")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{INTERNAL_BRIDGE}/_internal/subtitle_stream",
                timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
            ) as upstream:
                async for line in upstream.content:
                    await resp.write(line)
    except aiohttp.ClientError as e:
        logger.info("[proxy] 字幕 SSE pipe 断开: %s", e)
    except (ConnectionResetError, ConnectionAbortedError):
        pass

    return resp


async def handle_danmaku_stream(request: web.Request) -> web.StreamResponse:
    """转发 moss() 内部桥接的弹幕 SSE 流到浏览器。"""
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    logger.info("[proxy] 弹幕 SSE pipe 已连接 → 内部桥接")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{INTERNAL_BRIDGE}/_internal/danmaku_stream",
                timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
            ) as upstream:
                async for line in upstream.content:
                    await resp.write(line)
    except aiohttp.ClientError as e:
        logger.info("[proxy] 弹幕 SSE pipe 断开: %s", e)
    except (ConnectionResetError, ConnectionAbortedError):
        pass

    return resp


async def handle_state(request: web.Request) -> web.Response:
    return web.json_response({"mode": "idle", "status": "就绪"})


async def handle_courses(request: web.Request) -> web.Response:
    """从内部桥接获取课程列表（代理 moss() 的 /_internal/courses）。"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{INTERNAL_BRIDGE}/_internal/courses",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("[proxy] courses 失败: %s", body)
                    return web.json_response([], status=502)
                data = await resp.json()
                return web.json_response(data)
    except aiohttp.ClientError as e:
        logger.error("[proxy] courses 连接失败: %s", e)
        return web.json_response([], status=503)


async def handle_mode(request: web.Request) -> web.Response:
    data = await request.json()
    mode = data.get("mode", "idle")
    return web.json_response({
        "mode": mode,
        "status": f"切换到{mode}",
        "msg": f"切换到{mode}模式",
    })


async def handle_course_select(request: web.Request) -> web.Response:
    data = await request.json()
    name = data.get("name", "")
    return web.json_response({
        "mode": "idle",
        "status": f"已加载: {name}",
        "msg": f"已加载: {name}",
    })


async def handle_lecture_start(request: web.Request) -> web.Response:
    """原子编排讲课启动：转发到 moss() 内部桥接。"""
    data = await request.json()
    course = (data.get("course") or "").strip()
    if not course:
        return web.json_response({"error": "course is required"}, status=400)

    logger.info("[proxy] 转发 lecture_start → 内部桥接, course=%s", course)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{INTERNAL_BRIDGE}/_internal/lecture/start",
                json={"course": course},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                result = await resp.json()
                if resp.status != 200:
                    logger.warning("[proxy] lecture_start 失败: %s", result)
                    return web.json_response(result, status=502)
                logger.info("[proxy] lecture_start 成功: %s", result)
                return web.json_response(result)
    except aiohttp.ClientError as e:
        logger.error("[proxy] lecture_start 连接失败: %s", e)
        return web.json_response({"error": "内部桥接不可用"}, status=503)


async def handle_cmd(request: web.Request) -> web.Response:
    data = await request.json()
    cmd = data.get("cmd", "")
    return web.json_response({"msg": f"命令 {cmd} 已发送"})


# ── App ───────────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application()
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")
    app.router.add_get("/", lambda r: web.FileResponse(STATIC_DIR / "prepareL0.html"))
    app.router.add_get("/lecture", lambda r: web.FileResponse(STATIC_DIR / "teachingL0.html"))
    app.router.add_post("/api/state", handle_state)
    app.router.add_post("/api/courses", handle_courses)
    app.router.add_post("/api/mode", handle_mode)
    app.router.add_post("/api/course/select", handle_course_select)
    app.router.add_post("/api/cmd", handle_cmd)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/chat/stream", handle_chat_stream)
    app.router.add_get("/api/subtitle/stream", handle_subtitle_stream)
    app.router.add_get("/api/danmaku/stream", handle_danmaku_stream)
    app.router.add_post("/api/lecture/start", handle_lecture_start)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    app = create_app()
    logger.info(f"Moshi GUI → http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT, print=logger.info)


if __name__ == "__main__":
    main()
