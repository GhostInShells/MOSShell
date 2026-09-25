"""Probe #2 — can an MV3 service worker hold a WebSocket open?

The whole comms design of ghost-in-bilibili rests on one browser behaviour:
does Chrome's background service worker stay alive while a WS is open, or does
it get reaped after ~30s idle? This probe answers that, and three smaller
questions from the design doc's "未验证声明".

It answers nothing about ghost-in-bilibili itself — it is throwaway.

Run:  .venv/bin/python server.py
Then: load extension/ as an unpacked extension in Chrome, open a bilibili video
      page, and leave it alone for 10+ minutes (refresh the tab once, open a
      second tab). Every event lands on stdout and in probe.log.

Kill with Ctrl-C to get the summary.
"""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.protocol import State

HOST = "127.0.0.1"
PORT = 23881
"""23881, not 23880 — the old probe's extension may still be loaded and
pointing at 23880."""

PING_EVERY = 20.0
"""Application-level ping. Chrome's 30s idle timer counts WebSocket *activity*,
but a protocol-level ping frame is handled by the network stack — it may not
reach the service worker's JS at all. An application-level message does. If the
connection survives at 20s cadence, this is the mechanism; to isolate protocol
from application pings, re-run with APP_PING = None."""

APP_PING: float | None = PING_EVERY

LOG = Path(__file__).resolve().parent / "probe.log"


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(line: str) -> None:
    text = f"[{stamp()}] {line}"
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


class Session:
    def __init__(self, connection: ServerConnection) -> None:
        self.conn = connection
        self.opened = asyncio.get_running_loop().time()
        self.pongs = 0
        self.pings = 0
        self.frames = 0
        self.name = "?"
        self.tasks: list[asyncio.Task] = []

    def alive_for(self) -> float:
        return asyncio.get_running_loop().time() - self.opened


SESSIONS: dict[int, Session] = {}

MILESTONES = (60.0, 180.0, 300.0, 600.0, 900.0)


async def _milestones(s: Session) -> None:
    """Announce survival at intervals — the whole point is a readable timeline."""
    last = 0.0
    for mark in MILESTONES:
        await asyncio.sleep(mark - last)
        last = mark
        if s.conn.state is not State.OPEN:
            return
        log(f"★ {s.name} 存活 {mark:.0f}s (frames={s.frames} pings={s.pings} pongs={s.pongs})")
    log(f"★★ {s.name} 存活超过 {MILESTONES[-1]:.0f}s — 心跳续命有效,SW 没被回收")


async def _app_ping(s: Session) -> None:
    if APP_PING is None:
        return
    while s.conn.state is State.OPEN:
        await asyncio.sleep(APP_PING)
        if s.conn.state is not State.OPEN:
            return
        s.pings += 1
        try:
            await s.conn.send(json.dumps({"type": "ping", "n": s.pings}))
        except Exception as e:
            log(f"!! {s.name} app-ping 发送失败: {e!r}")
            return


async def handler(connection: ServerConnection) -> None:
    s = Session(connection)
    SESSIONS[id(connection)] = s
    s.tasks = [
        asyncio.create_task(_milestones(s)),
        asyncio.create_task(_app_ping(s)),
    ]
    try:
        async for raw in connection:
            s.frames += 1
            try:
                msg = json.loads(raw)
            except Exception:
                log(f"<- {s.name} 非 JSON: {raw[:200]!r}")
                continue
            kind = msg.get("type")
            if kind == "hello":
                s.name = msg.get("session", "?")
                log(
                    f"<- hello session={s.name} boot={msg.get('boot', '?')} "
                    f"ua={msg.get('ua', '')[:50]!r}"
                )
            elif kind == "pong":
                s.pongs += 1
                if s.pongs <= 3 or s.pongs % 15 == 0:
                    log(f"<- pong #{s.pongs} (存活 {s.alive_for():.0f}s)")
            elif kind == "log":
                # The SW buffers its own console lines and flushes them on
                # reconnect — this is how we see what happened while the node
                # was not connected.
                for line in msg.get("lines", [])[:60]:
                    log(f"  [sw] {line}")
            else:
                log(f"<- {s.name} {kind}: {json.dumps(msg, ensure_ascii=False)[:300]}")
    except Exception as e:
        log(f"!! {s.name} 读取异常: {e!r}")
    finally:
        log(f"-- {s.name} 断开 (存活 {s.alive_for():.0f}s, frames={s.frames})")
        for t in s.tasks:
            t.cancel()
        SESSIONS.pop(id(connection), None)


BROWSER_HIT = """这个端口只讲两件事:

  - WebSocket 握手(扩展的 service worker 连过来)
  - GET /probe     (内容脚本的跨域试探)

你现在能在浏览器里读到这段话,说明请求不是上面任何一种 —— 通常是有人直接
打开了这个地址。这**不说明扩展的状态**,别把它当故障。

扩展是否连上,看服务端有没有这两行:
  *  握手 ... origin='chrome-extension://...'
  <- hello session=...

没有就是没连上。排查顺序见 probe/README.md。
"""


def process_request(connection: ServerConnection, request):
    """Log the handshake; answer plain GET /probe so the content script's
    cross-origin attempt gets a real CORS-enabled response instead of a
    protocol error (otherwise "blocked" and "server refused" look identical).

    The Origin here is the empirical answer to "what must the allowlist contain"
    — websockets' own `serve(origins=[...])` can enforce it once we know.
    """
    # 必须走 Headers 自己的大小写不敏感查找 —— 转成普通 dict 再 .get("Origin")
    # 会漏掉 wire 上实际发的 "origin",把结论反着报。
    headers = request.headers
    origin = headers.get("Origin", "<none>")
    ua = headers.get("User-Agent", "")[:80]

    if request.path.startswith("/probe"):
        log(f"*  /probe 试探 origin={origin!r} ua={ua!r}")
        response = connection.respond(200, json.dumps({"ok": True, "from": "node"}))
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Content-Type"] = "application/json"
        return response

    if headers.get("Upgrade", "").lower() != "websocket":
        # 普通 HTTP —— 回一段人话,别让它变成 InvalidUpgrade 的 traceback。
        log(f"*  非 WS 请求 path={request.path} ua={ua!r} → 多半是浏览器直接打开了这个地址,"
            f"与扩展状态无关")
        response = connection.respond(200, BROWSER_HIT)
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        return response

    log(f"*  握手 path={request.path} origin={origin!r}")
    log(f"      ua={ua!r}")
    if origin == "<none>":
        log("      !! 没有 Origin 头 —— Origin 白名单这条路可能不成立,记下来")
    return None


async def main() -> None:
    global APP_PING, PORT
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--no-app-ping",
        action="store_true",
        help=(
            "只留协议层 ping(20s)。用来分离「协议流量是否足以续命」——"
            "若这次连接会断,而带 app ping 那一次不断,说明必须让消息抵达 JS。"
        ),
    )
    args = parser.parse_args()
    PORT = args.port
    if args.no_app_ping:
        APP_PING = None

    LOG.write_text("", encoding="utf-8")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-unix
            pass

    async with serve(handler, HOST, PORT, process_request=process_request):
        log(f"listening ws://{HOST}:{PORT}  (app-ping {APP_PING}s)")
        log("下一步:加载 extension/ 为「已解压的扩展程序」,打开 B 站视频页。")
        await stop.wait()

    log("=== 汇总 ===")
    if not SESSIONS:
        log("没有任何连接。检查:扩展是否加载、页面是否是 bilibili.com/video/*。")
    for s in SESSIONS.values():
        log(f"{s.name}: 存活 {s.alive_for():.0f}s frames={s.frames} pings={s.pings} pongs={s.pongs}")


if __name__ == "__main__":
    asyncio.run(main())
