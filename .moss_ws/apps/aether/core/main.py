"""Aether Core UI app — MOSS ghost 的全双工能量核心可视化通道。

后端不再把 listen/think/speak/interrupt 压成互斥五态，而是维护
并发 activity layers。``state`` 只作为主视觉基调向旧前端兼容：
listen、think、speak 可以同时为 true，interrupt 是短暂抢占层。
"""
import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import web, WSMsgType

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport
from ghoshell_moss.topics.audio import AudioRuntimeTopic, SpeechTopic

# Frontend static files live with the app so aether/core is self-contained.
WEB_ROOT = Path(__file__).resolve().parent / "webroot"
WS_PORT = 8765
INTERRUPT_HOLD = 0.7  # 急刹视觉保持时长（秒），超时回 idle

_log = logging.getLogger("moss.aether.core")


@web.middleware
async def _cross_origin_isolation_middleware(
    request: web.Request,
    handler,
) -> web.StreamResponse:
    response = await handler(request)
    # Keep the UI cross-origin isolated so future WebGL/AudioWorklet features
    # can use SharedArrayBuffer without changing the deployment surface.
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or _log
    logger.info("aether/core app starting, WEB_ROOT=%s", WEB_ROOT)

    transport = MatrixAudioTransport(matrix=matrix)
    speech_win = transport.topic_window(SpeechTopic, max_size=16)
    audio_win = transport.topic_window(AudioRuntimeTopic, max_size=16)

    clients: set = set()
    # 状态容器（避免 nonlocal 地狱）。state 是兼容字段；layers 是新契约。
    st = {
        "state": "idle",
        "layers": {
            "listen": False,
            "queue": False,
            "think": False,
            "speak": False,
            "interrupt": False,
        },
        "last_speech_ts": 0.0,
        "last_speaker_running": False,
        "last_asr_running": False,
        "interrupt_until": 0.0,
        "last_interrupt_started_at": 0.0,
        "_tts_end_at": 0.0,
        "think_started_at": 0.0,
        "queued_started_at": 0.0,
        "last_asr_diag_key": "",
        "last_vpio_diag_key": "",
        "asr_current": None,
        "asr_finals": [],
        "asr_error": None,
        "asr_control": {"mode": "continuous", "enabled": True},
        "vpio_diag": "",
    }
    # 初始化 last_speech_ts 为当前 window 的最大值，避免启动即触发历史
    init_speeches = list(speech_win.values())
    if init_speeches:
        st["last_speech_ts"] = max(t.timestamp for t in init_speeches)

    def _primary_state() -> str:
        layers = st["layers"]
        if layers["interrupt"]:
            return "interrupt"
        if layers["speak"]:
            return "speak"
        if layers["think"]:
            return "think"
        if layers["listen"]:
            return "listen"
        return "idle"

    def _snapshot(**extra) -> dict:
        st["state"] = _primary_state()
        msg = {
            "state": st["state"],
            "layers": dict(st["layers"]),
            "ts": time.time(),
            "diagnostics": {
                "asr_current": dict(st["asr_current"]) if st["asr_current"] else None,
                "asr_finals": list(st["asr_finals"]),
                "asr_error": dict(st["asr_error"]) if st["asr_error"] else None,
                "asr_control": dict(st["asr_control"]),
                "vpio": st["vpio_diag"],
            },
        }
        msg.update(extra)
        return msg

    async def broadcast(msg: dict) -> None:
        if not clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        dead = []
        for c in clients:
            try:
                await c.send_str(data)
            except Exception:
                dead.append(c)
        for c in dead:
            clients.discard(c)

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        logger.info("ws client connected, total=%d", len(clients))
        # 连接时先推当前状态
        await ws.send_str(json.dumps(_snapshot(), ensure_ascii=False))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except Exception:
                        continue
                    if payload.get("type") == "interrupt" or payload.get("state") == "interrupt":
                        st["interrupt_until"] = time.monotonic() + INTERRUPT_HOLD
                        st["layers"]["interrupt"] = True
                        st["layers"]["listen"] = False
                        st["layers"]["queue"] = False
                        st["layers"]["speak"] = False
                        st["layers"]["think"] = False
                        await broadcast(_snapshot(interrupt_burst=1.0))
                        logger.info("interrupt received from frontend")
                        transport.pub_topic(AudioRuntimeTopic(
                            running=True,
                            device_name="interrupt",
                            device_explain="frontend_manual_stop",
                            started_at=time.monotonic(),
                            last_heartbeat=time.monotonic(),
                        ))
                        logger.info("★ Frontend manual stop → audio interrupt topic published")
                        # 发 interrupt signal 到 ghost 主进程 (通过 Zenoh 跨进程)
                        # → mindflow.InterruptNucleus → FATAL impulse → shell.clear() → 停 TTS + 停 LLM
                        # aether/core 是独立子进程，不能直接访问 ghost 的 Mindflow，
                        # 必须通过 session.add_signal 走 Zenoh 发布。
                        try:
                            sig = new_interrupt_signal(
                                "立刻停下",
                                description="前端VAD检测到SPEAK中开口，触发shell.clear停TTS",
                            )
                            matrix.session.add_signal(sig)
                            logger.info("★ Interrupt signal sent via zenoh (shell.clear will stop TTS)")
                        except Exception as e:
                            logger.warning("Failed to send interrupt signal: %s", e)
                    elif payload.get("type") == "asr_control":
                        mode = str(payload.get("mode", "continuous")).strip().lower()
                        if mode not in {"continuous", "manual"}:
                            mode = "continuous"
                        enabled = bool(payload.get("enabled", mode == "continuous"))
                        if mode == "continuous":
                            enabled = True
                        st["asr_control"] = {"mode": mode, "enabled": enabled}
                        transport.pub_topic(AudioRuntimeTopic(
                            running=enabled,
                            device_name="asr_control",
                            device_explain=json.dumps(
                                {"source": "aether/core", "mode": mode, "enabled": enabled},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            started_at=time.monotonic(),
                            last_heartbeat=time.monotonic(),
                        ))
                        await broadcast(_snapshot(event="asr_control"))
                        logger.info("ASR control from frontend: mode=%s enabled=%s", mode, enabled)
                    elif payload.get("type") == "listen":
                        running = bool(payload.get("running"))
                        st["layers"]["listen"] = running
                        if payload.get("pending_think"):
                            st["layers"]["think"] = True
                            st["think_started_at"] = time.monotonic()
                        await broadcast(_snapshot(event="listen"))
                    elif payload.get("type") == "reset":
                        # 切 idle。TopicWindow 不提供 clear；用 last_speech_ts 跳过已有历史。
                        # 不发布 "/reset" SpeechTopic，否则 reset 会被当作一轮用户语音，
                        # 造成一次没有实际回复需求的假 think。
                        st["layers"] = {k: False for k in st["layers"]}
                        speeches = list(speech_win.values())
                        if speeches:
                            st["last_speech_ts"] = max(t.timestamp for t in speeches)
                        else:
                            st["last_speech_ts"] = time.monotonic()
                        st["think_started_at"] = 0
                        st["queued_started_at"] = 0
                        await broadcast(_snapshot())
                        logger.info("reset received from frontend — visual context cleared")
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            clients.discard(ws)
            logger.info("ws client disconnected, total=%d", len(clients))
        return ws

    async def index_handler(request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_ROOT / "index.html")

    app = web.Application(middlewares=[_cross_origin_isolation_middleware])
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web", path=str(WEB_ROOT / "web"), name="web")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WS_PORT)
    await site.start()
    logger.info("aether/core http+ws server on http://127.0.0.1:%d", WS_PORT)

    async def state_loop() -> None:
        while True:
            now = time.monotonic()
            now_ts = time.time()
            # interrupt 超时回退
            if st["layers"]["interrupt"] and now > st["interrupt_until"]:
                st["layers"]["interrupt"] = False
                await broadcast(_snapshot())

            # think 超时兜底：LLM 崩溃/超时时不卡在 think
            if st["layers"]["think"] and now - st.get("think_started_at", 0) > 30.0:
                st["layers"]["think"] = False
                st["layers"]["queue"] = False
                st["think_started_at"] = 0
                st["queued_started_at"] = 0
                await broadcast(_snapshot())
                logger.warning("think timeout 30s → %s", _primary_state())

            # 全双工打断：检测 listener 发布的"停下"关键词 interrupt 信号
            for t in reversed(list(audio_win.values())):
                if getattr(t, "device_name", "") == "interrupt" and t.running:
                    started_at = float(getattr(t, "started_at", 0.0) or 0.0)
                    if started_at > st["last_interrupt_started_at"]:
                        st["last_interrupt_started_at"] = started_at
                        st["layers"]["interrupt"] = True
                        st["layers"]["listen"] = False
                        st["layers"]["queue"] = False
                        st["layers"]["speak"] = False
                        st["layers"]["think"] = False
                        st["interrupt_until"] = now + INTERRUPT_HOLD
                        await broadcast(_snapshot(interrupt_burst=1.0))
                        logger.info("★ Wake word barge-in → interrupt (from listener)")
                    break

            # 检查 speech（用户说完一句 → think）
            speeches = list(speech_win.values())
            if speeches:
                latest = speeches[-1]
                if getattr(latest, "role", "") == "human" and latest.timestamp > st["last_speech_ts"]:
                    st["last_speech_ts"] = latest.timestamp
                    if not st["layers"]["interrupt"]:
                        queued = bool(st["layers"]["speak"])
                        st["layers"]["listen"] = False
                        if queued:
                            st["layers"]["queue"] = True
                            st["queued_started_at"] = now
                        st["layers"]["think"] = True
                        st["think_started_at"] = now
                        await broadcast(_snapshot(text=latest.text))
                        logger.info("%s: %s", "speech→queue+think" if queued else "speech→think", latest.text[:60])

            # 后端 ASR 活动（区别于浏览器本地 VAD）：ASR partial 出现才是真正
            # 进入后端听写链路。若 ASR 空等或浏览器误触，不能伪装成 think。
            asr_running = None
            asr_topic = None
            for t in reversed(list(audio_win.values())):
                if getattr(t, "device_name", "") == "asr":
                    asr_topic = t
                    asr_running = t.running
                    break
            if asr_topic is not None and asr_topic.running:
                explain = getattr(asr_topic, "device_explain", "") or ""
                diag_key = f"{getattr(asr_topic, 'started_at', 0)}:{explain}"
                if explain and diag_key != st["last_asr_diag_key"]:
                    st["last_asr_diag_key"] = diag_key
                    changed = False
                    try:
                        parsed = json.loads(explain)
                        if parsed.get("error"):
                            st["asr_error"] = {
                                "error": str(parsed.get("error")),
                                "code": str(parsed.get("code", "")),
                                "message": str(parsed.get("message", "")),
                                "backoff": float(parsed.get("backoff", 0) or 0),
                                "consecutive": int(parsed.get("consecutive", 0) or 0),
                                "ts": now_ts,
                            }
                            changed = True
                        text = str(parsed.get("text", "")).strip()
                        if text:
                            item = {
                                "text": text,
                                "final": bool(parsed.get("final")),
                                "ts": now_ts,
                            }
                            st["asr_error"] = None
                            if item["final"]:
                                st["asr_current"] = None
                                st["asr_finals"].append(item)
                                st["asr_finals"] = st["asr_finals"][-3:]
                            else:
                                st["asr_current"] = item
                            changed = True
                    except Exception:
                        st["asr_current"] = {"text": explain, "final": False, "ts": now_ts}
                        changed = True
                    if changed:
                        await broadcast(_snapshot(event="asr_diag"))
            elif asr_topic is not None and not asr_topic.running:
                explain = getattr(asr_topic, "device_explain", "") or ""
                diag_key = f"{getattr(asr_topic, 'started_at', 0)}:{explain}"
                if explain and diag_key != st["last_asr_diag_key"]:
                    st["last_asr_diag_key"] = diag_key
                    try:
                        parsed = json.loads(explain)
                    except Exception:
                        parsed = {}
                    if parsed.get("error"):
                        st["asr_error"] = {
                            "error": str(parsed.get("error")),
                            "code": str(parsed.get("code", "")),
                            "message": str(parsed.get("message", "")),
                            "backoff": float(parsed.get("backoff", 0) or 0),
                            "consecutive": int(parsed.get("consecutive", 0) or 0),
                            "ts": now_ts,
                        }
                        await broadcast(_snapshot(event="asr_error"))
            if asr_running is not None and asr_running != st["last_asr_running"]:
                st["last_asr_running"] = asr_running
                if asr_running:
                    if not st["layers"]["interrupt"]:
                        st["layers"]["listen"] = True
                        await broadcast(_snapshot(event="asr_listen"))
                        logger.info("ASR→listen")
                else:
                    if st["layers"]["listen"] and not st["layers"]["think"] and not st["layers"]["speak"]:
                        st["layers"]["listen"] = False
                        await broadcast(_snapshot(event="asr_idle"))
                        logger.info("ASR listen ended → idle")

            vpio_topic = None
            for t in reversed(list(audio_win.values())):
                if getattr(t, "device_name", "") == "vpio":
                    vpio_topic = t
                    break
            if vpio_topic is not None and vpio_topic.running:
                explain = getattr(vpio_topic, "device_explain", "") or ""
                diag_key = f"{getattr(vpio_topic, 'last_heartbeat', 0)}:{explain}"
                if explain and diag_key != st["last_vpio_diag_key"]:
                    st["last_vpio_diag_key"] = diag_key
                    st["vpio_diag"] = explain
                    await broadcast(_snapshot(event="vpio_diag"))

            # 检查 TTS speaker（running → speak，stopped → idle）
            running = None
            for t in reversed(list(audio_win.values())):
                if getattr(t, "device_name", "") == "speaker":
                    running = t.running
                    break
            if running is not None and running != st["last_speaker_running"]:
                st["last_speaker_running"] = running
                if running:
                    if not st["layers"]["interrupt"]:
                        st["_tts_end_at"] = 0.0
                        if st["layers"]["queue"]:
                            st["layers"]["queue"] = False
                            st["queued_started_at"] = 0
                        st["layers"]["speak"] = True
                        # LLM may continue preparing the next delta while TTS starts;
                        # leave think true briefly only if a new speech turn owns it.
                        await broadcast(_snapshot())
                        logger.info("TTS→speak")
                else:
                    # TTS 结束后加 800ms 保护期，让尾音播完，避免 VAD 误触 listen
                    if st["layers"]["speak"] or st["layers"]["interrupt"]:
                        st["_tts_end_at"] = now
                        logger.info("TTS ended → 800ms grace before idle")

            # TTS 结束保护期：800ms 后切 idle
            if st.get("_tts_end_at") and now - st["_tts_end_at"] > 0.8:
                if (st["layers"]["speak"] or st["layers"]["interrupt"]) and not st["last_speaker_running"]:
                    st["layers"]["speak"] = False
                    st["layers"]["interrupt"] = False
                    if not st["layers"]["queue"] and st.get("think_started_at", 0) <= st.get("_tts_end_at", 0):
                        st["layers"]["think"] = False
                    await broadcast(_snapshot())
                    logger.info("TTS grace end → %s", _primary_state())
                st["_tts_end_at"] = 0.0

            await asyncio.sleep(0.03)

    try:
        await state_loop()
    except asyncio.CancelledError:
        logger.info("aether/core cancelled")
    finally:
        await runner.cleanup()
        logger.info("aether/core stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
