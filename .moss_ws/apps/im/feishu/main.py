"""Feishu IM Channel App for MOSS.

Push-pull architecture:
  - Push: Incoming messages → lightweight Signal (summary + metadata in description)
  - Pull: Ghost calls channel commands to fetch full messages / send replies

Uses lark-channel-sdk (high-level Feishu Channel SDK) for WebSocket transport,
event normalization, token management, and auto-reconnect.
"""
import asyncio
import janus
import logging
import os
import time
from collections import deque
from typing import Optional

from dotenv import load_dotenv
from lark_channel import (
    ChannelConfig,
    FeishuChannel,
    InboundMessage,
    OutboundText,
    SendOpts,
)

from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

from lark_channel.api.contact.v3.model.get_user_request import GetUserRequest
from lark_channel.channel.types import Identity as SDKIdentity

# ── Env ──────────────────────────────────────────────────────────────────────

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────


class FeishuConfig(ConfigType):
    """飞书 IM 集成配置。凭据通过环境变量注入 ($FEISHU_APP_ID 等)，ConfigStore 读取时自动解析。"""

    app_id: str
    app_secret: str
    domain: str = "https://open.feishu.cn"

    @classmethod
    def conf_name(cls) -> str:
        return "feishu"


# ── Message Buffer ────────────────────────────────────────────────────────────


class MessageBuffer:
    """Per-chat message buffer with dedup and reply tracking."""

    def __init__(self, max_per_chat: int = 100):
        self._chats: dict[str, deque[InboundMessage]] = {}
        self._seen: set[str] = set()
        self._replied: set[str] = set()
        self._max = max_per_chat
        self._log = logging.getLogger("moss.Feishu.Buffer")

    # ── internal helpers ──

    def _total(self) -> int:
        return sum(len(q) for q in self._chats.values())

    def _snapshot(self) -> str:
        return (
            f"chats={len(self._chats)} total={self._total()}"
            f" seen={len(self._seen)} replied={len(self._replied)}"
        )

    # ── public API ──

    def put(self, msg: InboundMessage) -> bool:
        """Store a message. Returns False if duplicate (already seen)."""
        if msg.id in self._seen:
            self._log.info("BUF_PUT DUP msg_id=%s %s", msg.id, self._snapshot())
            return False
        self._seen.add(msg.id)
        q = self._chats.setdefault(msg.chat_id, deque(maxlen=self._max))
        q.append(msg)
        self._log.info(
            "BUF_PUT NEW msg_id=%s chat_id=%s chat_msgs=%d %s",
            msg.id, msg.chat_id, len(q), self._snapshot(),
        )
        return True

    def pull(self, chat_id: str, limit: int = 20, before: Optional[str] = None) -> list[dict]:
        """Pull messages for a chat, most recent last."""
        q = self._chats.get(chat_id, deque())
        if not q:
            return []
        msgs = list(q)
        if before:
            msgs = [m for m in msgs if m.id < before]
        return [
            {
                "message_id": m.id,
                "chat_id": m.chat_id,
                "chat_type": m.chat_type,
                "sender_id": m.sender_id,
                "sender_name": m.sender_name or "",
                "content_type": m.raw_content_type or "unknown",
                "text": m.content_text,
                "mentioned_bot": m.mentioned_bot,
                "reply_to": m.reply_to_message_id or "",
            }
            for m in msgs[-limit:]
        ]

    def mark_replied(self, message_id: str) -> None:
        """Mark a message as replied."""
        self._replied.add(message_id)
        self._log.info(
            "BUF_REPLY msg_id=%s replied_set_size=%d %s",
            message_id, len(self._replied), self._snapshot(),
        )

    def is_replied(self, message_id: str) -> bool:
        """Check if a message has been replied to."""
        return message_id in self._replied

    def get_unreplied(self) -> list[InboundMessage]:
        """All unreplied messages (newest first), for context_messages display."""
        all_msgs: list[InboundMessage] = []
        for q in self._chats.values():
            for msg in q:
                if not self.is_replied(msg.id):
                    all_msgs.append(msg)
        result = sorted(all_msgs, key=lambda m: m.id, reverse=True)
        self._log.info(
            "BUF_UNREPLIED count=%d ids=%s %s",
            len(result), [m.id for m in result], self._snapshot(),
        )
        return result

# ── Signal construction ───────────────────────────────────────────────────────


def _build_signal_description(msg: InboundMessage) -> str:
    """Build a pipe-delimited description string from inbound message metadata.

    Format:
      飞书[{sender_name}][{chat_type_label}]: {text[:50]} | chat_id=xxx | msg_id=xxx | ...

    Ghost parses this string to extract routing info for pull/send commands.
    """
    sender = msg.sender_name or msg.sender_id
    text = msg.content_text[:50].replace("\n", " ")
    chat_type_label = {"p2p": "私聊", "group": "群聊"}.get(msg.chat_type, msg.chat_type)

    parts = [f"飞书[{sender}][{chat_type_label}]: {text}"]
    parts.append(f"chat_id={msg.chat_id}")
    parts.append(f"msg_id={msg.id}")
    parts.append(f"chat_type={msg.chat_type}")
    parts.append(f"sender_id={msg.sender_id}")
    if msg.mentioned_bot:
        parts.append("mentioned_bot=true")
    if msg.reply_to_message_id:
        parts.append(f"quoted_msg={msg.reply_to_message_id}")

    return " | ".join(parts)


# ── Channel Definition ────────────────────────────────────────────────────────

channel = new_channel(
    name="im_feishu",
    description=(
        "飞书 IM Channel — 接收飞书消息并发送回复。"
    ),
)

class AppState:
    """飞书 App 运行时状态。只存 SDK 独有的对象；logger/session 走 Matrix.discover() 进程单例。"""

    def __init__(self) -> None:
        self.buffer = MessageBuffer()
        self.fs: Optional[FeishuChannel] = None
        self.signal_queue: janus.Queue[InboundMessage] = janus.Queue(maxsize=50)

    @property
    def logger(self) -> logging.Logger:
        return Matrix.discover().logger

    @property
    def session(self):
        return Matrix.discover().session

_state = AppState()



@channel.build.command()
async def send_stream(chat_id: str, chunks__, reply_to: str = "") -> str:
    """流式回复飞书消息。模型边生成边在飞书卡片中实时刷新，适合长回复。
    短回复（1-2句）请优先使用 send_message。文本放 CDATA 内避免转义问题。

    :param chat_id: 目标会话 ID
    :param chunks__: 消息正文（开放-闭合标签间文本，MOSS CTML 实时推流）
    :param reply_to: 被回复的消息 ID，设置后内联回复该消息
    """
    if _state.fs is None:
        return "发送失败：飞书未连接"

    # Bridge MOSS chunks__ AsyncIterator → SDK MarkdownStreamController
    async def _producer(ctl) -> None:
        async for chunk in chunks__:
            await ctl.append(chunk)

    opts = SendOpts(reply_to=reply_to) if reply_to else None
    try:
        result = await _state.fs.stream(
            to=chat_id,
            spec={"markdown": _producer},
            opts=opts,
        )
    except Exception as e:
        _state.logger.error("STREAM_FAILED: chat_id=%s error=%s", chat_id, e)
        return f"飞书流式回复失败：{e}"

    if not result.success:
        error_msg = result.error.message if result.error else "未知错误"
        return f"飞书流式回复失败：{error_msg}"

    if reply_to:
        _state.buffer.mark_replied(reply_to)
    for m in _state.buffer.get_unreplied():
        if m.chat_id == chat_id:
            _state.buffer.mark_replied(m.id)
    return f"已流式回复到飞书 chat_id={chat_id} (msg_id={result.message_id})"


@channel.build.command(always_observe=True)
async def pull_messages(chat_id: str, limit: int = 20, before: str = "") -> list[dict]:
    """获取指定聊天更早的历史消息（当前新消息已自动展示在上下文中，通常无需调用此命令）。"""
    return _state.buffer.pull(chat_id, limit=limit, before=before or None)


@channel.build.command()
async def send_message(chat_id: str, text__: str, reply_to: str = "") -> str:
    """在飞书中回复用户。收到消息后优先使用此命令回复。如设置了 reply_to 则内联回复该消息。

    :param chat_id: 目标会话 ID
    :param text__: 消息正文（放在开闭标签之间，无需转义引号和特殊字符）
    :param reply_to: 被回复的消息 ID，设置后内联回复该消息
    """
    if _state.fs is None:
        return "发送失败：飞书未连接，请检查 App 运行状态"
    opts = SendOpts(reply_to=reply_to) if reply_to else None
    result = await _state.fs.send(
        to=chat_id,
        message=OutboundText(text=text__),
        opts=opts,
    )
    if result.success:
        if reply_to:
            _state.buffer.mark_replied(reply_to)
            _state.logger.info("CLEAR: marked reply_to=%s", reply_to)
        # Clear all unreplied for this chat — Ghost has responded
        for m in _state.buffer.get_unreplied():
            if m.chat_id == chat_id:
                _state.buffer.mark_replied(m.id)
                _state.logger.info("CLEAR: marked msg_id=%s for chat_id=%s", m.id, chat_id)
    if result.success:
        msg_id = result.message_id or "unknown"
        reply_info = f"，内联回复 msg_id={reply_to}" if reply_to else ""
        return f"已发送到飞书 chat_id={chat_id} (msg_id={msg_id}{reply_info})"
    else:
        error_msg = result.error.message if result.error else "未知错误"
        return f"飞书发送失败 chat_id={chat_id}：{error_msg}"


@channel.build.context_messages
async def _feishu_context() -> list[str]:
    """展示未回复的飞书消息，每个 think cycle 自动刷新。

    Ghost 无需调用任何命令即可看到待处理消息，
    直接使用消息中的 chat_id 调用 send_message 回复。
    """
    import time
    t0 = time.monotonic()
    unreplied = _state.buffer.get_unreplied()
    _state.logger.info("CTX: %d unreplied msgs: %s", len(unreplied), [m.id for m in unreplied])
    if not unreplied:
        _state.logger.info("CTX_RETURN: empty (no unreplied) t=%.3f", time.monotonic() - t0)
        return []

    # Connection status line
    conn_state = "未连接"
    bot_name = ""
    if _state.fs:
        snap = _state.fs.connection_snapshot()
        if snap and snap.state == "connected":
            conn_state = "已连接"
        identity = await _state.fs.resolve_bot_identity()
        if identity:
            bot_name = identity.name or identity.open_id

    header = f"[飞书 | {conn_state}]"
    if bot_name:
        header += f" | Bot: {bot_name}"

    lines = [header, "[飞书未回复消息]"]
    chat_type_label = {"p2p": "私聊", "group": "群聊"}
    for msg in unreplied:
        sender = msg.sender_name or msg.sender_id
        ctl = chat_type_label.get(msg.chat_type, msg.chat_type)
        lines.append(
            f"  [{ctl}] {sender} | chat_id:{msg.chat_id} | msg_id:{msg.id}"
            f"\n    {msg.content_text}"
            f'\n    → 回复: <apps.im_feishu:send_message chat_id="{msg.chat_id}" reply_to="{msg.id}">...</apps.im_feishu:send_message>'
        )
    result = ["\n".join(lines)]
    _state.logger.info(
        "CTX_RETURN: %d msgs conn=%s bot=%s len=%d t=%.3f",
        len(unreplied), conn_state, bot_name, len(result[0]), time.monotonic() - t0,
    )
    return result


@channel.build.instruction
async def instruction() -> str:
    return (
        "飞书消息回复规则【强约束】：\n"
        "1. 收到飞书消息后，必须通过 apps.im_feishu:send_message 或 apps.im_feishu:send_stream 回复用户\n"
        "2. 短回复（1-2句）→ send_message；长回复或需要卡片格式 → send_stream\n"
        "3. 直接从上下文中获取 chat_id 和 msg_id 作为参数，设置了 reply_to 则内联回复\n"
        "4. **禁止仅在终端输出——你的回复必须送达飞书用户**"
    )




# ── Event Handler (runs in SDK background thread) ─────────────────────────────


def _on_message(msg: InboundMessage) -> None:
    """Handle incoming message from Feishu SDK.

    Runs in SDK's background thread. Enqueues the message to a janus
    cross-thread queue; the async consumer on the main event loop builds
    the Signal and dispatches it to MOSS session.
    """
    # Dedup
    if not _state.buffer.put(msg):
        _state.logger.info("MSG_DUP: msg_id=%s chat_id=%s", msg.id, msg.chat_id)
        return

    _state.logger.info("MSG_IN: msg_id=%s chat_id=%s sender=%s text=%s",
        msg.id, msg.chat_id,
        msg.sender.display_name or msg.sender.open_id,
        msg.content_text[:60])

    try:
        _state.signal_queue.sync_q.put(msg)
    except janus.SyncQueueShutDown:
        pass  # shutting down, discard residual callback


def _on_error(error) -> None:
    _state.logger.error("Feishu SDK error: %s", error)


# ── Signal Consumer (runs on main event loop) ────────────────────────────────


async def _signal_consumer() -> None:
    """Consume InboundMessage from the janus cross-thread queue.

    Builds Signal body + description on the main loop and dispatches
    to MOSS session.  This replaces the call_soon_threadsafe bridge.
    """
    while True:
        try:
            msg: InboundMessage = await _state.signal_queue.async_q.get()
            description = _build_signal_description(msg)
            sender = msg.sender_name or msg.sender_id
            chat_type_label = {"p2p": "私聊", "group": "群聊"}.get(msg.chat_type, msg.chat_type)
            signal_text = (
                f"[飞书|{chat_type_label}|来自:{sender}|chat_id:{msg.chat_id}|msg_id:{msg.id}]"
                f"\n{msg.content_text}"
            )
            if _state.session:
                _state.session.add_input_signal(signal_text, description=description)
            _state.logger.info("Signal: %s", description[:120])
        except janus.QueueClosed:
            break
        except Exception:
            _state.logger.exception("SIGNAL_CONSUMER: failed to process message")


# ── Main Entry ────────────────────────────────────────────────────────────────


async def main(matrix: Matrix) -> None:
    logging.basicConfig(level=logging.INFO)

    # ── Fresh queue per run (previous run's close() makes old queue unusable) ──
    _state.signal_queue = janus.Queue(maxsize=50)

    # ── Start signal consumer (janus queue: SDK thread → main loop) ──
    consumer_task = asyncio.create_task(_signal_consumer())

    # ── Load config (app-scoped via cell workspace, minecraft_bot pattern) ──
    # Config file lives at apps/im/feishu/configs/feishu.yml alongside source.
    # $VAR placeholders resolved via .resolve() using os.environ (loaded by load_dotenv).
    config = matrix.cell_workspace.configs().read_yaml("feishu", FeishuConfig)
    if config is None:
        _state.logger.warning("No configs/feishu.yml found, falling back to env vars")
        config = FeishuConfig()
    config = config.resolve()
    # Resolve domain shorthand ("feishu"/"lark") to full URL; pass custom URLs through
    domain = config.domain
    domain = {
        "feishu": "https://open.feishu.cn",
        "lark": "https://open.larksuite.com",
    }.get(domain, domain)
    _state.logger.info(
        "Feishu app starting. app_id=%s domain=%s",
        config.app_id[:8] + "***" if config.app_id else "MISSING",
        domain,
    )

    # ── Sender name resolver (Open Claw pattern) ──
    # Per-user contact.v3.user.get with ID-type auto-detection, 10min cache,
    # and explicit permission error (code 99991672) handling.
    # See: .ai_partners/features/.../FEATURE.md §13.2
    _name_cache: dict[str, tuple[str, float]] = {}
    _NAME_CACHE_TTL = 600.0  # 10 minutes

    def _detect_id_type(sender_id: str) -> str:
        if sender_id.startswith("ou_"):
            return "open_id"
        if sender_id.startswith("on_"):
            return "union_id"
        return "user_id"

    async def _name_lookup(open_ids: list[str]) -> dict:

        now = time.monotonic()
        result: dict = {}
        unresolved: list[str] = []

        for oid in open_ids:
            if not oid:
                continue
            cached = _name_cache.get(oid)
            if cached and cached[1] > now:
                result[oid] = SDKIdentity(open_id=oid, display_name=cached[0])
            else:
                unresolved.append(oid)

        if not unresolved:
            return result

        client = _state.fs._client  # noqa: SLF001
        for oid in unresolved:
            try:
                id_type = _detect_id_type(oid)
                req = (
                    GetUserRequest.builder()
                    .user_id(oid)
                    .user_id_type(id_type)
                    .build()
                )
                resp = await client.contact.v3.user.aget(req)
                user = resp.data.user if resp.data else None
                name = None
                if user:
                    name = (
                        getattr(user, "name", None)
                        or getattr(user, "nickname", None)
                        or getattr(user, "en_name", None)
                    )
                if name:
                    _name_cache[oid] = (name, now + _NAME_CACHE_TTL)
                    result[oid] = SDKIdentity(open_id=oid, display_name=name)
                    _state.logger.info("NAME_LOOKUP: %s → %s", oid, name)
                else:
                    _state.logger.warning(
                        "NAME_LOOKUP: no name for %s (id_type=%s)", oid, id_type
                    )
            except Exception as e:
                code = getattr(e, "code", None)
                if code == 99991672:
                    _state.logger.error(
                        "NAME_LOOKUP_PERM: user lookup permission denied "
                        "(code=99991672). Grant contact:user:readonly AND publish "
                        "the app in Feishu Open Platform. Error: %s", e
                    )
                else:
                    _state.logger.error(
                        "NAME_LOOKUP_FAILED: open_id=%s error=%s", oid, e
                    )

        return result

    # ── Create SDK channel ──
    sdk_config = ChannelConfig(
        app_id=config.app_id,
        app_secret=config.app_secret,
        domain=domain,
    )
    _state.fs = FeishuChannel(config=sdk_config, name_lookup=_name_lookup)

    # ── Register handlers ──
    _state.fs.on("message", _on_message)
    _state.fs.on("error", _on_error)

    # ── Provide MOSS channel (non-blocking, vision app pattern) ──
    matrix.provide_channel(channel)

    # ── Connect (start WebSocket in background thread) ──
    try:
        await _state.fs.start_background(timeout=30.0)
    except Exception as e:
        _state.logger.error("Failed to connect to Feishu: %s", e)
        return

    identity = await _state.fs.resolve_bot_identity()
    if identity:
        _state.logger.info(
            "Bot connected: %s (open_id=%s)",
            identity.name or "unnamed",
            identity.open_id,
        )
    _state.logger.info("Feishu app ready — listening for messages")

    # ── Keep alive ──
    quit_event = asyncio.Event()
    try:
        await quit_event.wait()
    except asyncio.CancelledError:
        _state.logger.info("Feishu app cancelled, shutting down")
    finally:
        _state.fs.stop()
        _state.signal_queue.close()
        consumer_task.cancel()
        try:
            await consumer_task
        except (asyncio.CancelledError, janus.QueueClosed):
            pass
        _state.logger.info("Feishu app stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
