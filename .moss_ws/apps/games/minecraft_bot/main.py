"""Minecraft Bot App — mineflayer-based channel for Ghost control.

This app bridges a Minecraft bot (via mineflayer + Python-JS bridge) into MOSS
as a standard Channel App. The bot connects to a Minecraft server and exposes
its actions as CTML commands.

Usage:
    # 1. Start a Minecraft server (see server/docker-compose.yml)
    cd server && docker compose up -d

    # 2. Test the app
    moss apps test games/minecraft_bot

    # 3. In CTML:
    <games_minecraft_bot:move x="10" y="64" z="20" />
    <games_minecraft_bot:reply text="Hello from MOSS!" />

Key constraints:
- The bot's replies to Minecraft chat MUST go through the `reply` command.
  Ghost instruction enforces this routing.
- Minecraft chat messages are reported as input Signals to Mindflow.
- Server connection state is managed explicitly; commands fail gracefully
  when disconnected.
"""

import asyncio
import atexit
import logging
import os
import signal
from typing import Any

import javascript
from javascript import On, require
from pydantic import Field

from ghoshell_moss.contracts.configs import ConfigType
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Signal, Priority
from ghoshell_moss.message import Message, Text

# ── JS bridge: mineflayer + plugins ──────────────────────────────────────────

mineflayer = require("mineflayer")
pathfinder = require("mineflayer-pathfinder")
blockfinder = require("mineflayer-blockfinder")

# ── Config ───────────────────────────────────────────────────────────────────


class MinecraftBotConfig(ConfigType):
    """Minecraft bot connection settings."""

    host: str = Field(default="127.0.0.1", description="Minecraft server host")
    port: int = Field(default=25565, description="Minecraft server port")
    bot_name: str = Field(default="Jarvis", description="Bot username in game")

    @classmethod
    def conf_name(cls) -> str:
        return "minecraft_bot"


# ── Module-level bot state ───────────────────────────────────────────────────

_bot = None
_bot_ready = False
_to_follow_player: str = ""
_bot_name: str = "Jarvis"

# Connection lifecycle events (JS thread → asyncio bridge)
_conn_event = asyncio.Event()
_conn_error: str | None = None

# Chat messages from JS thread → asyncio main loop
_chat_queue: asyncio.Queue[Message] = asyncio.Queue()


# ── Mineflayer lifecycle ─────────────────────────────────────────────────────


def _init_mineflayer(conf: MinecraftBotConfig, logger: logging.Logger) -> None:
    """Create bot, load plugins, wire JS events to Python asyncio."""
    global _bot, _bot_ready, _conn_error

    _bot = mineflayer.createBot({
        "host": conf.host,
        "port": conf.port,
        "username": conf.bot_name,
    })
    _bot.loadPlugin(pathfinder.pathfinder)
    _bot.loadPlugin(blockfinder)

    @On(_bot, "spawn")
    def _on_spawn() -> None:
        global _bot_ready
        movements = pathfinder.Movements(_bot)
        _bot.pathfinder.setMovements(movements)
        _bot_ready = True
        logger.info("Bot spawned, pathfinder ready")

    @On(_bot, "chat")
    def _on_chat(sender, message, *args) -> None:
        # Python-JS bridge 可能传递包装对象，强制转换为原生 str
        sender_str = str(sender) if sender else ""
        message_str = str(message) if message else ""

        # 调试日志：确认实际收到的 sender / message
        logger.info("chat event: sender=%r message=%r", sender_str, message_str)

        # 过滤空 sender、bot 自己发的消息、以及包含未解析翻译占位符的系统消息
        if not sender_str:
            logger.debug("filtered: empty sender")
            return
        if sender_str == conf.bot_name:
            logger.debug("filtered: self message from %s", sender_str)
            return
        if "<%s" in message_str:
            logger.debug("filtered: unformatted translation string")
            return

        msg = Message.new(name=sender_str).with_content(Text(text=f"[from minecraft] sender={sender_str}: {message_str}"))
        _chat_queue.put_nowait(msg)

    @On(_bot, "error")
    def _on_error(this, err) -> None:
        global _conn_error
        _conn_error = str(err)
        _conn_event.set()
        logger.error("Mineflayer error: %s", err)

    @On(_bot, "end")
    def _on_end(*args) -> None:
        global _conn_error, _bot_ready
        _bot_ready = False
        if _conn_error is None:
            _conn_error = "与服务器断开连接"
        _conn_event.set()
        logger.warning("Mineflayer connection ended")

    @On(_bot, "login")
    def _on_login(*args) -> None:
        _conn_event.set()
        logger.info("Bot logged in to server")


# ── Channel definition ───────────────────────────────────────────────────────

RANGE_GOAL = 1

chan = new_channel(
    name="minecraft",
    description="Minecraft 机器人控制通道 — 移动、挖掘、聊天、跟随玩家。",
)


@chan.build.instruction
def _minecraft_instruction() -> str:
    return (
        f"你是一个名为 {_bot_name} 的智能体，生活在 Minecraft 世界中。\n"
        "你可以感知周围环境、移动、挖掘、查找方块，并与玩家聊天。\n"
        "\n"
        "【重要】你的所有回复必须通过 <apps.games_minecraft_bot:reply>{你的消息}</apps.games_minecraft_bot:reply>"
        "发送到游戏内聊天栏。不要依赖默认输出通道，否则玩家看不到你的回复。\n"
        "\n"
        "你收到的 input 来自 Minecraft 聊天消息，你的 reply 会出现在同一个聊天栏。\n"
        "command 的 return 值只给你自己看（如坐标、执行结果），人机对话必须通过 reply。"
    )


# ── Commands ─────────────────────────────────────────────────────────────────


@chan.build.command()
async def move(x: int, y: int, z: int) -> str:
    """移动到指定坐标。

    Args:
        x: 目标 X 坐标
        y: 目标 Y 坐标
        z: 目标 Z 坐标
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器，无法移动。"
    try:
        _bot.pathfinder.setGoal(pathfinder.goals.GoalNear(x, y, z, RANGE_GOAL))
        return f"正在移动到 ({x}, {y}, {z})"
    except Exception as e:
        return f"移动失败：{e}"


@chan.build.command()
async def come(sender: str) -> str:
    """前往指定玩家的位置。

    Args:
        sender: 玩家名称
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        player = _bot.players[sender]
        target = player.entity
        if not target:
            return f"看不到玩家 {sender}"
        pos = target.position
        _bot.pathfinder.setGoal(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, RANGE_GOAL))
        return f"正在前往 {sender} 的位置"
    except Exception as e:
        return f"前往失败：{e}"


@chan.build.command(always_observe=True)
async def where_player_is(name: str) -> str:
    """获取指定玩家的位置。

    Args:
        name: 玩家名称
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        player = _bot.players[name]
        target = player.entity
        if not target:
            return f"找不到玩家 {name}"
        p = target.position
        return p.toString()
    except Exception as e:
        return f"获取位置失败：{e}"


@chan.build.command(always_observe=True)
async def find_blocks(block_name: str, max_distance: int = 128, count: int = 10) -> str:
    """查找附近指定类型的方块。

    Args:
        block_name: 方块名称（如 oak_log, diamond_ore）
        max_distance: 最大搜索距离，默认 128
        count: 最多返回数量，默认 10
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        if _bot.registry.blocksByName[block_name] is None:
            return f"{block_name} 不是有效的方块名称"
        ids = [_bot.registry.blocksByName[block_name].id]
        blocks = _bot.findBlocks({"matching": ids, "maxDistance": max_distance, "count": count})
        return f"找到 {blocks.length} 个 {block_name} 方块：{blocks}"
    except Exception as e:
        return f"查找失败：{e}"


@chan.build.command()
async def dig_under() -> str:
    """挖掘脚下的方块。"""
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        target = _bot.blockAt(_bot.entity.position.offset(0, -1, 0))
        if target and _bot.canDigBlock(target):
            _bot.chat(f"starting to dig {target.name}")
            _bot.dig(target)
            return f"挖掘了 {target.name}"
        return "脚下没有可挖掘的方块"
    except Exception as e:
        return f"挖掘失败：{e}"


@chan.build.command()
async def dig_target(x: int, y: int, z: int) -> str:
    """挖掘指定坐标的方块。

    Args:
        x: 目标 X 坐标
        y: 目标 Y 坐标
        z: 目标 Z 坐标
    """
    if not _bot_ready or _bot is None:
        return "未连接到服务器。"
    try:
        bp = _bot.entity.position
        target = _bot.blockAt(_bot.entity.position.offset(x - bp.x, y - bp.y, z - bp.z))
        if target and _bot.canDigBlock(target):
            _bot.lookAt(target.position)
            _bot.dig(target)
            return f"挖掘了 {target.name}"
        return "目标位置没有可挖掘的方块"
    except Exception as e:
        return f"挖掘失败：{e}"


@chan.build.command()
async def set_follow_player(sender: str) -> str:
    """设置跟随目标玩家。

    Args:
        sender: 玩家名称
    """
    global _to_follow_player
    _to_follow_player = sender
    return f"开始跟随玩家 {sender}"


@chan.build.command()
async def stop_follow_player() -> str:
    """停止跟随玩家。"""
    global _to_follow_player
    _to_follow_player = ""
    return "已停止跟随"


@chan.build.command()
async def reply(text__: str) -> None:
    """发送消息到 Minecraft 聊天栏。

    【这是你的主要输出通道】当你想回复玩家时，必须调用此命令。

    text__ 是纯文本字符串，必须在开闭标签间传递：
    <apps.games_minecraft_bot:reply>你的消息</apps.games_minecraft_bot:reply>

    Args:
        text__: 要发送到游戏内聊天栏的消息内容
    """
    if not _bot_ready or _bot is None:
        return
    _bot.chat(text__)


# ── Idle hook ────────────────────────────────────────────────────────────────


@chan.build.idle
async def on_idle() -> None:
    """持续跟随目标玩家。"""
    while True:
        await asyncio.sleep(0.5)
        if not _to_follow_player or not _bot_ready or _bot is None:
            continue
        try:
            player = _bot.players[_to_follow_player]
            target = player.entity
            if target:
                pos = target.position
                _bot.pathfinder.setGoal(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, RANGE_GOAL))
        except Exception:
            pass


# ── Context messages ─────────────────────────────────────────────────────────


@chan.build.context_messages
async def context_messages() -> list[Message]:
    """向 Ghost 报告当前状态。"""
    messages: list[Message] = []

    # 连接状态
    if not _bot_ready or _bot is None:
        messages.append(
            Message.new(name="__minecraft__").with_content(
                "【状态】未连接到 Minecraft 服务器。"
            )
        )
        return messages

    if _conn_error:
        messages.append(
            Message.new(name="__minecraft__").with_content(
                f"【状态】与服务器断开连接：{_conn_error}"
            )
        )
        return messages

    # 位置与周围环境
    try:
        pos = _bot.entity.position
        status_lines = [
            f"【位置】{pos.toString()}",
            f"【跟随】{_to_follow_player or '无'}",
            "【周围方块】",
        ]
        for x_off in range(-2, 3):
            for z_off in range(-2, 3):
                for y_off in range(-1, 3):
                    block = _bot.blockAt(pos.offset(x_off, y_off, z_off))
                    if block and block.name != "air":
                        status_lines.append(f"  {block.name} at {block.position.toString()}")
        messages.append(Message.new(name="__minecraft__").with_content("\n".join(status_lines)))
    except Exception as e:
        messages.append(
            Message.new(name="__minecraft__").with_content(f"【状态】获取环境信息失败：{e}")
        )

    return messages


# ── Background tasks ─────────────────────────────────────────────────────────


async def _chat_bridge_task(matrix: Matrix) -> None:
    """消费 JS 回调队列中的聊天消息，转为 input Signal 上报 Mindflow。"""
    logger = matrix.logger
    _last_key: str = ""  # 简单去重：相同 sender+content 的连续消息只发一次
    while True:
        msg = await _chat_queue.get()
        try:
            content = msg.to_content_string()
            dup_key = f"{msg.name}:{content}"
            if dup_key == _last_key:
                logger.info("Deduplicated duplicate chat: %s", msg.name)
                continue
            _last_key = dup_key

            sig = Signal.new(
                "input",
                msg,
                priority=Priority.NOTICE,
                description=f"chat from {msg.name}",
            )
            matrix.session.add_signal(sig)
            logger.info("Forwarded chat to Signal: %s", msg.name)
        except Exception:
            logger.exception("Failed to forward chat message to Signal")


async def _connection_monitor_task(matrix: Matrix) -> None:
    """监控连接状态，断开后上报故障 Signal。"""
    logger = matrix.logger
    global _conn_error

    # Wait for initial connection
    await _conn_event.wait()
    if _conn_error:
        logger.error("Initial connection failed: %s", _conn_error)
        return

    # Monitor for disconnections
    while True:
        await asyncio.sleep(5.0)
        if _conn_error and not _bot_ready:
            try:
                sig = Signal.new(
                    "input",
                    Message.new(name="__minecraft__").with_content(
                        f"【故障】与 Minecraft 服务器断开连接：{_conn_error}"
                    ),
                    priority=Priority.ERROR,
                    description="Minecraft bot disconnected",
                )
                matrix.session.add_signal(sig)
            except Exception:
                logger.exception("Failed to emit disconnect Signal")
            break


# ── App entry ────────────────────────────────────────────────────────────────


def _force_kill_node_bridge() -> None:
    """Backup cleanup: SIGKILL the javascript bridge if SIGTERM was ignored."""
    try:
        import javascript.connection as _js_conn

        if _js_conn.proc is not None and _js_conn.proc.poll() is None:
            os.kill(_js_conn.proc.pid, signal.SIGKILL)
    except Exception:
        pass


atexit.register(_force_kill_node_bridge)


async def main(matrix: Matrix) -> None:
    """Connect bot to server, start bridge tasks, register channel."""
    print(f"DEBUG: main() started pid={os.getpid()}", flush=True)
    logger = matrix.logger or logging.getLogger("minecraft_bot")
    logger.setLevel(logging.INFO)

    # Register signal handlers early so SIGTERM/SIGINT are caught even during
    # the potentially slow mineflayer initialisation phase.
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _on_sync_signal(signum: int, _frame: Any) -> None:
        logger.info("Received signal %d, scheduling cancel...", signum)
        loop.call_soon_threadsafe(main_task.cancel)

    signal.signal(signal.SIGTERM, _on_sync_signal)
    signal.signal(signal.SIGINT, _on_sync_signal)

    try:
        # ── Load config ──────────────────────────────────────────────────────
        conf = matrix.cell_workspace.configs().read_yaml("minecraft_bot", MinecraftBotConfig)
        if conf is None:
            conf = MinecraftBotConfig()
            logger.info("Using default config")
        else:
            logger.info("Loaded config from cell workspace")

        global _bot_name
        _bot_name = conf.bot_name

        logger.info(
            "Starting Minecraft bot: name=%s host=%s port=%d",
            conf.bot_name, conf.host, conf.port,
        )

        # ── Start background tasks FIRST so queue is consumed while connecting ─
        matrix.create_task(_chat_bridge_task(matrix), name="minecraft_chat_bridge")
        matrix.create_task(_connection_monitor_task(matrix), name="minecraft_conn_monitor")

        # ── Initialize mineflayer ────────────────────────────────────────────
        _init_mineflayer(conf, logger)

        # ── Wait for connection with timeout ─────────────────────────────────
        try:
            await asyncio.wait_for(_conn_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"无法连接到 Minecraft 服务器 {conf.host}:{conf.port}：连接超时（30s）。"
                f"请确认服务器已启动（docker compose up -d）"
            )

        if _conn_error:
            raise RuntimeError(
                f"无法连接到 Minecraft 服务器 {conf.host}:{conf.port}：{_conn_error}"
            )

        logger.info("Connected to Minecraft server successfully")

        # ── Register channel ─────────────────────────────────────────────────
        await matrix.provide_channel(chan)
        logger.info("Minecraft channel registered")

        # ── Keep alive until signalled or cancelled ──────────────────────────
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    finally:
        logger.info("Minecraft bot shutting down...")
        if _bot is not None:
            try:
                _bot.end()
                logger.info("Mineflayer bot disconnected")
            except Exception:
                logger.exception("Failed to disconnect mineflayer bot")
        try:
            javascript.terminate()
            logger.info("JavaScript bridge terminated")
        except Exception:
            logger.exception("Failed to terminate JavaScript bridge")
        # javascript.terminate() only sends SIGTERM; node may ignore it.
        # Ensure the bridge process is actually killed so it doesn't orphan.
        try:
            import javascript.connection as _js_conn

            if _js_conn.proc is not None and _js_conn.proc.poll() is None:
                os.kill(_js_conn.proc.pid, signal.SIGKILL)
                logger.info("Force-killed node bridge process (pid=%s)", _js_conn.proc.pid)
        except Exception:
            pass


if __name__ == "__main__":
    Matrix.discover().run(main)
