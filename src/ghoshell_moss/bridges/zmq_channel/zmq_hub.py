"""
ZMQ Hub — 基于 ROUTER/DEALER 的动态节点注册与发现。

Hub 绑定 ROUTER socket 在知名地址，子节点通过 DEALER 连接注册。
registered_nodes() 即时返回缓存，零阻塞。

Registry 协议 (JSON over ZMQ multipart):
  DEALER → ROUTER (multipart: [b"", json_bytes]):
    {"action": "register",   "name": "...", "channel_address": "...", "description": "..."}
    {"action": "unregister", "name": "..."}
    {"action": "heartbeat",  "name": "..."}
    {"action": "query"}

  ROUTER → DEALER (multipart: [identity, b"", json_bytes]):
    {"status": "ok"}
    {"status": "ok", "nodes": [...]}
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

try:
    import zmq
    import zmq.asyncio
except ImportError:
    raise ImportError("zmq module not found, please pip install ghoshell-moss[zmq]")

from ghoshell_moss.core.concepts.channel import Channel, ChannelName
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state, ChannelState
from ghoshell_moss.bridges.zmq_channel.zmq_channel import ZMQChannelProxy
from ghoshell_moss.contracts import LoggerItf

__all__ = [
    "ZMQHub",
    "ZMQHubChannelState",
    "NodeInfo",
    "ManagedProcess",
    "zmq_register",
    "zmq_unregister",
    "zmq_query",
]


# ------------------------------------------------------------------
# NodeInfo — 注册节点信息
# ------------------------------------------------------------------

@dataclass
class NodeInfo:
    """注册在 ZMQHub 上的节点信息。"""
    name: str
    channel_address: str
    description: str = ""
    registered_at: float = field(default_factory=time.time)


# ------------------------------------------------------------------
# ManagedProcess — 子进程管理器 (保留自 alpha)
# ------------------------------------------------------------------

class ManagedProcess:
    """异步子进程资源管理器。实现上下文管理器协议，确保退出时进程一定被关闭。"""

    def __init__(self, name: str, script_path: str, env: dict, logger: logging.Logger):
        self.name = name
        self.script_path = script_path
        self.env = env
        self.logger = logger

        self.process: Optional[asyncio.subprocess.Process] = None
        self.start_time: float = 0.0
        self._monitor_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self.logger.info("--- 启动子进程: %s", self.name)
        self.start_time = time.time()

        creationflags = 0
        start_new_session = False

        if sys.platform == "win32":
            creationflags = asyncio.subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            start_new_session = True

        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            self.script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

        self._monitor_task = asyncio.create_task(self._monitor_logs())
        return self

    async def _monitor_logs(self):
        async def read_stream(stream, level):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="ignore").rstrip()
                self.logger.log(level, "[%s] %s", self.name, decoded)

        try:
            await asyncio.gather(
                read_stream(self.process.stdout, logging.INFO),
                read_stream(self.process.stderr, logging.ERROR),
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("监控子进程 %s 日志时出错", self.name)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._monitor_task:
            self._monitor_task.cancel()

        if not self.process or self.process.returncode is not None:
            return

        self.logger.info("--- 正在关闭子进程: %s (PID: %s)", self.name, self.process.pid)

        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
                self.logger.info("子进程 %s 已优雅退出", self.name)
            except asyncio.TimeoutError:
                self.logger.warning("子进程 %s 响应超时，正在强制关闭...", self.name)
                if sys.platform == "win32":
                    self.process.kill()
                else:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await self.process.wait()
                self.logger.info("子进程 %s 已强制关闭", self.name)

        except Exception:
            self.logger.exception("关闭子进程 %s 时发生错误", self.name)

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    @property
    def is_running(self) -> bool:
        if not self.process:
            return False
        if self.process.returncode is not None:
            return False
        try:
            return psutil.pid_exists(self.process.pid)
        except Exception:
            return False


# ------------------------------------------------------------------
# ZMQHub — 动态节点注册/发现
# ------------------------------------------------------------------

class ZMQHub:
    """
    ZMQ 动态 Hub — 节点自注册，自动发现。

    使用方式:
        hub = ZMQHub(name="my-hub", registry_address="ipc:///tmp/moss-zmq-hub.sock")
        async with hub:
            # 查看已注册节点
            nodes = hub.registered_nodes()

            # 为节点创建 proxy
            proxy = hub.create_proxy("node-a")
            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                # ...

            # 或集成到 Shell
            channel = hub.as_channel()
    """

    DEFAULT_REGISTRY_ADDRESS = "ipc:///tmp/moss-zmq-hub.sock"

    def __init__(
        self,
        name: str,
        registry_address: str | None = None,
        *,
        logger: LoggerItf | None = None,
        heartbeat_timeout: float | None = None,
    ):
        """
        :param name: Hub 名称
        :param registry_address: ROUTER socket 绑定地址。
                                 默认 ipc:///tmp/moss-zmq-hub-{name}.sock
        :param logger: 日志接口
        :param heartbeat_timeout: 心跳超时秒数。None 表示不启用超时清理。
        """
        self._name = name
        self._registry_addr = registry_address or f"ipc:///tmp/moss-zmq-hub-{name}.sock"
        self._logger = logger or logging.getLogger(f"zmq_hub.{name}")
        self._heartbeat_timeout = heartbeat_timeout

        self._ctx = zmq.asyncio.Context.instance()
        self._router: Optional[zmq.asyncio.Socket] = None
        self._nodes: dict[str, NodeInfo] = {}
        self._nodes_lock = threading.Lock()
        self._registry_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = threading.Event()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动 hub registry。"""
        if self._router is not None:
            return

        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.bind(self._registry_addr)
        self._logger.debug("ZMQHub '%s' ROUTER bound to %s", self._name, self._registry_addr)

        loop = asyncio.get_running_loop()
        self._registry_task = loop.create_task(self._registry_loop())
        if self._heartbeat_timeout is not None:
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """停止 hub，关闭所有连接。"""
        if self._closed.is_set():
            return
        self._closed.set()

        for task in [self._registry_task, self._cleanup_task]:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._registry_task = None
        self._cleanup_task = None

        if self._router is not None:
            self._router.close(linger=0)
            self._router = None

        self._logger.debug("ZMQHub '%s' stopped", self._name)

    async def __aenter__(self) -> "ZMQHub":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Registry Loop
    # ------------------------------------------------------------------

    async def _registry_loop(self) -> None:
        """后台: ROUTER socket 接收注册/注销/心跳/查询消息。"""
        while not self._closed.is_set():
            try:
                frames = await self._router.recv_multipart()
            except zmq.ZMQError:
                if self._closed.is_set():
                    return
                continue

            if len(frames) < 3:
                continue

            identity = frames[0]
            # frames[1] is the empty delimiter frame from DEALER
            data_bytes = frames[2]

            try:
                msg = json.loads(data_bytes)
            except json.JSONDecodeError:
                self._logger.warning("ZMQHub '%s' bad JSON", self._name)
                continue

            action = msg.get("action", "")
            node_name = msg.get("name", "")

            if action == "register":
                if not node_name:
                    continue
                channel_addr = msg.get("channel_address", "")
                description = msg.get("description", "")
                with self._nodes_lock:
                    self._nodes[node_name] = NodeInfo(
                        name=node_name,
                        channel_address=channel_addr,
                        description=description,
                    )
                self._logger.info(
                    "ZMQHub '%s': node '%s' registered (addr=%s)",
                    self._name, node_name, channel_addr,
                )
                await self._send_response(identity, {"status": "ok"})

            elif action == "unregister":
                if not node_name:
                    continue
                with self._nodes_lock:
                    self._nodes.pop(node_name, None)
                self._logger.info("ZMQHub '%s': node '%s' unregistered", self._name, node_name)
                await self._send_response(identity, {"status": "ok"})

            elif action == "heartbeat":
                if not node_name:
                    continue
                with self._nodes_lock:
                    info = self._nodes.get(node_name)
                    if info is not None:
                        info.registered_at = time.time()

            elif action == "query":
                with self._nodes_lock:
                    nodes_data = [
                        {
                            "name": info.name,
                            "description": info.description,
                            "channel_address": info.channel_address,
                        }
                        for info in self._nodes.values()
                    ]
                await self._send_response(identity, {"status": "ok", "nodes": nodes_data})

    async def _send_response(self, identity: bytes, msg: dict) -> None:
        """通过 ROUTER 向指定 identity 的 DEALER 发送响应。"""
        if self._router is None:
            return
        try:
            await self._router.send_multipart([
                identity,
                b"",
                json.dumps(msg).encode(),
            ])
        except zmq.ZMQError:
            pass

    async def _cleanup_loop(self) -> None:
        """后台: 定期清理超时节点。"""
        while not self._closed.is_set():
            try:
                await asyncio.sleep(max(self._heartbeat_timeout / 2, 1.0))
            except asyncio.CancelledError:
                return

            now = time.time()
            with self._nodes_lock:
                stale = [
                    name for name, info in self._nodes.items()
                    if now - info.registered_at > self._heartbeat_timeout
                ]
                for name in stale:
                    del self._nodes[name]
                    self._logger.info(
                        "ZMQHub '%s': node '%s' timed out", self._name, name,
                    )

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def registered_nodes(self) -> dict[str, NodeInfo]:
        """即时返回所有已注册节点（非阻塞）。"""
        with self._nodes_lock:
            return dict(self._nodes)

    def node_info(self, name: str) -> NodeInfo | None:
        """查询单个节点信息。"""
        with self._nodes_lock:
            return self._nodes.get(name)

    def create_proxy(
        self,
        node_name: str,
        *,
        proxy_name: str | None = None,
        description: str = "",
    ) -> ZMQChannelProxy:
        """
        为已注册节点创建 ZMQChannelProxy。

        :param node_name: 已注册的节点名
        :param proxy_name: proxy 的 channel 名称，默认同 node_name
        :param description: proxy 描述
        :raises KeyError: 节点未注册
        """
        info = self.node_info(node_name)
        if info is None:
            raise KeyError(f"Node '{node_name}' not registered in hub '{self._name}'")

        return ZMQChannelProxy(
            name=proxy_name or node_name,
            description=description or f"ZMQ Hub proxy for '{node_name}'",
            address=info.channel_address,
        )

    # ------------------------------------------------------------------
    # as_channel — 集成到 Shell
    # ------------------------------------------------------------------

    def as_channel(self) -> Channel:
        """
        将 hub 导出为 PyChannel，可集成到 Shell 中。

        生成的 Channel 提供:
        - list_nodes: 查看已注册节点
        - open_node: 打开节点（创建 proxy 作为 virtual child）
        - close_node: 关闭已打开节点
        - context_messages: 展示节点状态
        """
        state = ZMQHubChannelState(hub=self)
        return new_channel_from_state(state)


# ------------------------------------------------------------------
# HubChannelState — as_channel() 的状态驱动实现
# ------------------------------------------------------------------

class ZMQHubChannelState(ChannelState):
    """ZMQHub 的 ChannelState 实现。"""

    def __init__(self, *, hub: ZMQHub):
        self._hub = hub
        self._proxy_channels: dict[str, ZMQChannelProxy] = {}
        self._proxy_channels_lock = threading.Lock()
        self._opened_nodes: set[str] = set()
        self._opened_lock = threading.Lock()
        self._own_commands = self._build_commands()

    def _build_commands(self) -> dict[str, Command]:
        from ghoshell_moss.core.concepts.command import PyCommand

        async def list_nodes() -> str:
            """
            列出当前 hub 上所有已注册的节点及其状态。
            """
            nodes = self._hub.registered_nodes()
            if not nodes:
                return "No registered nodes."

            lines = [f"Registered nodes ({len(nodes)}):"]
            for name, info in nodes.items():
                marker = " [OPEN]" if name in self._opened_nodes else ""
                lines.append(f"  - {name}{marker}: {info.channel_address}")
            return "\n".join(lines)

        async def open_node(name: str) -> str:
            """
            打开指定的节点，使其 channel 可用。
            :param name: 节点名称
            """
            info = self._hub.node_info(name)
            if info is None:
                available = list(self._hub.registered_nodes().keys())
                return f"Node '{name}' not found. Available: {available}"

            with self._opened_lock:
                if name in self._opened_nodes:
                    return f"Node '{name}' is already open."
                self._opened_nodes.add(name)
            return f"Node '{name}' opened."

        async def close_node(name: str) -> str:
            """
            关闭已打开的节点连接。
            :param name: 节点名称
            """
            with self._opened_lock:
                if name not in self._opened_nodes:
                    return f"Node '{name}' is not open."
                self._opened_nodes.discard(name)
            return f"Node '{name}' closed."

        return {
            "list_nodes": PyCommand(list_nodes),
            "open_node": PyCommand(open_node),
            "close_node": PyCommand(close_node),
        }

    def name(self) -> str:
        return self._hub._name

    def description(self) -> str:
        return f"ZMQ Hub '{self._hub._name}' — 动态节点发现与管理"

    def is_available(self) -> bool:
        return not self._hub._closed.is_set()

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands.copy()

    def get_own_command(self, name: str) -> Command | None:
        return self._own_commands.get(name)

    async def get_context_messages(self) -> list[str]:
        nodes = self._hub.registered_nodes()
        if not nodes:
            return [f"### [ZMQ Hub: {self._hub._name}]\nNo registered nodes."]

        lines = [f"### [ZMQ Hub: {self._hub._name}]\n"]
        lines.append("**Registered nodes** (use `open_node <name>` to connect):\n")
        for name, info in nodes.items():
            marker = " [OPEN]" if name in self._opened_nodes else ""
            lines.append(f"- `{name}`{marker} ({info.channel_address})")

        if self._opened_nodes:
            lines.append("\n**Opened nodes**:")
            for name in self._opened_nodes:
                lines.append(f"- {name}")
        else:
            lines.append("\nNo nodes opened. Use `open_node <name>` to connect.")

        return ["\n".join(lines)]

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        with self._opened_lock:
            opened = self._opened_nodes.copy()

        channels: dict[ChannelName, Channel] = {}
        stale: list[str] = []

        for node_name in opened:
            info = self._hub.node_info(node_name)
            if info is None:
                stale.append(node_name)
                continue

            safe_name = node_name.replace("/", "_")
            with self._proxy_channels_lock:
                existing = self._proxy_channels.get(safe_name)
            if existing is not None:
                channels[safe_name] = existing
            else:
                proxy = ZMQChannelProxy(
                    name=safe_name,
                    description=f"ZMQ Hub node: {node_name}",
                    address=info.channel_address,
                )
                channels[proxy.name()] = proxy

        for name in stale:
            with self._opened_lock:
                self._opened_nodes.discard(name)
            safe_name = name.replace("/", "_")
            with self._proxy_channels_lock:
                self._proxy_channels.pop(safe_name, None)

        with self._proxy_channels_lock:
            for ch_name, ch in channels.items():
                self._proxy_channels[ch_name] = ch

        return channels.copy()


# ------------------------------------------------------------------
# 辅助工具 — 节点侧注册/注销/查询
# ------------------------------------------------------------------

async def zmq_register(
    hub_address: str,
    name: str,
    channel_address: str,
    description: str = "",
    timeout: float = 5.0,
) -> dict:
    """
    向 ZMQHub 注册一个节点。

    :param hub_address: Hub 的 registry 地址
    :param name: 节点名称
    :param channel_address: 节点 channel 的 ZMQ 地址
    :param description: 节点描述
    :param timeout: 等待响应超时
    :return: hub 的响应
    """
    ctx = zmq.asyncio.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(hub_address)
        await dealer.send_multipart([
            b"",
            json.dumps({
                "action": "register",
                "name": name,
                "description": description,
                "channel_address": channel_address,
            }).encode(),
        ])
        frames = await asyncio.wait_for(dealer.recv_multipart(), timeout=timeout)
        return json.loads(frames[-1])
    finally:
        dealer.close(linger=0)


async def zmq_unregister(
    hub_address: str,
    name: str,
    timeout: float = 3.0,
) -> None:
    """
    从 ZMQHub 注销一个节点。

    :param hub_address: Hub 的 registry 地址
    :param name: 节点名称
    :param timeout: 等待超时
    """
    ctx = zmq.asyncio.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(hub_address)
        await dealer.send_multipart([
            b"",
            json.dumps({"action": "unregister", "name": name}).encode(),
        ])
        # 等待 hub 确认收到
        await asyncio.wait_for(dealer.recv_multipart(), timeout=timeout)
    finally:
        dealer.close(linger=0)


async def zmq_query(
    hub_address: str,
    timeout: float = 5.0,
) -> list[dict]:
    """
    查询 ZMQHub 上所有已注册节点。

    :param hub_address: Hub 的 registry 地址
    :param timeout: 等待响应超时
    :return: 节点列表 [{"name": ..., "channel_address": ..., "description": ...}, ...]
    """
    ctx = zmq.asyncio.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.LINGER, 0)
    try:
        dealer.connect(hub_address)
        await dealer.send_multipart([
            b"",
            json.dumps({"action": "query"}).encode(),
        ])
        frames = await asyncio.wait_for(dealer.recv_multipart(), timeout=timeout)
        response = json.loads(frames[-1])
        return response.get("nodes", [])
    finally:
        dealer.close(linger=0)
