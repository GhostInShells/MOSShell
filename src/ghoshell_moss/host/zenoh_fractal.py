import asyncio
import threading
from pathlib import Path

from ghoshell_moss.core.blueprint.matrix import Fractal, Cell
from ghoshell_moss.core.blueprint.environment import MossMeta
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state, ChannelState
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.contracts import LoggerItf
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

__all__ = ['FractalCell', 'ZenohSessionFractal', 'FractalHubChannelState']


class FractalCell(Cell):
    """
    type='fractal' 的 Cell。表示通过 fractal 协议连接到当前 Matrix 的外部节点。
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.type = 'fractal'
        self.description = description or f"Fractal node: {name}"
        self.where = ''
        self._alive = True

    @property
    def address(self) -> str:
        return Cell.make_address('fractal', self.name)

    def is_alive(self) -> bool:
        return self._alive

    def mark_dead(self) -> None:
        self._alive = False


class ZenohSessionFractal(Fractal):
    """
    基于 zenoh 实现 Fractal 分形通讯协议。

    1. 创建独立的 zenoh session，读取 workspace 中的配置文件。
    2. 通过后台异步任务定期发现子节点 (zenoh liveness)，connected() 即时返回缓存。
    3. 通过 ZenohChannelProvider 将本地 channel 暴露给父节点。
    4. 通过 channel_hub 提供一个被动 Channel 来展示已连接的子节点。

    Key space:
      - Liveness discovery: MOSS/fractal/{name}
      - Channel bridge:     MOSS/fractal/node/{name}/channel_bridge/{role}
    """

    FRACTAL_SESSION_SCOPE = "fractal"
    FRACTAL_LIVENESS_PREFIX = "MOSS/fractal"
    DEFAULT_REFRESH_INTERVAL = 5.0

    def __init__(
            self,
            meta: MossMeta,
            zenoh_conf_file: Path,
            *,
            logger: LoggerItf,
            refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    ):
        self._conf_file = zenoh_conf_file
        self._name = meta.name
        self._description = meta.description
        self._logger = logger
        self._session: zenoh.Session | None = None
        self._session_lock = threading.Lock()
        self._provided_future: asyncio.Task | None = None
        self._transport_endpoint: str | None = None
        self._liveness_token: zenoh.LivelinessToken | None = None
        # 子节点缓存与异步刷新
        self._cells: dict[str, Cell] = {}
        self._cells_lock = threading.Lock()
        self._refresh_interval = refresh_interval
        self._refresh_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def session(self) -> zenoh.Session:
        """懒加载打开独立的 zenoh session。有 transport 时注入 connect/endpoints 实现反向注册。"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    conf = zenoh.Config.from_file(str(self._conf_file))
                    if self._transport_endpoint:
                        conf.insert_json5(
                            "connect/endpoints",
                            f'["{self._transport_endpoint}"]',
                        )
                    self._session = zenoh.open(conf)
        return self._session

    # ------------------------------------------------------------------
    # 生命周期 (async context manager)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ZenohSessionFractal":
        loop = asyncio.get_running_loop()
        self._refresh_task = loop.create_task(self._refresh_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._liveness_token is not None:
            try:
                self._liveness_token.undeclare()
            except RuntimeError:
                pass
            self._liveness_token = None
        if self._session is not None:
            if not self._session.is_closed():
                try:
                    self._session.close()
                except RuntimeError:
                    pass
            self._session = None

    # ------------------------------------------------------------------
    # 异步子节点发现
    # ------------------------------------------------------------------

    async def _refresh_loop(self) -> None:
        """后台循环：定期通过 zenoh liveness 查询子节点并更新缓存。"""
        while True:
            try:
                await self._refresh_connected_cells()
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.warning(
                    "Fractal node '%s' liveness refresh failed", self._name, exc_info=True
                )
            await asyncio.sleep(self._refresh_interval)

    async def _refresh_connected_cells(self) -> None:
        """在 executor 中执行同步 zenoh liveness 查询，无阻塞更新 cells 缓存。"""
        loop = asyncio.get_running_loop()
        new_cells = await loop.run_in_executor(None, self._query_liveness)
        with self._cells_lock:
            self._cells = {c.name: c for c in new_cells}

    def _query_liveness(self) -> list[Cell]:
        """同步 zenoh liveness wildcard 查询，在 executor 线程中调用。"""
        s = self.session
        if s.is_closed():
            return []

        prefix = self.FRACTAL_LIVENESS_PREFIX
        key_expr = f"{prefix}/**"

        cells: list[Cell] = []
        for sample in s.liveliness().get(key_expr):
            key = str(sample.result.key_expr)
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) + 1:]  # "MOSS/fractal/" 之后的部分
            if not name or name == self._name:
                continue
            cells.append(FractalCell(name))

        return cells

    def connected(self) -> list[Cell]:
        """即时返回缓存的子节点列表（非阻塞）。"""
        with self._cells_lock:
            return list(self._cells.values())

    # ------------------------------------------------------------------
    # Channel Provider
    # ------------------------------------------------------------------

    def create_channel_provider(self) -> ZenohChannelProvider:
        """
        创建绑定到此 fractal zenoh session 的 ChannelProvider。
        供上层 Matrix 集成时直接使用。
        """
        return ZenohChannelProvider(
            address=self._name,
            session_scope=self.FRACTAL_SESSION_SCOPE,
            zenoh_session=self.session,
        )

    def provide_channel(
            self,
            channel: Channel | ChannelRuntime,
            transport: str | None = None,
    ) -> asyncio.Future[None]:
        """
        通过 fractal 自己的 zenoh session 用 ZenohChannelProvider 暴露 channel。
        声明 liveness token 使父节点可发现。

        :param transport: 父节点的 zenoh 端点地址，如 "tcp/192.168.1.100:20770"。
                         为 None 时使用 peer 多播自发现（默认）。
                         设置后 zenoh session 会主动 connect 到该端点，实现反向注册。
        """
        if self._provided_future is not None:
            raise RuntimeError(
                f"Channel already provided for fractal node '{self._name}'"
            )

        if transport is not None:
            self._transport_endpoint = transport
            self._logger.info(
                "Fractal node '%s' reverse-registering to parent: %s",
                self._name, transport,
            )

        provider = self.create_channel_provider()

        # 声明 liveness token 使父节点可发现
        liveness_key = f"{self.FRACTAL_LIVENESS_PREFIX}/{self._name}"
        self._liveness_token = self.session.liveliness().declare_token(liveness_key)

        loop = asyncio.get_running_loop()

        if isinstance(channel, ChannelRuntime):
            task = loop.create_task(provider.arun_channel_runtime(channel))
        else:
            task = loop.create_task(provider.arun_until_closed(channel))

        self._provided_future = task
        self._logger.debug(
            "Fractal node '%s' provided channel, liveness_key=%s",
            self._name, liveness_key,
        )

        return task

    # ------------------------------------------------------------------
    # Hub
    # ------------------------------------------------------------------

    def explain(self) -> str:
        nodes = self.connected()
        lines = [
            f"Zenoh Fractal Protocol",
            f"Node: {self._name}",
        ]
        if self._transport_endpoint:
            lines.append(f"Parent: {self._transport_endpoint}")
        else:
            lines.append("Mode: peer multicast (no parent)")
        lines.append(f"Connected nodes ({len(nodes)}):")
        for c in nodes:
            lines.append(f"  - {c.name} (alive={c.is_alive()})")
        return "\n".join(lines)

    def channel_hub(self, name: str, description: str = '') -> Channel:
        """
        返回一个被动 Channel（无 start/stop 命令），用于展示已连接的 fractal 子节点。

        子节点通过 ZenohProxyChannel 代理，使用 address=cell.name,
        session_scope="fractal" 与 provider 对齐。
        """
        state = FractalHubChannelState(
            fractal=self,
            name=name,
            description=description or (
                "Fractal Hub 通道，用于发现和管理通过分形协议连接的远程 Matrix 节点。"
                "你可以通过此通道查看已连接的节点及其提供的子通道。"
            ),
        )
        return new_channel_from_state(state)


class FractalHubChannelState(ChannelState):
    """
    Fractal Hub 的 ChannelState 实现。

    类似 AppStoreChannelState 模式：
    - get_virtual_children() 调用 connected() 发现子节点
    - 对每个子节点创建 ZenohProxyChannel
    - 平铺返回，不嵌套
    - is_dynamic() -> True
    - 没有子节点时 is_available() -> False
    """

    def __init__(
            self,
            *,
            fractal: ZenohSessionFractal,
            name: str,
            description: str = "",
    ):
        self._fractal = fractal
        self._name = name
        self._description = description
        self._proxy_channels: dict[str, Channel] = {}
        self._proxy_channels_lock = threading.Lock()

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        return bool(self._fractal.connected())

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return {}

    def get_own_command(self, name: str) -> Command | None:
        return None

    async def get_context_messages(self) -> list[str]:
        cells = self._fractal.connected()
        if not cells:
            return ["### [Fractal Hub]\nNo connected fractal nodes."]
        lines = ["### [Fractal Hub - Connected Nodes]\n"]
        for cell in cells:
            lines.append(f"- **{cell.name}** (alive={cell.is_alive()})")
        return ["\n".join(lines)]

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        cells = self._fractal.connected()
        channels: dict[ChannelName, Channel] = {}

        for cell in cells:
            safe_name = cell.name.replace('/', '_')
            with self._proxy_channels_lock:
                existing = self._proxy_channels.get(safe_name)
            if existing is not None:
                channels[safe_name] = existing
            else:
                proxy = ZenohProxyChannel(
                    address=cell.name,
                    session_scope=ZenohSessionFractal.FRACTAL_SESSION_SCOPE,
                    name=safe_name,
                    description=f"Fractal child node: {cell.name}",
                    zenoh_session=self._fractal.session,
                )
                channels[proxy.name()] = proxy

        with self._proxy_channels_lock:
            self._proxy_channels = channels
        return channels.copy()
