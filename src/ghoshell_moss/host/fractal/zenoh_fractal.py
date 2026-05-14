import asyncio
import contextlib
import threading
from pathlib import Path

from typing import Iterable, Type
from typing_extensions import Self
from ghoshell_moss.core.blueprint.matrix import Cell
from ghoshell_moss.core.blueprint.host import FractalHub, FractalNodeProvider
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.states_channel import StatefulChannel
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime, ChannelNamePattern, ChannelProvider
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state, ChannelState
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.contracts import LoggerItf
from ghoshell_container import IoCContainer, Provider, INSTANCE
from ghoshell_moss.depends import depend_zenoh
from ._base import FractalCell, FractalKeyExpressions, FRACTAL_SESSION_SCOPE
import orjson
import regex as re
import time
depend_zenoh()
import zenoh

__all__ = [
    'ZenohSessionFractalHub', 'FractalHubChannelState', 'ZenohFractalHubProvider',
    'ZenohSessionFractalNodeProvider', 'ZenohFractalNodeProviderProvider',
]


class ZenohSessionFractalHub(FractalHub):
    """
    基于 zenoh 实现 Fractal 分形通讯协议。

    1. 创建独立的 zenoh session，读取 workspace 中的配置文件。
    2. 通过 subscriber 监听 manifest/** put 事件实时发现子节点。
    3. 后台定时清理超过 stale_timeout 未心跳的节点。
    4. 通过 channel_hub 提供一个被动 Channel 来展示已连接的子节点。

    Key space:
      - Manifest discovery: {prefix}/{hub}/manifests/{node_name}
      - Channel bridge:     {prefix}/{hub}/providers/{node_name}
    """

    def __init__(
            self,
            hub_name: str,
            zenoh_conf_file: Path,
            *,
            logger: LoggerItf,
            session_scope: str | None = None,
            address_prefix: str | None = None,
            transport_endpoint: str | None = None,
            refresh_interval: float = 2.0,
            auto_approve_connecting: bool = False,
    ):
        self._conf_file = zenoh_conf_file
        self._hub_name = hub_name
        self._logger = logger
        self._session: zenoh.Session | None = None
        self._session_lock = threading.Lock()
        self._transport_endpoint: str | None = transport_endpoint
        self._key_expr = FractalKeyExpressions(
            hub_name=self._hub_name,
            address_prefix=address_prefix,
        )
        self._auto_approve_connecting = auto_approve_connecting

        # 子节点缓存 — subscriber 回调写入，refresh loop 做 stale 清理
        self._connected_cells: dict[str, FractalCell] = {}
        self._cell_last_seen: dict[str, float] = {}
        self._connected_cells_lock = threading.Lock()
        self._approved_cell_names: set[str] = set()

        self._refresh_interval = refresh_interval
        self._stale_timeout = refresh_interval * 3
        self._refresh_task: asyncio.Task | None = None
        self._subscriber: zenoh.Subscriber | None = None
        self.session_scope = session_scope or FRACTAL_SESSION_SCOPE
        self._started = False
        self._closed = False
        self._log_prefix = f"<ZenohSessionFractalHub name={self._hub_name} session_scope={self.session_scope}>"

    @property
    def name(self) -> str:
        return self._hub_name

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

    def get_connected(self) -> list[Cell]:
        """即时返回缓存的子节点列表（非阻塞）。"""
        with self._connected_cells_lock:
            return list(self._connected_cells.values())

    def has_connected(self) -> bool:
        return len(self._connected_cells) > 0

    def is_running(self) -> bool:
        return self._started and not self._closed

    def as_channel(self, name: str = '', description: str = '') -> Channel:
        return self.as_channel_hub(name, description)

    # ------------------------------------------------------------------
    # Cell 访问控制
    # ------------------------------------------------------------------

    def is_cell_connected(self, name: str) -> bool:
        return name in self._connected_cells

    def is_cell_approved(self, name: str) -> bool:
        if self._auto_approve_connecting:
            return name in self._connected_cells
        return name in self._approved_cell_names

    def approve_cell(self, name: str):
        if name in self._connected_cells:
            self._approved_cell_names.add(name)
        else:
            raise KeyError(f"cell name '{name}' not found")

    def disapprove_cell(self, name: str):
        if name in self._connected_cells:
            self._approved_cell_names.discard(name)
        else:
            raise KeyError(f"cell name '{name}' not found")

    def make_proxy_address(self, cell_name: str) -> str:
        return self._key_expr.provider_node_address(cell_name)

    # ------------------------------------------------------------------
    # 生命周期 (async context manager)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ZenohSessionFractalHub":
        if self._started:
            raise RuntimeError("Fractal hub already started")
        self._started = True
        loop = asyncio.get_running_loop()
        _ = self.session
        # 声明 subscriber 监听 manifest put 事件（zenoh 后台线程回调）
        self._subscriber = self.session.declare_subscriber(
            self._key_expr.manifests_wildcard(),
            self._on_manifest_sample,
        )
        self._refresh_task = loop.create_task(self._refresh_loop())
        self._logger.debug(
            "%s subscriber declared on %s",
            self._log_prefix, self._key_expr.manifests_wildcard(),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._subscriber is not None:
            try:
                self._subscriber.undeclare()
            except RuntimeError:
                pass
            self._subscriber = None
        if self._session is not None:
            if not self._session.is_closed():
                try:
                    self._session.close()
                except RuntimeError:
                    pass
            self._session = None

    # ------------------------------------------------------------------
    # Subscriber 回调 (zenoh 后台线程中执行)
    # ------------------------------------------------------------------

    def _on_manifest_sample(self, sample: zenoh.Sample) -> None:
        """subscriber 回调：收到 manifest put 时更新 _connected_cells。"""
        key = str(sample.key_expr)
        try:
            data = orjson.loads(sample.payload.to_bytes())
            if not isinstance(data, dict):
                return
            cell = FractalCell.from_dict(data)
            if cell is None:
                return
            name = cell.name
            if re.fullmatch(ChannelNamePattern, name) is None:
                self._logger.warning(
                    "%s subscriber received invalid cell name: %s",
                    self._log_prefix, name,
                )
                return
            now = time.monotonic()
            with self._connected_cells_lock:
                existing = self._connected_cells.get(name)
                if existing is not None:
                    existing_key = self._key_expr.manifest_key(name)
                    if key != existing_key:
                        self._logger.warning(
                            "%s duplicate cell name %s from keys %s and %s",
                            self._log_prefix, name, existing_key, key,
                        )
                        del self._connected_cells[name]
                        self._cell_last_seen.pop(name, None)
                        return
                self._connected_cells[name] = cell
                self._cell_last_seen[name] = now
        except Exception:
            pass

    async def _refresh_loop(self) -> None:
        """后台循环：定期清理超过 stale_timeout 未心跳的子节点。"""
        while not self._closed:
            try:
                await asyncio.sleep(self._refresh_interval)
                self._prune_stale_cells()
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.warning(
                    "%s stale check failed", self._log_prefix, exc_info=True,
                )

    def _prune_stale_cells(self) -> None:
        """移除超过 stale_timeout 未心跳的子节点。"""
        now = time.monotonic()
        with self._connected_cells_lock:
            stale = [
                name for name, ts in self._cell_last_seen.items()
                if now - ts > self._stale_timeout
            ]
            for name in stale:
                del self._connected_cells[name]
                del self._cell_last_seen[name]
                self._approved_cell_names.discard(name)
                self._logger.debug(
                    "%s pruned stale node: %s", self._log_prefix, name,
                )

    def usage(self) -> str:
        lines = [
            f"Moss Zenoh Fractal Protocol",
            f"Hub Name: {self._hub_name}",
            f"Cell Manifest Prefix: {self._key_expr.manifests_prefix()}",
        ]
        if self._transport_endpoint:
            lines.append(f"Parent: {self._transport_endpoint}")
        else:
            lines.append("Mode: peer multicast (no parent)")

        lines.append(self._conf_file.read_text())
        return "\n".join(lines)

    def status(self) -> str:
        lines = [
            "Zenoh Fractal Hub",
            f"Running: {self.is_running()}",
        ]
        nodes = self.get_connected()
        if len(nodes) > 0:
            lines.append(f"Connected nodes ({len(nodes)}):")
            for c in nodes:
                lines.append(f"  - {c.name} (alive={c.is_alive()})")
        else:
            lines.append("No connected nodes")
        return "\n".join(lines)

    def as_channel_hub(self, name: str, description: str = '') -> StatefulChannel:
        """
        返回一个被动 Channel（无 start/stop 命令），用于展示已连接的 fractal 子节点。

        子节点通过 ZenohProxyChannel 代理，使用 address=cell.name,
        session_scope="fractal" 与 provider 对齐。
        """
        state = FractalHubChannelState(
            hub=self,
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

    - 通过 fractal.connected() 发现子节点，展示可用列表
    - AI 使用 open_node/close_node 命令手动管理需要连接哪些节点
    - get_virtual_children() 只返回已打开的节点
    - 已打开的节点 proxy 会被复用
    """

    def __init__(
            self,
            *,
            hub: ZenohSessionFractalHub,
            name: str | None = None,
            description: str = "",
    ):
        self._hub = hub
        self._name = name or hub.name
        self._description = description or "Moss Fractal "
        self._proxy_channels: dict[str, Channel] = {}
        self._proxy_channels_lock = threading.Lock()
        self._own_commands = self._build_commands()

    def _build_commands(self) -> dict[str, Command]:
        from ghoshell_moss.core.blueprint.channel_builder import new_command

        async def open_node(name: str) -> str:
            """
            打开指定的分形子节点，使其 channel 可用。
            应先通过 context 查看可用节点列表，再决定打开哪些。
            :param name: 节点名称，来自 context 列表中的 node name。
            """
            if not self._hub.is_cell_connected(name):
                return f"Node '{name}' not found."
            elif self._hub.is_cell_approved(name):
                return f"Node '{name}' already opened."
            try:
                self._hub.approve_cell(name)
            except KeyError:
                return f"Node '{name}' not found."
            return f"Fractal node '{name}' opened."

        async def close_node(name: str) -> str:
            """
            关闭指定的分形子节点，从 virtual children 中移除。
            :param name: 节点名称。
            """
            if not self._hub.is_cell_connected(name):
                return f"Node '{name}' not found."
            elif not self._hub.is_cell_approved(name):
                return f"Node '{name}' is not opened"
            try:
                self._hub.disapprove_cell(name)
            except KeyError:
                return f"Node '{name}' not found."
            return f"Fractal node '{name}' closed."

        return {
            'open_node': new_command(open_node),
            'close_node': new_command(close_node),
        }

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        # 如果没有任何连接, 直接隐藏.
        return self._hub.is_running() and self._hub.has_connected()

    def is_dynamic(self) -> bool:
        return True

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands.copy()

    def get_own_command(self, name: str) -> Command | None:
        return self._own_commands.get(name)

    async def get_context_messages(self) -> list[str]:
        return [self._hub.status()]

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        """只返回已打开（approved）的节点的 proxy。"""
        cells = self._hub.get_connected()

        channels: dict[ChannelName, Channel] = {}
        with self._proxy_channels_lock:
            exists = self._proxy_channels.copy()

        for cell in cells:
            cell_name = cell.name
            if not self._hub.is_cell_approved(cell_name):
                continue
            existing = exists.get(cell_name)
            if existing is not None:
                channels[cell_name] = existing
            else:
                address = self._hub.make_proxy_address(cell_name)
                proxy = ZenohProxyChannel(
                    address=address,
                    session_scope=self._hub.session_scope,
                    name=cell_name,
                    description=f"Fractal child node: {cell.name}",
                    zenoh_session=self._hub.session,
                )
                channels[cell_name] = proxy
        # 更新 proxy 缓存，清理已断开或未批准的节点
        with self._proxy_channels_lock:
            self._proxy_channels = channels
        return channels.copy()


class ZenohFractalHubProvider(Provider[FractalHub]):
    """
    默认的 Fractal 实现 Provider。
    读取 workspace 中的 zenoh_config_fractal.json5 创建 ZenohSessionFractal。
    可被 manifest providers 覆盖（更高优先级）。
    """

    def __init__(self, conf_file: str = "zenoh_config_fractal_hub.json5"):
        self._conf_file = conf_file

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return FractalHub

    def aliases(self) -> Iterable[Type]:
        yield ZenohSessionFractalHub

    def factory(self, con: IoCContainer) -> ZenohSessionFractalHub:
        workspace = con.force_fetch(Workspace)
        logger = con.get(LoggerItf)

        config_path = workspace.configs().abspath() / self._conf_file
        if not config_path.exists():
            raise FileNotFoundError(
                f"Fractal zenoh config not found: {config_path}"
            )
        env = con.force_fetch(Environment)
        meta = env.meta_config
        return ZenohSessionFractalHub(
            hub_name=meta.name,
            zenoh_conf_file=config_path,
            logger=logger,
        )


class ZenohSessionFractalNodeProvider(FractalNodeProvider):
    """
    将本地 channel 通过 zenoh 暴露到远程 FractalHub。

    独立管理 zenoh session，通过 FractalKeyExpressions 与 Hub 对齐 key space。
    __aenter__ 时 put manifest + declare liveness token + 启动后台 re-put loop，
    使 Hub 的 subscriber 可持续收到心跳。
    """

    def __init__(
        self,
        hub_name: str,
        zenoh_conf_file: Path,
        *,
        logger: LoggerItf,
        node_name: str | None = None,
        transport_endpoint: str | None = None,
        address_prefix: str | None = None,
        reput_interval: float = 2.0,
    ):
        self._hub_name = hub_name
        self._conf_file = zenoh_conf_file
        self._logger = logger
        self._node_name = node_name or f"{hub_name}_provider"
        self._transport_endpoint = transport_endpoint
        self._key_expr = FractalKeyExpressions(
            hub_name=hub_name,
            address_prefix=address_prefix,
        )
        self._session: zenoh.Session | None = None
        self._session_lock = threading.Lock()
        self._liveness_token: zenoh.LivelinessToken | None = None
        self._reput_interval = reput_interval
        self._reput_task: asyncio.Task | None = None
        self._manifest_key: str = ""
        self._cell_data: bytes = b""
        self._started = False
        self._closed = False

    @property
    def session(self) -> zenoh.Session:
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

    def channel_provider(self, name: str) -> ChannelProvider | None:
        address = self._key_expr.provider_node_address(node_name=self._node_name)
        return ZenohChannelProvider(
            address=address,
            session_scope=FRACTAL_SESSION_SCOPE,
            zenoh_session=self.session,
        )

    def is_running(self) -> bool:
        return self._started and not self._closed

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError(f"FractalNodeProvider '{self._node_name}' already started")
        self._started = True
        _ = self.session

        self._manifest_key = self._key_expr.manifest_key(self._node_name)
        cell = FractalCell(
            name=self._node_name,
            description=f"Fractal provider: {self._node_name}",
            where="",
        )
        self._cell_data = orjson.dumps(cell.to_dict())

        # 初始 put + liveness token
        self._do_put_manifest()
        self._liveness_token = self.session.liveliness().declare_token(self._manifest_key)

        # 启动后台 re-put loop
        loop = asyncio.get_running_loop()
        self._reput_task = loop.create_task(self._reput_loop())

        self._logger.debug(
            "FractalNodeProvider '%s' started, manifest_key=%s",
            self._node_name, self._manifest_key,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closed = True
        if self._reput_task is not None:
            self._reput_task.cancel()
            self._reput_task = None
        if self._liveness_token is not None:
            try:
                self._liveness_token.undeclare()
            except RuntimeError:
                pass
            self._liveness_token = None
        if self._session is not None and not self._session.is_closed():
            try:
                self._session.close()
            except RuntimeError:
                pass
            self._session = None

    def _do_put_manifest(self) -> None:
        self.session.put(self._manifest_key, self._cell_data)

    async def _reput_loop(self) -> None:
        """后台循环：周期性 re-put manifest 作为心跳，使 Hub subscriber 保持发现。"""
        while not self._closed:
            try:
                await asyncio.sleep(self._reput_interval)
                if not self._closed:
                    self._do_put_manifest()
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.warning(
                    "FractalNodeProvider '%s' reput failed",
                    self._node_name, exc_info=True,
                )

    async def provide(
        self,
        moss: "MossRuntime",
        name: str = '',
    ) -> None:
        """
        将当前 moss runtime 的 main_channel 提供给远程 Hub。

        与 ABC 默认实现不同：不检查 moss.shell.is_running()，
        因为 fractal 场景下 shell 已启动是正常情况。
        """
        stack = contextlib.AsyncExitStack()
        async with stack:
            if not self.is_running():
                await stack.enter_async_context(self)
            channel_provider = self.channel_provider(name or moss.name)
            if channel_provider is None:
                raise RuntimeError(f"channel_provider not available for '{name or moss.name}'")

            async def provide_moss_channels():
                await channel_provider.arun_until_closed(moss.shell.main_channel)

            await moss.matrix.create_task(provide_moss_channels())


class ZenohFractalNodeProviderProvider(Provider[FractalNodeProvider]):
    """
    IoC Provider for FractalNodeProvider。
    读取 workspace 中的 zenoh_config_fractal.json5 创建 ZenohSessionFractalNodeProvider。
    """

    def __init__(self, conf_file: str = "zenoh_config_fractal.json5"):
        self._conf_file = conf_file

    def singleton(self) -> bool:
        return True

    def contract(self) -> type:
        return FractalNodeProvider

    def aliases(self) -> Iterable[Type]:
        yield ZenohSessionFractalNodeProvider

    def factory(self, con: IoCContainer) -> ZenohSessionFractalNodeProvider:
        workspace = con.force_fetch(Workspace)
        logger = con.get(LoggerItf)

        config_path = workspace.configs().abspath() / self._conf_file
        if not config_path.exists():
            raise FileNotFoundError(
                f"Fractal node zenoh config not found: {config_path}"
            )
        env = con.force_fetch(Environment)
        meta = env.meta_config
        return ZenohSessionFractalNodeProvider(
            hub_name=meta.name,
            zenoh_conf_file=config_path,
            logger=logger,
        )
