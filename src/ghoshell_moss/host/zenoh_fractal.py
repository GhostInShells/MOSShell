import asyncio
import threading
from pathlib import Path

from ghoshell_moss.core.blueprint.matrix import Fractal, Cell
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state, ChannelState
from ghoshell_moss.bridges.zenoh_bridge import ZenohChannelProvider, ZenohProxyChannel
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

__all__ = ['FractalCell', 'ZenohSessionFractal', 'FractalHubChannelState']


class FractalCell(Cell):
    """
    type='fractal' 的 Cell。表示通过 fractal 协议连接到当前 Matrix 的外部节点。
    """

    def __init__(self, name: str, event: threading.Event):
        # todo: 告知 ai 如何获得 name / description
        self.name = name
        self.type = 'fractal'
        self.description = ''
        self.where = ''
        self._alive_event = event

    @property
    def address(self) -> str:
        return Cell.make_address('fractal', self.name)

    def is_alive(self) -> bool:
        # todo: event 机制是为了懒刷新准备的.
        return self._alive_event.is_set()


class ZenohSessionFractal(Fractal):
    """
    基于 zenoh 实现 Fractal 分形通讯协议。

    1. 创建独立的 zenoh session，读取 workspace 中的配置文件。
    2. 通过 zenoh liveness 机制发现子节点。
    3. 通过 ZenohChannelProvider 将本地 channel 暴露给父节点。
    4. 通过 channel_hub 提供一个被动 Channel 来展示已连接的子节点。

    Key space:
      - Liveness discovery: MOSS/fractal/{name}
      - Channel bridge:     MOSS/fractal/node/{name}/channel_bridge/{role}
    """

    FRACTAL_SESSION_SCOPE = "fractal"
    FRACTAL_LIVENESS_PREFIX = "MOSS/fractal"

    def __init__(self, zenoh_conf_file: Path, name: str):
        self._conf_file = zenoh_conf_file
        self._name = name
        self._session: zenoh.Session | None = None
        self._session_lock = threading.Lock()
        self._provided_channels: dict[str, Channel] = {}
        self._provided_channels_lock = threading.Lock()
        self._liveness_token: zenoh.LivelinessToken | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def session(self) -> zenoh.Session:
        """懒加载打开独立的 zenoh session。"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    conf = zenoh.Config.from_file(str(self._conf_file))
                    self._session = zenoh.open(conf)
        return self._session

    def connected(self) -> list[Cell]:
        """
        通过 zenoh liveness wildcard query 发现外部子节点。
        参考 MatrixImpl._check_initial_liveness 模式。
        """
        # todo: 疑问, 这个同步逻辑是否有阻塞? 如果有阻塞, 它应该搬迁到一个异步任务里.
        #   我们可以加生命周期治理, 我在集成时去添加治理链路.
        s = self.session
        if s.is_closed():
            return []

        cells: list[Cell] = []
        prefix = self.FRACTAL_LIVENESS_PREFIX
        key_expr = f"{prefix}/**"

        for sample in s.liveliness().get(key_expr):
            key = str(sample.result.key_expr)
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) + 1:]  # "MOSS/fractal/" 之后的部分
            if not name or name == self._name:
                continue  # 跳过自身
            event = threading.Event()
            event.set()  # 通过 liveness 发现即视为存活
            cells.append(FractalCell(name, event))

        return cells

    def explain(self) -> str:
        nodes = self.connected()
        lines = [
            f"Zenoh Fractal Protocol",
            f"Node: {self._name}",
            f"Connected nodes ({len(nodes)}):",
        ]
        for c in nodes:
            lines.append(f"  - {c.name} (alive={c.is_alive()})")
        return "\n".join(lines)

    def provide_channel(
            self,
            channel: Channel | ChannelRuntime,
            transport: str | None = None,
    ) -> asyncio.Future[None]:
        """
        通过 fractal 自己的 zenoh session 用 ZenohChannelProvider 暴露 channel。
        声明 liveness token 使父节点可发现。
        唯一性检查：同名或同 ID 的 channel 不能重复提供。
        """
        # todo: 考虑优化.
        # 解析 channel 的 name 和 id
        if isinstance(channel, ChannelRuntime):
            channel_name = channel.name
            channel_id = channel.id
            channel_obj = channel.channel
        else:
            channel_name = channel.name()
            channel_id = channel.id()
            channel_obj = channel

        # 唯一性检查
        with self._provided_channels_lock:
            if channel_name in self._provided_channels:
                raise ValueError(
                    f"Channel with name '{channel_name}' already provided"
                )
            for existing_name, existing_channel in self._provided_channels.items():
                if existing_channel.id() == channel_id:
                    raise ValueError(
                        f"Channel with id '{channel_id}' already provided "
                        f"as '{existing_name}'"
                    )
            self._provided_channels[channel_name] = channel_obj

        # 创建 provider
        # todo: 太复杂了, 没有必要. 上层传入 matrix 即可.
        provider = ZenohChannelProvider(
            address=self._name,
            session_scope=self.FRACTAL_SESSION_SCOPE,
            zenoh_session=self.session,
        )

        # 声明 liveness token 使父节点可发现
        liveness_key = f"{self.FRACTAL_LIVENESS_PREFIX}/{self._name}"
        self._liveness_token = self.session.liveliness().declare_token(liveness_key)

        loop = asyncio.get_running_loop()

        if isinstance(channel, ChannelRuntime):
            task = loop.create_task(provider.arun_channel_runtime(channel))
        else:
            task = loop.create_task(provider.arun_until_closed(channel))

        return task

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

    def close(self) -> None:
        """关闭 fractal session 和 liveness token。"""
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


class FractalHubChannelState(ChannelState):
    """
    Fractal Hub 的 ChannelState 实现。

    类似 AppStoreChannelState 模式：
    - get_virtual_children() 调用 connected() 发现子节点
    - 对每个子节点创建 ZenohProxyChannel
    - 平铺返回，不嵌套
    - is_dynamic() -> True
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
        # todo: 考虑没有子节点时, 返回 false 隐藏.
        return True

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
