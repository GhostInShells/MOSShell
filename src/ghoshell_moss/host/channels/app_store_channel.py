from typing import Optional
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_channel_from_state, ChannelState
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.abcd.app import AppStore
from ghoshell_container import IoCContainer
from threading import Lock

__all__ = ['AppStoreChannel', 'build_apps_channel', 'AppStoreChannelState']


class AppStoreChannel(Channel):
    """
    the App Store Channel.
    """

    def __init__(self, name: str, description: str = ""):
        from ghoshell_common.helpers import uuid
        self._name = name
        self._description = description
        self._id = uuid()

    def name(self) -> ChannelName:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> ChannelRuntime:
        app_store = container.force_fetch(AppStore)
        matrix = container.force_fetch(Matrix)
        real_channel = build_apps_channel(
            store=app_store,
            matrix=matrix,
            name=self._name,
            description=self._description,
            id=self._id
        )
        return real_channel.bootstrap(container)


class AppStoreChannelState(ChannelState):

    def __init__(
            self,
            *,
            app_store: AppStore,
            matrix: Matrix,
            name: str,
            description: str = "",
    ):
        self._app_store = app_store
        self._matrix = matrix
        self._name = name
        self._description = description
        self._own_commands: dict[str, Command] = {}
        self._app_channels: dict[str, Channel] = {}
        self._app_channels_lock = Lock()
        self._bootstrap()

    def _bootstrap(self) -> None:
        from ghoshell_moss.core.concepts.command import PyCommand

        async def list_apps() -> str:
            """
            获取当前环境所有可发现 App 的详细清单及运行状态。
            AI 在尝试启动任何 App 前，应先通过此命令确认其 address 和当前状态。
            """
            return await self._app_store.get_apps_context()

        async def start(fullname: str, argument: str = "") -> str:
            """
            启动指定的 App。
            :param fullname: App 的完整名称，如 'group/name'。
            :param argument: 启动参数，将作为命令行参数传递给 App。
            注意：启动是异步的，可以通过 list 确认是否成功进入 running 状态。
            """
            return await self._app_store.start_app(fullname, argument)

        async def stop(fullname: str) -> str:
            """
            强制停止并卸载一个运行中的 App。
            :param fullname: 目标 App 全名。
            """
            return await self._app_store.stop_app(fullname)

        self._own_commands = {
            'start': PyCommand(start),
            'list_apps': PyCommand(list_apps),
            'stop': PyCommand(stop),
        }

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def is_available(self) -> bool:
        return self._matrix.is_running()

    def is_dynamic(self) -> bool:
        return True

    async def get_context_messages(self) -> list[str]:
        context_str = await self._app_store.get_apps_context()
        header = "### [App Runtime Status]\n"
        footer = "\n---\n注：若 App 处于 ERROR 状态，请检查日志或尝试重启。"
        return [header + context_str + footer]

    def get_virtual_children(self) -> dict[ChannelName, Channel]:
        channels = {}
        for app in self._app_store.list_apps():
            address = app.address
            if address in self._app_channels:
                channels[address] = self._app_channels[address]
                continue
            name = app.fullname.replace('/', '_')
            channel_proxy = self._matrix.channel_proxy(
                address=address,
                name=name,
                description=app.description,
            )
            channels[address] = channel_proxy
        with self._app_channels_lock:
            self._app_channels = channels
        return {chan.name(): chan for chan in self._app_channels.copy().values()}

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands.copy()

    def get_own_command(self, name: str) -> Command | None:
        return self._own_commands.get(name)


def build_apps_channel(
        store: AppStore,
        matrix: Matrix,
        name: str,
        description: str = '',
        id: str | None = None,
) -> Channel:
    """
    构建 App 管理中心通道。
    该通道允许 AI 发现、启动、停止和初始化物理/逻辑应用 (Apps)。
    """
    # 默认描述强调“中心化管理”
    default_description = (
        "App Store 核心通道，用于管理当前环境下的所有可用应用。"
        "你可以通过此通道拉起具有特定功能的子进程"
    )

    state = AppStoreChannelState(
        app_store=store,
        matrix=matrix,
        name=name,
        description=description or default_description,
    )
    return new_channel_from_state(state, id=id)
