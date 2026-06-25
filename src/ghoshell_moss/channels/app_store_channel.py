"""App 生命周期管理：发现、启动、停止独立进程应用 | 系统管理 | active

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.app_store_channel import AppStoreChannel
    main = new_shell_main_channel()
    main.import_channels(AppStoreChannel(name='apps'))
"""

import asyncio
from ghoshell_moss.core.concepts.channel import Channel, ChannelName, ChannelRuntime, ChannelCtx
from ghoshell_moss.core.concepts.command import Command
from ghoshell_moss.core.blueprint.states_channel import new_stateful_channel_from_main, ChannelState
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.app import AppStore
from ghoshell_container import IoCContainer
from ghoshell_moss.message import unique_id
from threading import Lock

__all__ = ['AppStoreChannel', 'build_apps_channel', 'AppStoreChannelState']


class AppStoreChannel(Channel):
    """
    the App Store Channel.
    """

    def __init__(self, name: str = '', description: str = ""):
        self._name = name or 'apps'
        self._description = description or (
            "App Store 核心通道，用于管理当前环境下的所有可用应用。"
            "你可以通过此通道拉起具有特定功能的子进程"
        )
        self._id = unique_id()

    def name(self) -> ChannelName:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def materialize(self, container: IoCContainer) -> ChannelRuntime:
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
        self._uid = unique_id()
        self._bootstrap()

    def id(self) -> str:
        return self._uid

    def _bootstrap(self) -> None:
        from ghoshell_moss.core.concepts.command import PyCommand

        async def list_apps() -> str:
            """
            获取当前环境所有可发现 App 的详细清单及运行状态。
            AI 在尝试启动任何 App 前，应先通过此命令确认其 address 和当前状态。
            """
            return await self._app_store.get_apps_context(refresh=True)

        async def start(fullname: str, argument: str = "", timeout: float = -1) -> str:
            """
            启动指定的 App。
            :param fullname: App 的完整名称，如 'group/name'。
            :param argument: 启动参数，将作为命令行参数传递给 App。
            :param timeout: 启动后等待 App Channel 就绪的超时秒数。
                -1 (默认): 不等待，立即返回。Channel 由 ChannelTree 后续 refresh 时自动连接。
                 0: 无限等待直到 App Channel 就绪。
                >0: 等待指定秒数，超时返回未连接状态。
            注意：启动是异步的，可以通过 list 确认是否成功进入 running 状态。
            """
            result = await self._app_store.start_app(fullname, argument)
            if timeout < 0:
                return result

            self_runtime = ChannelCtx.runtime()
            if self_runtime is None:
                return f"{result}\n[WARN] Not in channel runtime, cannot wait for connection"

            # 刷新 tree 结构，让 virtual children 被 tree 发现并 bootstrap
            await self_runtime.refresh_metas()

            # 从 tree 中查找已 bootstrap 的 child runtime
            proxy_name = self.make_app_proxy_name(fullname)
            child_runtime = self_runtime.fetch_sub_runtime(proxy_name)
            if child_runtime is None:
                return f"{result}\n[WARN] App proxy '{proxy_name}' not found in channel tree"

            try:
                if timeout == 0:
                    await child_runtime.wait_connected()
                else:
                    await asyncio.wait_for(child_runtime.wait_connected(), timeout=timeout)
                await child_runtime.refresh_metas()
                return f"{result}\n[OK] App channel connected and ready"
            except asyncio.TimeoutError:
                return f"{result}\n[WARN] App started but channel not connected within {timeout}s"

        async def stop(fullname: str) -> str:
            """
            强制停止并卸载一个运行中的 App。
            :param fullname: 目标 App 全名。
            """
            return await self._app_store.stop_app(fullname)

        self._own_commands = {
            'start': PyCommand(start),
            'list_apps': PyCommand(list_apps, always_observe=True),
            'stop': PyCommand(stop),
        }

    @staticmethod
    def make_app_proxy_name(fullname: str) -> str:
        proxy_name = fullname.replace('/', '_')
        return proxy_name

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
        result = {}
        for app in self._app_store.list_apps():
            proxy_name = self.make_app_proxy_name(app.fullname)
            if app.fullname in self._app_channels:
                exists = self._app_channels[app.fullname]
                channels[app.fullname] = exists
                result[proxy_name] = exists
                continue
            channel_proxy = self._matrix.channel_proxy(
                address=app.address,
                name=proxy_name,
                description=app.description,
            )
            channels[app.fullname] = channel_proxy
            result[proxy_name] = channel_proxy
        with self._app_channels_lock:
            self._app_channels = channels
        return result

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
    state = AppStoreChannelState(
        app_store=store,
        matrix=matrix,
        name=name,
        description=description,
    )
    return new_stateful_channel_from_main(state, id=id)
