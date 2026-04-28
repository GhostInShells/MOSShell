from abc import ABC, abstractmethod
from typing import Iterable, Optional
from typing_extensions import Self, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from ghoshell_moss.core.blueprint.channel_builder import Channel, new_channel
import frontmatter
import fnmatch

__all__ = [
    'AppInfo',
    'AppWatcher',
    'AppState',
    'AppStore',
]


class AppWatcher(BaseModel):
    """
    启动和管理 app 运行状态的对象.
    """

    cmd: str = Field(
        default='uv run main.py',
        description='The command to execute',
    )
    description: str = Field(
        default='',
        description='The description of the app',
    )
    respawn: bool = Field(
        default=False,
        description="respawn the app if it closed."
    )
    workers: int = Field(
        default=1,
        description='The number of the app workers',
    )
    max_age: int | None = Field(
        default=None,
        description='The maximum age (seconds) of the app to restart',
    )


AppState = Literal['stopped', 'starting', 'running', 'error']


class AppInfo(BaseModel):
    """
    环境中可发现的 app 应用.
    """
    name: str = Field(
        description='The name of the current app',
        pattern=r'^[a-zA-Z0-9_]+$',
    )
    group: str = Field(
        description='The group of the current app',
        pattern=r'^[a-zA-Z0-9_]+$',
    )
    description: str = Field(
        default='',
        description='The description of the current app.',
    )
    docstring: str = Field(
        default='',
        description='The docstring of the current app',
    )
    is_running: bool = Field(
        default=False,
        description='判断 app 是否在运行中. ',
    )
    state: AppState | str = Field(
        default='',
        description='The state of the app',
    )
    error: str = Field(
        default='',
        description='The error message of the app if in error state',
    )
    work_directory: str = Field(
        description="The work directory of the app",
    )
    watcher: AppWatcher = Field(
        default_factory=AppWatcher,
        description='The app watcher',
    )

    @property
    def address(self) -> str:
        return f"apps/{self.group}/{self.name}"

    @property
    def fullname(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def log_name(self) -> str:
        return f"moss.{self.group}.{self.name}"

    def to_circus_params(self, env: dict[str, str], arguments: str = '') -> dict:
        """
        将 AppInfo 转换为 Circus add 指令所需的参数属性
        """
        return {
            "name": self.address,
            "cmd": ' '.join([self.watcher.cmd, arguments]).strip(),
            "working_dir": self.work_directory,
            "numprocesses": self.watcher.workers,
            "respawn": self.watcher.respawn,
            "max_age": self.watcher.max_age,
            "env": env,
            "singleton": True,
        }

    @classmethod
    def from_markdown(cls, group: str, name: str, file: Path) -> Self:
        """
        约定的 markdown 方式
        """
        if not file.is_file() or not file.exists():
            raise FileNotFoundError(f"The file {file} does not exist.")
        post = frontmatter.loads(file.read_text())
        watcher_data = post.metadata
        watcher = AppWatcher(**watcher_data)
        workspace_dir = str(file.parent.absolute())
        docstring = post.content
        description = watcher.description or post.content.split('\n')[0]
        return cls(
            watcher=watcher,
            name=name,
            group=group,
            description=description,
            docstring=docstring,
            work_directory=workspace_dir,
        )

    def as_markdown(
            self,
    ) -> str:
        post = frontmatter.Post(
            content=self.docstring,
            **self.watcher.model_dump(exclude_none=True, exclude_defaults=False),
        )
        return frontmatter.dumps(post)

    @classmethod
    def from_apps_directory(cls, apps_directory: Path, filename: str = "APP.md") -> Iterable[Self]:
        """
        从指定的路径寻找.
        """
        for app_group in apps_directory.iterdir():
            if not app_group.is_dir():
                continue
            for app_dir in app_group.iterdir():
                expect_app_manifest = app_dir.joinpath(filename)
                if expect_app_manifest.exists() and expect_app_manifest.is_file():
                    group = app_group.name
                    app_name = app_dir.name
                    yield cls.from_markdown(group, app_name, expect_app_manifest)


AppFullname = str
AppFullnamePattern = str
""" group/name,  group/*, *, */*, */name"""


class AppStore(ABC):
    """
    local appstore
    """

    # 非运行时函数

    @abstractmethod
    def name(self) -> str:
        """
        App store 的名字, 通常就是 apps.
        """
        pass

    @abstractmethod
    def list_groups(self) -> list[str]:
        """
        对 App 的分组, 通常是 apps 目录下的一级目录.
        """
        pass

    @abstractmethod
    def list_apps(self, refresh: bool = False) -> Iterable[AppInfo]:
        """
        猎取环境中发现的每个 App, 通常拥有自己的独立目录.
        :param refresh: 是否刷新检查环境里的 apps.
        """
        pass

    @classmethod
    def match_apps(
            cls,
            apps: Iterable[AppInfo],
            include: list[AppFullnamePattern] | None = None,
            exclude: Optional[list[AppFullnamePattern]] = None
    ) -> Iterable[AppInfo]:
        """
        基于地址模式筛选 App。
        支持通配符:
        - 'group/app_name' (精确匹配)
        - 'group/*' (组内全选)
        - '*/app_name' (跨组选同名)
        """
        include_patterns = set(include) if include is not None else {"*/*"}
        if len(include_patterns) == 0:
            return

        exclude_patterns = set(exclude or [])
        for app in apps:
            address = app.address  # "app/group/name"

            # 1. 检查是否在包含范围内
            # 使用 fnmatch 实现标准的 Unix 通配符逻辑，比 startswith 更强大
            is_included = any(
                fnmatch.fnmatch(address, pat) or fnmatch.fnmatch(address, f"app/{pat}")
                for pat in include_patterns
            )

            if not is_included:
                continue

            # 2. 检查是否被排除
            is_excluded = any(
                fnmatch.fnmatch(address, pat) or fnmatch.fnmatch(address, f"app/{pat}")
                for pat in exclude_patterns
            )

            if not is_excluded:
                yield app

    @abstractmethod
    def init_app(self, fullname: str, description: str = '') -> str:
        """
        创建一个 app, 返回创建后的讯息.
        创建 app 的极简内容包含:
        1. 创建目录.
        2. 定义 APP.md (如果基于 markdown 范式)
        3. 定义 helloworld 的 main.py 脚本.
        """
        pass

    # 运行时函数

    @abstractmethod
    def get_app_info(self, fullname: str) -> AppInfo | None:
        """
        获取一个环境中可发现的 app.
        如果 running 为 True, 则需要发现 is alive 的 app.
        """
        pass

    @abstractmethod
    async def get_apps_context(self) -> str:
        """
        通过文本描述目前 apps 的状态. 包含:
        1. 发现的所有 apps, 他们的名称/ address 和描述. 不包含路径信息.
        2. 如果是运行时, 添加上运行状态的信息.
        """
        pass

    @abstractmethod
    async def start_app(self, app_fullname: str, argument: str = '') -> str:
        """
        尝试启动一个 App.
        其中 argument 是可以在启动脚本后附加的参数.
        返回描述信息.
        """
        pass

    @abstractmethod
    async def stop_app(self, app_fullname: str) -> str:
        """
        关闭一个指定的 app.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        判断 app store 是否在运行状态中.
        """
        pass


def build_apps_channel(store: AppStore, description: str = '') -> Channel:
    """
    构建 App 管理中心通道。
    该通道允许 AI 发现、启动、停止和初始化物理/逻辑应用 (Apps)。
    """
    # 默认描述强调“中心化管理”
    default_description = (
        "App Store 核心通道，用于管理当前环境下的所有可用应用。"
        "你可以通过此通道拉起具有特定功能的子进程（如机器人控制、数据分析等）。"
    )

    name = store.name()
    chan = new_channel(name=name, description=description or default_description)

    @chan.build.command(name="list")
    async def list_apps() -> str:
        """
        获取当前环境所有可发现 App 的详细清单及运行状态。
        AI 在尝试启动任何 App 前，应先通过此命令确认其 address 和当前状态。
        """
        return await store.get_apps_context()

    @chan.build.command(name="start")
    async def start(fullname: str, argument: str = "") -> str:
        """
        启动指定的 App。
        :param fullname: App 的完整名称，如 'group/name'。
        :param argument: 启动参数，将作为命令行参数传递给 App。
        注意：启动是异步的，可以通过 list 确认是否成功进入 running 状态。
        """
        return await store.start_app(fullname, argument)

    @chan.build.command(name="stop")
    async def stop(fullname: str) -> str:
        """
        强制停止并卸载一个运行中的 App。
        :param fullname: 目标 App 全名。
        """
        return await store.stop_app(fullname)

    @chan.build.command(name="init")
    async def init(fullname: str, description: str = "") -> str:
        """
        在工作空间中初始化一个新的 App 模板。
        会自动创建目录、APP.md 和 main.py 骨架。
        :param fullname: 期望的地址格式 'group/name'。
        :param description: App 的功能描述。
        """
        # 这里调用我们之前实现的 init_app
        return store.init_app(fullname, description)

    @chan.build.context_messages
    async def apps_status() -> str:
        """
        动态注入当前已发现 App 的状态简报到 AI 的上下文。
        确保 AI 始终知晓哪些 App 正在运行 (RUNNING) 及其潜在的错误 (ERROR)。
        """
        context_str = await store.get_apps_context()
        header = "### [App Runtime Status]\n"
        footer = "\n---\n注：若 App 处于 ERROR 状态，请检查日志或尝试重启。"
        return header + context_str + footer

    return chan
