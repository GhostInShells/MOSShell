from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable
from typing_extensions import Self

from pathlib import Path
import frontmatter

if TYPE_CHECKING:
    from circus.watcher import Watcher

from pydantic import BaseModel, Field


class AppWatcher(BaseModel):
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
    work_directory: str = Field(
        description="The work directory of the app",
    )
    watcher: AppWatcher = Field(
        default_factory=AppWatcher,
        description='The app watcher',
    )

    @property
    def address(self) -> str:
        return f"app/{self.group}/{self.name}"

    @property
    def log_name(self) -> str:
        return f"moss.{self.group}.{self.name}"

    def to_circus_watcher(self, env: dict[str, str], arguments: str = '') -> "Watcher":
        from circus.watcher import Watcher
        return Watcher(
            name=self.address,
            cmd=' '.join([self.watcher.cmd, arguments]),
            numprocesses=self.watcher.workers,
            env=env,
            working_dir=self.work_directory,
        )

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

    def as_markdown(self) -> str:
        post = frontmatter.Post(
            content=self.docstring,
            **self.watcher.model_dump(exclude_none=True, exclude_defaults=True),
        )
        return frontmatter.dumps(post)

    @classmethod
    def from_apps_directory(cls, apps_directory: Path, filename: str = "APP.md") -> Iterable[Self]:
        """
        从指定的路径寻找.
        """
        for app_group in apps_directory.iterdir():
            for app_dir in apps_directory.iterdir():
                expect_app_manifest = app_dir.joinpath(filename)
                if expect_app_manifest.exists() and expect_app_manifest.is_file():
                    group = app_group.name
                    app_name = app_dir.name
                    yield cls.from_markdown(group, app_name, expect_app_manifest)


class AppStore(ABC):
    """
    local appstore
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def directory(self) -> Path:
        pass

    @abstractmethod
    def list_apps(self) -> Iterable[AppInfo]:
        pass

    @abstractmethod
    def running_apps(self) -> Iterable[AppInfo]:
        pass

    @abstractmethod
    async def start_app(self, app_address: str, argument: str = '') -> str:
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        pass

    @abstractmethod
    async def stop_app(self, app_address: str) -> None:
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
