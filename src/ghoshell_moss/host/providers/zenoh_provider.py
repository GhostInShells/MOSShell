from typing import Type
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.core.blueprint.matrix import Matrix

depend_zenoh()
import zenoh

from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_container import IoCContainer, Provider
from pathlib import Path

__all__ = ['WorkspaceZenohProvider', 'HostEnvZenohProvider']


class WorkspaceZenohProvider(Provider[zenoh.Session]):
    """
    通过 workspace 发现并获取一个 zenoh 的进程级别实例.
    通过进程级容器持有它的生命周期.
    """

    def __init__(
            self,
            workspace_conf_file: str | Path
    ):
        self.config_path = Path(workspace_conf_file)

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[zenoh.Session]:
        return zenoh.Session

    def factory(self, con: IoCContainer) -> zenoh.Session:
        config_path = self.config_path
        # 如果给的是绝对路径, 则默认就是它.
        if not self.config_path.is_absolute():
            # 默认到 workspace 中查找文件.
            # 是相对路径.
            workspace = con.get(Workspace)
            if workspace is not None:
                # 从 workspace 中获取, 不带其它规则了.
                config_path = workspace.configs().abspath().joinpath(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Zenoh config file {config_path} does not exist")

        zenoh_config = zenoh.Config.from_file(config_path)
        session = zenoh.open(zenoh_config)
        return session


class HostEnvZenohProvider(Provider[zenoh.Session]):
    """
    通过 workspace 发现并获取一个 zenoh 的进程级别实例.
    通过进程级容器持有它的生命周期.
    """

    def __init__(
            self,
            app_conf_file: str = 'zenoh_config_app.json5',
            main_conf_file: str = 'zenoh_config_main.json5'
    ):
        self._app_conf_file = Path(app_conf_file)
        self._main_conf_file = Path(main_conf_file)

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[zenoh.Session]:
        return zenoh.Session

    def factory(self, con: IoCContainer) -> zenoh.Session:
        matrix = con.force_fetch(Matrix)
        if matrix.this.type == 'app':
            config_path = self._app_conf_file
        else:
            config_path = self._main_conf_file
        workspace = con.force_fetch(Workspace)
        if workspace is not None:
            # 从 workspace 中获取, 不带其它规则了.
            config_path = workspace.configs().abspath().joinpath(config_path).resolve()

        zenoh_config = zenoh.Config.from_file(config_path)
        session = zenoh.open(zenoh_config)
        return session
