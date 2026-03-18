from ghoshell_common.contracts.workspace import LocalWorkspace
from typing_extensions import Self
from os.path import abspath, join
from pathlib import Path
import loadenv


class AtomWorkspace:
    """
    Atom 默认的 workspace.
    """

    def __init__(self, atom_workspace_dir: Path) -> None:
        self._root = atom_workspace_dir.resolve()

    def assets(self) -> Path:
        """
        assets path
        """
        return self._root.joinpath("assets").resolve()

    def memory(self) -> Path:
        return self._root.joinpath("memory").resolve()

    def env_file(self) -> Path:
        return self._root.joinpath(".env").resolve()

    @classmethod
    def init_from_env(cls) -> Self:
        """
        从 env 初始化.
        """
        raise NotImplementedError("todo")

    @classmethod
    def load_env(cls, env_file: str) -> None:
        """
        load env file from workspace
        """
        raise NotImplementedError("todo")
