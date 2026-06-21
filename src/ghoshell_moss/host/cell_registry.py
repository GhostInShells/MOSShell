import asyncio
import os
from pathlib import Path

from ghoshell_moss.core.blueprint.cell import Cell, CellRegistry, CellManifest, MatchPattern, RelativePath
from ghoshell_moss.core.blueprint.environment import Environment


class EnvCellRegistry(CellRegistry):

    def __init__(
            self,
            env: Environment,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ):
        self._env = env
        self._cells_registry_dir = self._env.workspace_cell_registry.abspath()
        self._include = include or []
        self._exclude = exclude or []

    def root(self) -> Path:
        return self._cells_registry_dir

    def list_cell_manifests(
            self,
            refresh: bool = True,
            *,
            installed: bool = True,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[RelativePath, CellManifest]:
        result: dict[RelativePath, CellManifest] = {}
        registry_dir = self._cells_registry_dir
        if not registry_dir.is_dir():
            return result

        include_pats = include or self._include
        exclude_pats = exclude or self._exclude

        for group_dir in sorted(registry_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            for name_dir in sorted(group_dir.iterdir()):
                if not name_dir.is_dir():
                    continue
                cell_md = name_dir / CellManifest.MANIFEST_FILENAME
                if not cell_md.is_file():
                    continue
                manifest = CellManifest.read_from_file(cell_md)
                if manifest is None:
                    continue
                if installed and not manifest.installed:
                    continue
                relative = '/'.join([group_dir.name, name_dir.name])
                result[relative] = manifest

        # apply include/exclude via the static matcher
        if include_pats or exclude_pats:
            filtered = dict(CellRegistry.match_cells(
                result, include=include_pats or None, exclude=exclude_pats or None,
            ))
            return filtered

        return result

    def get_cell_manifest(self, relative_path: str) -> CellManifest | None:
        path = self._cells_registry_dir / relative_path
        if not path.exists():
            return None
        return CellManifest.read_from_directory(path)

    def local_runtime_cells(self) -> list[Cell]:
        result = []
        for cell in Cell.find_runtime_cells(self._env.runtime_registry_dir, throw=False):
            result.append(cell)
        return result

    def add_cell_runtime(self, cell: Cell) -> None:
        cell.write_runtime_file(self._env.runtime_registry_dir)

    def remove_cell_runtime(self, address: str) -> bool:
        filename = Cell.make_runtime_filename(address)
        file = self._env.runtime_registry_dir.joinpath(filename)
        if file.exists():
            file.unlink()
            return True
        return False

    def get_cell_runtime(self, address: str) -> Cell | None:
        filename = Cell.make_runtime_filename(address)
        file = self._env.runtime_registry_dir.joinpath(filename)
        if not file.exists():
            return None
        return Cell.read_from_runtime_file(file)

    def discover_current_cell(self) -> Cell:
        """
        对于 host 节点, cell 是构造的; 非 host 节点, cell 是发现的.
        """
        this_cell_address = self._env.this_cell_address
        cell = None
        if this_cell_address:
            cell = self.get_cell_runtime(this_cell_address)
            if cell and cell.status.pid is not None and cell.status.pid != os.getpid():
                # 不是同一个节点.
                cell = None

        if cell is None:
            cell = Cell.from_proc()
        if cell.status.pid is None:
            cell.status.pid = os.getpid()

        return cell

    def dump_spawn_env(self, address: str) -> dict[str, str]:
        return self._env.dump_moss_env(
            cell_address=address,
            parent_cell_address=self._env.this_cell_address,
            with_os_env=False,
        )

    def cell_runtime_exists(self, address: str) -> bool:
        """判断一个 cell 的 runtime 文件是否存在. """
        filename = Cell.make_runtime_filename(address)
        filepath = self._env.runtime_registry_dir.joinpath(filename)
        return filepath.exists()
