"""
ProjectCellRegistry — 只读 inventory 实现.

扫描治理域领地内的 CELL.md 声明, 只回答 "领地里装了什么".
不拉起、不杀灭、不持有任何运行时状态.
"""
# -- TT-11 / UU-9: registry 退化为 glob(CELL.md).
#    原 spawn_cell / kill_all_runtime_cells / dump_spawn_env / cell_runtimes_dir
#    已在 blueprint 层删除 — spawn 归 run_cell 咽喉, kill 归 CLI (ledger 唯一读者).
#    本实现只保留扫描与查表.

from pathlib import Path

from ghoshell_moss.core.blueprint.cell import (
    CellRegistry, CellManifest, MatchPattern, RelativePath,
)
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = ['ProjectCellRegistry']


class ProjectCellRegistry(CellRegistry):

    # 扫描时跳过的目录名
    _SKIP_DIRS: set[str] = {'__pycache__', 'node_modules'}

    def __init__(
            self,
            env: Environment,
            cell_dirs: list[Path],
    ):
        self._env = env
        self._cell_dirs = cell_dirs
        self._cache: dict[RelativePath, CellManifest] | None = None

    def list_cell_manifests(
            self,
            refresh: bool = True,
            *,
            installed: bool | None = None,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[RelativePath, CellManifest]:
        if refresh or self._cache is None:
            self._cache = self._scan()

        result = self._cache.copy()
        # installed=None → 全部返回 (含未安装, 让 CLI 能提示 INSTALL.md 路径, WW-5 故事 3).
        if installed is not None:
            result = {k: v for k, v in result.items() if v.installed is installed}
        if include or exclude:
            result = dict(self.match_cells(result, include=include, exclude=exclude))
        return result

    def get_cell_manifest(self, relative_path: str | Path) -> CellManifest | None:
        relative_path = str(relative_path)
        path = self._env.project_path / relative_path
        if path.is_dir():
            return CellManifest.read_from_directory(path)
        return None

    def _scan(self) -> dict[RelativePath, CellManifest]:
        result: dict[RelativePath, CellManifest] = {}
        project_root = self._env.project_path
        for cell_dir in self._cell_dirs:
            if not cell_dir.is_dir():
                continue
            self._walk(cell_dir, project_root, result)
        return result

    def _walk(
            self,
            directory: Path,
            project_root: Path,
            result: dict[RelativePath, CellManifest],
    ) -> None:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('.') or entry.name in self._SKIP_DIRS:
                continue
            cell_md = entry / CellManifest.MANIFEST_FILENAME
            if cell_md.is_file():
                try:
                    manifest = CellManifest.read_from_file(cell_md)
                except Exception:
                    manifest = None
                if manifest is not None:
                    relative = str(entry.relative_to(project_root))
                    result[relative] = manifest
                    continue  # cell 目录内部不再递归
            self._walk(entry, project_root, result)
