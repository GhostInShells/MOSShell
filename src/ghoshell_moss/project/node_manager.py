"""
ProjectNodeManager — NodeManager 的只读 inventory 实现.

扫描治理域领地内的 NODE.md 声明, 只回答 "领地里可以拉起什么".
不拉起、不杀灭、不持有任何运行时状态.
"""
# TT-11 / UU-9: registry 退化为 glob(NODE.md).
#   spawn 归 matrix.run_node 咽喉; kill 归 project.kill_cell (owner 侧孤儿清理).
#   本实现只保留静态 inventory 扫描 + launcher 装配糖.

from pathlib import Path

from ghoshell_moss.core.blueprint.cell import (
    NodeManager, NodeManifest, NodeLauncher, MatchPattern, ProjectRelativePath,
)
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = ['ProjectNodeManager']


class ProjectNodeManager(NodeManager):
    """扫描指定目录集合, 生成 project-relative 路径到 NodeManifest 的字典视图."""

    # 扫描时跳过的目录名 (常见非 cell 目录, 减少无谓 IO).
    _SKIP_DIRS: set[str] = {'__pycache__', 'node_modules'}

    def __init__(
            self,
            env: Environment,
            cell_dirs: list[Path],
    ):
        self._env = env
        # 默认扫描根目录集合. list_nodes(paths=...) 可 override.
        self._cell_dirs = cell_dirs
        self._cache: dict[ProjectRelativePath, NodeManifest] | None = None

    def list_nodes(
            self,
            refresh: bool = True,
            *,
            paths: list[Path] | None = None,
            installed: bool | None = None,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[ProjectRelativePath, NodeManifest]:
        # paths 显式给定 → override, 不写缓存 (缓存只服务默认根目录集合).
        # 默认路径下的 refresh 才刷缓存.
        if paths is not None:
            scanned = self._scan(paths)
        else:
            if refresh or self._cache is None:
                self._cache = self._scan(self._cell_dirs)
            scanned = self._cache.copy()

        # installed=None → 全部返回 (含未安装, 让 CLI 能提示 INSTALL.md 路径, WW-5 故事 3).
        if installed is not None:
            scanned = {k: v for k, v in scanned.items() if v.installed is installed}
        if include or exclude:
            scanned = dict(self.match_nodes(scanned, include=include, exclude=exclude))
        return scanned

    def get_node(self, relative_path: str | Path) -> NodeManifest | None:
        relative_path = str(relative_path)
        path = self._env.project_path / relative_path
        if path.is_dir():
            return NodeManifest.read_from_directory(path)
        return None

    def get_node_launcher(self, relative_path: str | Path) -> NodeLauncher | None:
        # 薄组合: manifest 反查 → 装 launcher (build_node + dump_cell_env + argv 解析
        # 全部在 NodeLauncher.from_manifest 内). launcher 消费者 (Subprocesses.execute)
        # 负责起进程 + 回填 pid/pgid.
        manifest = self.get_node(relative_path)
        if manifest is None:
            return None
        return NodeLauncher.from_manifest(self._env, manifest)

    def _scan(self, roots: list[Path]) -> dict[ProjectRelativePath, NodeManifest]:
        result: dict[ProjectRelativePath, NodeManifest] = {}
        project_root = self._env.project_path
        for root in roots:
            if not root.is_dir():
                continue
            self._walk(root, project_root, result)
        return result

    def _walk(
            self,
            directory: Path,
            project_root: Path,
            result: dict[ProjectRelativePath, NodeManifest],
    ) -> None:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('.') or entry.name in self._SKIP_DIRS:
                continue
            node_md = entry / NodeManifest.MANIFEST_FILENAME
            if node_md.is_file():
                try:
                    manifest = NodeManifest.read_from_file(node_md)
                except Exception:
                    manifest = None
                if manifest is not None:
                    relative = str(entry.relative_to(project_root))
                    result[relative] = manifest
                    continue  # node 目录内部不再递归 (一个目录一个身份锚)
            self._walk(entry, project_root, result)
