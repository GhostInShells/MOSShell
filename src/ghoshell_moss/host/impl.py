"""
Host — MOSS 顶层入口. 从环境发现 project + mode, 编排 MossRuntime / GhostRuntime.

wire-up 契约:
- Host 侧 Matrix 走 concrete: build_host_cell → CellRuntimeInfo → new_matrix(cell),
  不经 factory (factory 是 patch escape hatch, 服务无上下文的 Matrix.discover).
- new_matrix(cell) 显式收 cell — 未来 host 分化 (ghost/shell) 时, 分歧点全在
  caller 构造的 cell, matrix 装配对分化无感.
- Host 不缓存 matrix: 一个 Host 生命周期里 run / run_ghost 只调一次,
  返回的 matrix 由 MossRuntimeImpl / GhostRuntimeImpl 持有生命周期.
"""
from typing_extensions import Self

import pathlib
from pathlib import Path

from ghoshell_moss.core.blueprint.host import MossHost, MossRuntime, GhostRuntime
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.cell import Cell, CellRuntimeInfo, build_host_cell
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.contracts.workspace import LocalWorkspace

from ghoshell_moss.matrix.matrix_impl import MatrixImpl
from ghoshell_moss.factory import create_project, resolve_matrix_adapter

from ghoshell_moss.host.moss_runtime import MossRuntimeImpl
from ghoshell_moss.host.ghost_runtime import GhostRuntimeImpl

__all__ = ['Host']


class Host(MossHost):
    """MOSS 顶层入口的 concrete.

    显式构造姿态 (§UU-1 seal 判决 + 用户 2026-07 明示"host 不走 discover 而是走正常 init"):
    - 入口点 (CLI callback / moss-as-mcp / moss-repl / tui) 负责构造 Environment(**cli_args)
      + seal(), 然后 Host(env=env).
    - 库直接使用 (Host()) 走 Environment.discover(bootstrap=True), 构造裸 env + seal.
    - 无 module 级单例, 无 Host.discover classmethod — 每次 Host(env=env) 是新实例.
      需要"全局唯一 host"语义时, 由调用方 (通常是 MossRuntime 或 GhostRuntime) 自持引用.
    """

    def __init__(
            self,
            *,
            env: Environment | None = None,
    ):
        if env is None:
            env = Environment.discover()
        if not env.is_sealed:
            raise RuntimeError(
                "Host requires a sealed Environment. "
                "Entry-point should construct Environment(**cli_args) + seal() first, "
                "then pass to Host(env=env). "
                "Library-direct users can call Environment.discover() which auto-seals."
            )
        self._env = env

        # Project: 通过 factory.create_project 拿 LocalProject 实例, bootstrap 会
        # 加载 .env / 挂 moss.log handler / 注册全局单例.
        self._project = create_project(self._env)
        self._project.bootstrap()

        # workspace: mode 无关的项目 workspace (matrix 层已经从 project 拿到,
        # 这里额外持一份供 MossRuntimeImpl.__init__ 使用).
        self._workspace = LocalWorkspace(self._env.workspace_path)

    def name(self) -> str:
        return self._env.moss_meta.name

    def description(self) -> str:
        return self._env.moss_meta.description

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def project(self) -> Project:
        return self._project

    def new_matrix(self, cell: Cell) -> Matrix:
        """基于给定 cell 构造未启动的 Matrix.

        cell 由 caller 显式构造 (build_host_cell / 未来 build_*_cell 分化),
        这里只负责: CellRuntimeInfo 组装 + adapter/network 解析 + MatrixImpl 装配.

        每次调用返回新实例, Host 不缓存. 调用方 (MossRuntimeImpl 等) 自持生命周期.
        """
        runtime_info = CellRuntimeInfo(
            address=cell.address,
            pid=self._env.pid,
            cell=cell,
        )
        adapter, network = resolve_matrix_adapter(
            self._env, self._project, cell=cell,
        )
        return MatrixImpl(
            env=self._env,
            project=self._project,
            runtime_info=runtime_info,
            adapter=adapter,
            network=network,
        )

    def run(
            self,
            *,
            run_shell: bool = True,
            name: str | None = None,
            description: str | None = None,
    ) -> MossRuntime:
        mode = self._project.current_mode()
        if mode is None:
            raise RuntimeError(
                f"No mode available (env.mode_name={self._env.mode_name!r}, "
                f"env.no_mode={self._env.no_mode}). "
                f"Set --mode or ensure the workspace has at least one HOST.md."
            )
        mode.bootstrap()

        cell = build_host_cell(self._env)
        matrix = self.new_matrix(cell)
        return MossRuntimeImpl(
            env=self._env,
            workspace=self._workspace,
            mode=mode,
            matrix=matrix,
            run_shell_on_start=run_shell,
            name=name,
            description=description,
        )

    def run_ghost(
            self,
            ghost: 'str | GhostMeta',
            *,
            run_shell: bool = True,
    ) -> GhostRuntime:
        if isinstance(ghost, str):
            ghost_meta = self._project.get_ghost(ghost)
            source_path = self._find_ghost_source(ghost_meta)
        elif isinstance(ghost, GhostMeta):
            ghost_meta = ghost
            # 直接传实例 = 测试/自定义构造场景, 无发现路径, 源码定位由 caller 自负.
            source_path = None
        else:
            raise TypeError(
                f"run_ghost: expected str or GhostMeta, got {type(ghost).__name__}"
            )

        # env.ghost_name 必须在 seal 前构造 Environment 时设置; 已 seal 后无路径
        # (Environment.set_ghost_name 已删, seal 是一次性跃迁). host 侧无法在
        # 已 seal env 上改 ghost_name — Ghost 归属由 env 构造时决定.
        # (moss_as_mcp / moss-run-ghost 应通过 Environment(ghost=...) 传入)
        moss_runtime = self.run(run_shell=run_shell)
        return GhostRuntimeImpl(
            moss_runtime=moss_runtime,
            ghost_meta=ghost_meta,
            source_path=source_path,
        )

    def _find_ghost_source(self, ghost_meta: GhostMeta) -> Path | None:
        """从 project.ghosts() 迭代反查 ghost 源码目录.

        LocalProject.ghosts() 产出 (found_at, meta) tuple, found_at 是 ghost
        源文件绝对路径. .parent = 源码目录, 供 GhostWorkspace.source 消费.
        get_ghost() 只返回 meta, 反查是这里的责任; identity 比较避免同名歧义.
        """
        for path, meta in self._project.ghosts():
            if isinstance(meta, Exception):
                continue
            if meta is ghost_meta:
                return pathlib.Path(path).parent.absolute()
        return None
