"""
Host — MOSS 顶层入口. 从环境发现 project + mode + matrix, 编排 MossRuntime / GhostRuntime.

wire-up 契约: §ZZ 全套 + §YY-2 (HostMode.cells 删, project.cells 一条链路) +
§ZZ-4 (build_host_presence 走 Host 抽象 concrete, 不走 factory).

Host 侧 Matrix concrete 构造路径:
    build_host_presence(env) + create_matrix_helper(env, project, is_host=True)
    → 与 worker 路径 (factory._create_matrix) 分开约定, 避免 build_self_presence
      god function 化 (§ZZ-4).
"""
from typing_extensions import Self

import importlib
import pathlib

from ghoshell_moss.core.blueprint.host import MossHost, MossRuntime, GhostRuntime
from ghoshell_moss.core.blueprint.ghost import GhostMeta
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.cell import CellManifest, build_host_presence
from ghoshell_moss.contracts.workspace import LocalWorkspace

from ghoshell_moss.matrix.matrix_impl import MatrixImpl
from ghoshell_moss.matrix.adapter import get_adapter_class, list_adapter_drivers
from ghoshell_moss.factory import resolve_network

from ghoshell_moss.host.moss_runtime import MossRuntimeImpl
from ghoshell_moss.host.ghost_runtime import GhostRuntimeImpl

__all__ = ['Host']

_host_instance: 'Host | None' = None


class Host(MossHost):
    """MOSS 顶层入口的 concrete."""

    def __init__(
            self,
            *,
            env: Environment | None = None,
    ):
        # §UU-1 seal 定案: Environment 无 set_*, 参数一次性塞 __init__, seal 一次性事实.
        # 入口点 (CLI callback / moss-as-mcp / moss-repl) 负责构造 + seal, Host 只做消费.
        # 库直接使用 (Host()) 走 Environment.discover(bootstrap=True), 构造裸 env + seal.
        # Host 不承担 CLI 参数收集责任 (那是入口的活), 也不重复 seal (§UU-1 一次性).
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
        from ghoshell_moss.factory import create_project
        self._project = create_project(self._env)
        self._project.bootstrap()

        # workspace: mode 无关的项目 workspace (matrix 层已经从 project 拿到,
        # 这里额外持一份供 MossRuntimeImpl.__init__ 使用).
        self._workspace = LocalWorkspace(self._env.workspace_path)

        # Matrix concrete 单例 — 首次 matrix() 调用时构造, 后续复用.
        self._matrix: MatrixImpl | None = None

    def name(self) -> str:
        return self._env.moss_meta.name

    def description(self) -> str:
        return self._env.moss_meta.description

    @classmethod
    def discover(cls, env: Environment | None = None) -> Self:
        global _host_instance
        if _host_instance is None:
            _host_instance = Host(env=env)
        return _host_instance

    def reboot(self) -> Self:
        global _host_instance
        _host_instance = None
        new_host = Host(env=self._env)
        _host_instance = new_host
        return new_host

    @property
    def env(self) -> Environment:
        return self._env

    @property
    def project(self) -> Project:
        return self._project

    # scan_errors 已作废: TUI 直接 walk host.project 的 manifests 通路即可
    # (Manifest 自持异常载体). Host 不背 alerts 汇聚这份担子 —
    # 未来 alerts 归 TopicService ringbuffer (tui.py L613 承诺).

    def matrix(self) -> MatrixImpl:
        """Host 侧 Matrix concrete — build_host_presence + adapter registry (§ZZ-4).

        单进程内单例: 首次调用构造并缓存, 后续调用返回同一实例.

        与 factory._create_matrix (worker cell 路径) 的差别只在 presence 构造:
          host → build_host_presence(env), address='host/{moss_name}/{project_id_short}'
          worker → build_self_presence(env, manifest), address='cell/{name}/{uid}'
        adapter driver / network 解析 / IoC 装配全走同一套.
        """
        if self._matrix is not None:
            return self._matrix

        # register_adapter 副作用触发 (与 factory._create_matrix 保持一致)
        import ghoshell_moss.matrix.networks.zenoh_adapter  # noqa: F401

        network = resolve_network(self._env, self._project)
        adapter_cls = get_adapter_class(network.driver)
        if adapter_cls is None:
            raise RuntimeError(
                f"No MatrixNetworkAdapter registered for driver {network.driver!r}. "
                f"Registered drivers: {list_adapter_drivers()}"
            )

        # host 的 "manifest" 用 moss_meta 承接身份 (§ZZ-4 host 独立约定, 不走 CELL.md).
        # name 是 home 稳定身份键 (§YY-1 第 6 条), 用 moss_meta.name 作 host 身份锚.
        manifest = CellManifest(
            name=self._env.moss_meta.name or 'host',
            description=self._env.moss_meta.description or '',
            installed=True,
        )
        presence = build_host_presence(self._env)
        adapter = adapter_cls.from_metadata(network, is_host=True)

        self._matrix = MatrixImpl(
            env=self._env,
            project=self._project,
            manifest=manifest,
            presence=presence,
            adapter=adapter,
            network=network,
        )
        return self._matrix

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

        matrix = self.matrix()
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
            module = importlib.import_module(ghost_meta.import_path()) if hasattr(ghost_meta, 'import_path') and callable(getattr(ghost_meta, 'import_path', None)) else None
            source_path = pathlib.Path(module.__file__).parent.absolute() if module is not None else None
        elif isinstance(ghost, GhostMeta):
            ghost_meta = ghost
            source_path = None
        else:
            raise ValueError(f"invalid ghost argument type {type(ghost)}")

        # env.ghost_name 必须在 run() 之前 seal 前构造时设置; 已 seal 后无路径
        # (Environment.set_ghost_name 已删, seal 是一次性跃迁). host 侧无法在
        # 已 seal env 上改 ghost_name — Ghost 归属由 env 构造时决定.
        # (moss_as_mcp / moss-run-ghost 应通过 Environment(ghost=...) 传入)
        moss_runtime = self.run(run_shell=run_shell)
        return GhostRuntimeImpl(
            moss_runtime=moss_runtime,
            ghost_meta=ghost_meta,
            source_path=source_path,
        )
