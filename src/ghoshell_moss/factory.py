from typing import Callable

from ghoshell_moss.core.blueprint.cell import Cell, CellRuntimeInfo
from ghoshell_moss.core.blueprint.host import IHost
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = [
    'create_host',
    'create_project',
    'create_matrix',
    'resolve_network',
    'resolve_matrix_adapter',
]


def _create_host(env: Environment, project: Project) -> IHost:
    raise NotImplementedError


def _create_project(env: Environment) -> Project:
    from ghoshell_moss.project.local_project import LocalProject
    return LocalProject(env)


def _create_matrix(env: Environment, project: Project, cell: CellRuntimeInfo | None = None) -> Matrix:
    """
    Matrix.discover() 走的 patch escape hatch — worker path 专用.

    Host 有 env + 自造 cell 语境, 不走本函数, 走 Host.new_matrix(cell) concrete 构造.
    本函数只服务无上下文的入口 (纯 worker cell 进程自我发现).

    装配:
      1. discover_this_node(env): 有 ledger 读盘 (spawn path), 无则 from_proc 认亲 (bare script)
      2. resolve_matrix_adapter(env, project, cell): 网络元信息 + adapter 实例
      3. MatrixImpl 组装
    """
    from ghoshell_moss.core.blueprint.cell import discover_this_node
    from ghoshell_moss.matrix.matrix_impl import MatrixImpl

    runtime_info = cell or discover_this_node(env)
    adapter, network = resolve_matrix_adapter(env, project, cell=runtime_info.cell)
    return MatrixImpl(
        env=env,
        project=project,
        runtime_info=runtime_info,
        adapter=adapter,
        network=network,
    )


def resolve_network(env: Environment, project: Project) -> NetworkMetadata:
    """
    从 env.network 名字 → project.network_metas() 拿 NetworkMetadata.
    找不到时兜底: driver=zenoh, name/scope 取 env 值.

    host 侧 (Host.new_matrix concrete) 与 worker 侧 (_create_matrix factory) 共用
    本函数, 保证网络接线一致.
    """
    all_networks = project.network_metas()
    metadata = all_networks.get(env.network)
    if metadata is not None:
        return metadata
    # 兜底 default — 供裸 workspace (无 networks/*.json) 场景直接可跑
    return NetworkMetadata(
        name=env.network,
        driver='zenoh',
        scope=env.network_scope,
    )


def resolve_matrix_adapter(
        env: Environment,
        project: Project,
        *,
        cell: Cell,
):
    """
    Matrix 构造的共享装配步骤 — network 元信息解析 + adapter class 查询 + adapter 实例化.

    Host.new_matrix (concrete) 与 factory._create_matrix (worker) 共同调用,
    保证两条路径拿到的 adapter/network 完全一致.
    :return: (adapter 实例, network 元信息) — 未 __aenter__.
    """
    # register_adapter 副作用触发 (未来分驱动时按 driver 名条件 import)
    import ghoshell_moss.matrix.networks.zenoh_adapter  # noqa: F401
    from ghoshell_moss.matrix.adapter import get_adapter_class, list_adapter_drivers

    network = resolve_network(env, project)
    adapter_cls = get_adapter_class(network.driver)
    if adapter_cls is None:
        raise RuntimeError(
            f"No MatrixNetworkAdapter registered for driver {network.driver!r}. "
            f"Registered drivers: {list_adapter_drivers()}"
        )
    adapter = adapter_cls.from_metadata(network, is_host=cell.is_host)
    return adapter, network


create_host: Callable[[Environment, Project], IHost] = _create_host

create_matrix: Callable[[Environment, Project, CellRuntimeInfo | None], Matrix] = _create_matrix

create_project: Callable[[Environment], Project] = _create_project
