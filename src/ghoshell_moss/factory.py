from typing import Callable

from ghoshell_moss.core.blueprint.host import MossHost
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = [
    'create_host',
    'create_project',
    'create_matrix',
    'resolve_network',
]


def _create_host(env: Environment, project: Project) -> MossHost:
    raise NotImplementedError


def _create_project(env: Environment) -> Project:
    from ghoshell_moss.project.local_project import LocalProject
    return LocalProject(env)


def _create_matrix(env: Environment, project: Project) -> Matrix:
    """
    matrix.discover 的糖工厂 — worker cell 路径专用 (§ZZ-4 / §ZZ-9 step 7).

    host 路径不走本函数, 走 Host 抽象 concrete 显式构造 (build_self_presence
    只服务 worker, host 独立约定).

    装配流程 (§ZZ-5 IoC 两阶段的 factory 前戏):
      1. 从 env.network 名字 → project.network_metas() 拿 NetworkMetadata
         (兜底 default: driver=zenoh, scope=env.network_scope)
      2. 触发 adapter 模块 import (register_adapter 副作用注册)
      3. 用 driver 找 adapter class (get_adapter_class)
      4. CellManifest.from_proc() — worker cell 从进程反射身份
      5. build_self_presence(env, manifest) — worker 硬编 is_host=False
      6. adapter_cls.from_metadata(...) — driver 私有构造
      7. MatrixImpl 组装
    """
    from ghoshell_moss.core.blueprint.cell import CellManifest, build_self_presence
    # import 副作用触发 register_adapter (zenoh)
    import ghoshell_moss.matrix.networks.zenoh_adapter  # noqa: F401
    from ghoshell_moss.matrix.adapter import get_adapter_class, list_adapter_drivers
    from ghoshell_moss.matrix.matrix_impl import MatrixImpl

    # 1. 解析 network metadata
    network = resolve_network(env, project)

    # 2/3. 查 adapter class
    adapter_cls = get_adapter_class(network.driver)
    if adapter_cls is None:
        raise RuntimeError(
            f"No MatrixNetworkAdapter registered for driver {network.driver!r}. "
            f"Registered drivers: {list_adapter_drivers()}"
        )

    # 4. worker cell 从进程反射
    manifest = CellManifest.from_proc()

    # 5. worker presence (is_host=False 硬编)
    presence = build_self_presence(env, manifest)

    # 6. adapter driver 私有构造 (§ZZ-3 构造签名不进 ABC, 走 from_metadata helper)
    adapter = adapter_cls.from_metadata(network, is_host=presence.is_host)

    # 7. 组装 MatrixImpl
    return MatrixImpl(
        env=env,
        project=project,
        manifest=manifest,
        presence=presence,
        adapter=adapter,
        network=network,
    )


def resolve_network(env: Environment, project: Project) -> NetworkMetadata:
    """
    从 env.network 名字 → project.network_metas() 拿 NetworkMetadata.
    找不到时兜底: driver=zenoh, name/scope 取 env 值.

    host 侧 (Host.matrix concrete) 与 worker 侧 (_create_matrix factory) 共用
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


create_host: Callable[[Environment, Project], MossHost] = _create_host

create_matrix: Callable[[Environment, Project], Matrix] = _create_matrix

create_project: Callable[[Environment], Project] = _create_project
