from typing import Literal, Type
from ghoshell_container import INSTANCE, IoCContainer
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.host import MossHost

__all__ = ['get_container', 'get_contract']


def get_container(source: Literal['project', 'matrix', 'host'] = 'project') -> IoCContainer:
    # 三种做法都没有真正启动.
    if source == 'project':
        container = Project.discover().container
    elif source == 'matrix':
        container = Matrix.discover().container
    elif source == 'host':
        container = MossHost.discover().run(run_shell=False).container
    else:
        raise ValueError(f"Unknown container source: {source}")
    container.bootstrap()
    return container


def get_contract(abstract: Type[INSTANCE], source: Literal['project', 'matrix', 'host'] = 'project') -> INSTANCE | None:
    """根据 cli 的参数约定, 取出特定的 contract. 符合 moss 所有组件化 debug 手段."""
    container = get_container(source)
    return container.get(abstract)
