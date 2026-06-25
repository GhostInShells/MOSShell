from typing import Callable

from ghoshell_moss.core.blueprint.host import MossHost
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = [
    'create_host',
    'create_project',
    'create_matrix',
]


def _create_host(env: Environment, project: Project) -> MossHost:
    raise NotImplementedError


def _create_project(env: Environment) -> Project:
    from ghoshell_moss.project.local_project import LocalProject
    return LocalProject(env)


def _create_matrix(env: Environment, project: Project) -> Matrix:
    pass


create_host: Callable[[Environment, Project], MossHost] = _create_host

create_matrix: Callable[[Environment, Project], Matrix] = _create_matrix

create_project: Callable[[Environment], Project] = _create_project
