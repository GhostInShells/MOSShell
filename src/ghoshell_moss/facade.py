from typing import Callable

from ghoshell_moss.core.blueprint.host import MossHost
from ghoshell_moss.core.blueprint.environment import Environment

__all__ = ['discover_host']


def _discover_host(env: Environment | None = None) -> MossHost:
    from ghoshell_moss.host import Host
    return Host.discover(env)


discover_host: Callable[[Environment | None], MossHost] = _discover_host
