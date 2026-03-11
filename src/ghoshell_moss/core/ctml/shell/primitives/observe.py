from ghoshell_moss.types import Observe

__all__ = ["observe"]


async def observe() -> Observe:
    """
    force to observe
    """
    return Observe()
