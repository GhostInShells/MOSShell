from ghoshell_moss.core.concepts.command import Observe

__all__ = ["observe"]


async def observe() -> Observe:
    """
    force to observe
    """
    return Observe()
