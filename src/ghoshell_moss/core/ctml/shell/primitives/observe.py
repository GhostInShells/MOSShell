from ghoshell_moss.core.concepts.command import Observe

__all__ = ["observe"]


async def observe() -> Observe:
    """
    Insert an observation point to ensure the current logos stream is observed.
    """
    return Observe()
