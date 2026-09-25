__all__ = ["noop"]


async def noop() -> None:
    """
    Yield — a deliberate no-op. End your turn without acting: when a moment needs no reaction (e.g.
    you are listening and don't want to interrupt), emit ``noop`` and stop; the next moment wakes you.
    """
    pass
