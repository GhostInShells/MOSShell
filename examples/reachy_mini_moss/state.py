import asyncio


class ReachyMiniState:
    def __init__(self):
        self.waken = asyncio.Event()
        self.tracking = asyncio.Event()