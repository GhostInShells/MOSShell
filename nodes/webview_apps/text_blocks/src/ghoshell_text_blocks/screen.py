"""ScreenPush — contract for pushing block updates to the visual surface.

ScreenAddr tells the model where the surface lives (URL).
ScreenPush is the channel-side API for pushing streaming content.

For S1 (standalone Reflex app), the Reflex render loop reads directly
from BlockStore and ScreenPush is a signal-only bridge. In S2 (Matrix
integration), it becomes the WebSocket push layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScreenAddr:
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def free_port(cls) -> ScreenAddr:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return cls(port=s.getsockname()[1])


class ScreenPush(ABC):
    """Push contract — channel writes, surface renders."""

    @abstractmethod
    async def serve(self) -> None:
        """start the web server (called from channel startup)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """stop the web server (called from channel close)."""
        ...

    @abstractmethod
    async def block_start(self, block_id: int) -> None:
        """a new block is being streamed — show placeholder."""
        ...

    @abstractmethod
    async def push_chunk(self, block_id: int, chunk: str) -> None:
        """push a streaming chunk to the surface."""
        ...

    @abstractmethod
    async def block_done(self, block_id: int) -> None:
        """streaming complete — block is sealed."""
        ...

    @abstractmethod
    async def block_held(self, block_id: int) -> None:
        """streaming paused, lock still held."""
        ...

    @abstractmethod
    async def push_block(self, block_id: int, content: str) -> None:
        """push a complete (non-streaming) block to the surface."""
        ...


class NoopScreenPush(ScreenPush):
    """S1 no-op: Reflex reads directly from BlockStore via polling."""

    async def serve(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def block_start(self, block_id: int) -> None:
        pass

    async def push_chunk(self, block_id: int, chunk: str) -> None:
        pass

    async def block_done(self, block_id: int) -> None:
        pass

    async def block_held(self, block_id: int) -> None:
        pass

    async def push_block(self, block_id: int, content: str) -> None:
        pass
