from typing import Optional

from ghoshell_moss.message import unique_id

from ghoshell_moss.contracts.speech import Speech, SpeechStream

__all__ = ["NullSpeech"]


class _NullSpeechStream(SpeechStream):
    """丢弃所有文本，不缓冲，不播放，零开销。"""

    def __init__(self, id: str = ""):
        super().__init__(id=id or unique_id())

    def _buffer(self, text: str) -> None:
        pass

    def _commit(self) -> None:
        pass

    async def fail(self, err: Exception) -> None:
        pass

    async def start_synthesis(self) -> None:
        pass

    async def start_play(self) -> None:
        pass

    async def wait_played(self) -> None:
        pass

    def is_closed(self) -> bool:
        return False

    def buffered(self) -> str:
        return ""

    async def close(self):
        pass

    def close_sync(self):
        pass


class NullSpeech(Speech):
    """默认空 speech，丢弃所有输出。

    CTMLShell 在没有配置 speech 时的兜底实现。
    零内存累积，不创建线程，所有操作即时返回。
    """

    def new_segment(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        return _NullSpeechStream(id=batch_id or unique_id())

    def is_running(self) -> bool:
        return True

    async def clear(self) -> list[str]:
        return []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass
