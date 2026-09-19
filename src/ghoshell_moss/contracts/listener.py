from abc import ABC, abstractmethod
from typing import Awaitable, Callable
from typing_extensions import Self

from .asr import ASR, RecognitionEvent, RecognitionSegment
from .audio import AudioChunk

Discard = Callable[[], None]


class Listener(ABC):
    """缝合 capture + asr 的"耳朵"器官 — 对称 Speech (嘴).

    持有一个 capture source (设备) 和一个 asr (注入), 拥有两者的生命周期.
    ``listen()`` 产出一条 listening session (ListenerState), 同一时刻至多一条 —
    再次 listen cancel 前一条.
    """

    @property
    @abstractmethod
    def state(self) -> 'ListenerState | None':
        """当前活跃 session, 无则 None."""

    @abstractmethod
    async def listen(self) -> 'ListenerState':
        """开一条 listening session (未启动), cancel 前一条."""

    @abstractmethod
    async def close(self) -> None:
        """关闭 Listener: 关 asr + capture, 结束活跃 session."""

    def is_listening(self) -> bool:
        state = self.state
        return state is not None and state.is_running()

    @abstractmethod
    def is_running(self) -> bool:
        """器官是否已启动 (entered 且未 close) — 对称 ``Speech.is_running``.

        宿主据此决定托管归属: 已在运行 → 由启动方持有, 后来者只借用, 不重复
        enter 也不代它退出; 未运行 → 进入者即持有者.
        与 ``is_listening()`` 不同: 后者指"当前是否有一条活跃 session".
        """

    # 三个 on_*: 自动装线到当前 session (跨 session 稳定订阅)
    @abstractmethod
    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


class ASRListener(Listener, ABC):
    """支持 ASR 的 listener — 暴露内部 ASR 供控制层调参/自解释. 对称 TTSSpeech.

    控制层 (ListenerController) 需要访问 listener 内部那份 ASR (非单例, 实例级
    params) 才能让 configure_asr/VAD 作用到实际识别流; 本接口把这份 ASR 暴露出来,
    避免控制层经 IoC 二次 fetch 拿到不同实例.
    """

    @abstractmethod
    def asr(self) -> ASR:
        """返回内部持有的 ASR 实例 (与识别流同源)."""


class ListenerState(ABC):
    """一条 listening session 的独立生命周期 — 对称 SpeechStream.

    ``async with state`` 启动后台 pump (订阅 capture consumer + asr.recognize),
    ``__aexit__`` 停 pump 并结束 recognition. 三个 on_* 观察本条 session.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @abstractmethod
    def commit(self) -> None:
        """透传 recognition.commit() — 发负序号讯号, 切一个 segment (不关流)."""

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        ...

    @abstractmethod
    def on_event_creating(
            self,
            callback: Callable[[RecognitionEvent], Awaitable[None] | None],
    ) -> None:
        """透传 recognition.on_event_creating — 判停逻辑挂载点.

        awaitable 回调 inline await (阻塞消费点), sync 回调 to_thread 卸载. 判停逻辑
        是 per-session 的 (每次 listen 重新挂载), 不返回 Discard.
        """


class ListenLifecycle(ABC):
    """听侧治理的生命周期接线表面 — moss runtime 据此 enter/exit 听侧.

    ListenerController 的完整表面 (判停/信号/自解释) 太特殊、还在演化, 不上 IoC.
    这里只承诺生命周期 + 急停: enter = 启动听器官 (capture+asr), exit = 关闭,
    pause = 急停/恢复 (对称 shell/mindflow 的 pause 级联).
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...

    @abstractmethod
    def pause(self, toggle: bool = True) -> None:
        """急停/恢复: True 停听 (stop active session), False 恢复默认礼仪."""
        ...
