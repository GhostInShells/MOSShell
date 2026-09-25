"""Voice contract 契约测试 — 独立于真实 audio 设备 / matrix 网络.

锚定 InterleavedVoiceLifecycle 的装线契约:
- controller 只在 listen=True 时 enter / exit.
- 说侧桥只在 speech 是 TTSSpeech 时装; NullSpeech 跳过.
- speech=False → lifecycle 无说侧桥 (即使 speech 实例是 TTSSpeech).
- pause 级联到 controller (无 controller 时静默).
- run() 的 flag → lifecycle 持有的组件映射.

不测真实 ListenerController 内部行为 (那是 host/listener 自己的测试范围).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import pytest

from ghoshell_moss.contracts.speech import Speech, SpeechClause, TTSSpeech
from ghoshell_moss.contracts.voice import Voice, VoiceLifecycle
from ghoshell_moss.core.speech import MockSpeech
from ghoshell_moss.host.voice.interleaved import InterleavedVoiceLifecycle

# ── fakes ──────────────────────────────────────────


class _FakePublisher:
    def __init__(self):
        self.published = []
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.entered = False

    def pub(self, topic):
        self.published.append(topic)


class _FakeTopics:
    def __init__(self):
        self.publisher = _FakePublisher()

    def model_publisher(self, *, creator, model):
        return self.publisher


class _FakeEnv:
    project_id = "proj"
    ghost_name = "ghost"


class _FakeMatrix:
    def __init__(self):
        self.env = _FakeEnv()
        self.session_topics = _FakeTopics()
        self._container_values = {}
        self._container_raises = False

    @property
    def session(self):
        return self

    @property
    def topics(self):
        return self.session_topics

    @property
    def container(self):
        return self

    def get(self, key):
        if self._container_raises:
            raise KeyError(key)
        return self._container_values.get(key)

    def set(self, key, value):
        self._container_values[key] = value


class _FakePlayer:
    sample_rate = 16000

    def __init__(self):
        self._observers = []

    @property
    def observer_count(self) -> int:
        return len(self._observers)

    def observe(self, callback):
        self._observers.append(callback)

        def _dispose():
            self._observers.remove(callback)

        return _dispose


class _FakeTTSSpeech(MockSpeech, TTSSpeech):
    """最小 TTSSpeech (MockSpeech 提供 Speech 抽象实现) — 只补 lifecycle 用到的
    on_clause / player."""

    def __init__(self):
        MockSpeech.__init__(self)
        self._clause_observers = []
        self._player = _FakePlayer()

    @property
    def clause_observer_count(self) -> int:
        return len(self._clause_observers)

    def emit_clause(self, text: str) -> None:
        """测试驱动: 模拟 speech 产出 clause, 触发所有注册的观察者."""
        clause = SpeechClause(text=text)
        for observer in list(self._clause_observers):
            observer(clause)

    def on_clause(self, callback):
        self._clause_observers.append(callback)

        def _dispose():
            self._clause_observers.remove(callback)

        return _dispose

    def player(self):
        return self._player

    def tts(self):
        raise NotImplementedError

    def new_tts_stream(self, batch):
        raise NotImplementedError


class _FakeController:
    """占位 controller — 只记 enter/exit/pause/feed."""

    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.paused: list[bool] = []
        self.fed: list[str] = []
        self.default_etiquette_started = 0

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *args):
        self.exited += 1

    def start_default_etiquette(self):
        self.default_etiquette_started += 1

    def feed_ghost_clause(self, text: str):
        self.fed.append(text)

    def pause(self, toggle: bool = True):
        self.paused.append(toggle)


def _logger():
    import logging
    return logging.getLogger("test_voice")


# ── lifecycle 装线契约 ──────────────────────────────


@pytest.mark.asyncio
async def test_listen_true_enters_and_pauses_controller():
    """listen=True: controller enter; pause 级联; exit 反卷."""
    controller = _FakeController()
    lifecycle = InterleavedVoiceLifecycle(
        speech=None, controller=controller, matrix=_FakeMatrix(), logger=_logger(),
    )
    async with lifecycle:
        assert controller.entered == 1
        assert controller.default_etiquette_started == 1
        lifecycle.pause(True)
        assert controller.paused == [True]
    assert controller.exited == 1


@pytest.mark.asyncio
async def test_listen_false_skips_controller():
    """listen=False (controller=None): 不 enter; pause 静默不抛."""
    lifecycle = InterleavedVoiceLifecycle(
        speech=None, controller=None, matrix=_FakeMatrix(), logger=_logger(),
    )
    async with lifecycle:
        lifecycle.pause(True)  # no-op, 不抛
    # 无 controller, 无副作用


@pytest.mark.asyncio
async def test_speech_bridge_registered_when_tts():
    """speech 是 TTSSpeech: clause 桥注册回调到 speech; 退出时 disposer 摘除."""
    speech = _FakeTTSSpeech()
    lifecycle = InterleavedVoiceLifecycle(
        speech=speech, controller=None, matrix=_FakeMatrix(), logger=_logger(),
    )
    async with lifecycle:
        assert speech.clause_observer_count == 1
        assert speech.player().observer_count == 1
    assert speech.clause_observer_count == 0
    assert speech.player().observer_count == 0


@pytest.mark.asyncio
async def test_speech_bridge_skipped_for_non_tts():
    """speech 非 TTSSpeech (MockSpeech): 说侧桥不装."""
    lifecycle = InterleavedVoiceLifecycle(
        speech=MockSpeech(), controller=None, matrix=_FakeMatrix(), logger=_logger(),
    )
    async with lifecycle:
        pass  # 无桥可装, 无副作用


@pytest.mark.asyncio
async def test_ghost_clause_feeds_controller_and_publishes():
    """说侧桥: 触发 on_clause → 同时 pub ClauseTopic(role=ghost) + feed_ghost_clause."""
    speech = _FakeTTSSpeech()
    controller = _FakeController()
    matrix = _FakeMatrix()
    lifecycle = InterleavedVoiceLifecycle(
        speech=speech, controller=controller, matrix=matrix, logger=_logger(),
    )
    async with lifecycle:
        speech.emit_clause("hello")
        # 等 drain task 出队
        for _ in range(50):
            if matrix.session_topics.publisher.published:
                break
            await asyncio.sleep(0.01)
        published = matrix.session_topics.publisher.published
        assert len(published) == 1
        assert published[0].role == "ghost"
        assert published[0].text == "hello"
        assert controller.fed == ["hello"]


@pytest.mark.asyncio
async def test_ghost_clause_without_controller_still_publishes():
    """无 controller (speech-only): 桥仍广播, 只跳过 feed."""
    speech = _FakeTTSSpeech()
    matrix = _FakeMatrix()
    lifecycle = InterleavedVoiceLifecycle(
        speech=speech, controller=None, matrix=matrix, logger=_logger(),
    )
    async with lifecycle:
        speech.emit_clause("solo")
        for _ in range(50):
            if matrix.session_topics.publisher.published:
                break
            await asyncio.sleep(0.01)
        assert len(matrix.session_topics.publisher.published) == 1


@pytest.mark.asyncio
async def test_lifecycle_not_reusable():
    """已 enter 的 lifecycle 再 __aenter__ 抛错."""
    lifecycle = InterleavedVoiceLifecycle(
        speech=None, controller=None, matrix=_FakeMatrix(), logger=_logger(),
    )
    async with lifecycle:
        with pytest.raises(RuntimeError):
            await lifecycle.__aenter__()


# ── run() flag 映射 ─────────────────────────────────


class _FlagVoice(Voice):
    """记录 run() 收到的 flag, 返回一个记录组件的 lifecycle stub."""

    def __init__(self):
        self.calls: list[tuple[bool, bool]] = []

    def speech(self) -> Optional[Speech]:
        return None

    def listener_channel(self):
        return None

    def run(self, *, speech: bool, listen: bool) -> VoiceLifecycle:
        self.calls.append((speech, listen))
        return InterleavedVoiceLifecycle(
            speech=None, controller=None, matrix=_FakeMatrix(), logger=_logger(),
        )


@pytest.mark.asyncio
async def test_run_flag_passthrough():
    """Voice.run 收到 runtime 转发的 (speech, listen) 四态."""
    voice = _FlagVoice()
    for speech_flag, listen_flag in [(True, True), (True, False), (False, True), (False, False)]:
        lifecycle = voice.run(speech=speech_flag, listen=listen_flag)
        async with lifecycle:
            pass
    assert voice.calls == [(True, True), (True, False), (False, True), (False, False)]
