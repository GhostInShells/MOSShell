import asyncio
import time

import pytest

from ghoshell_moss.types.topics.audio import AudioSampleTopic

from ghoshell_screen_manager.audio import MockAudioSource, TopicAudioSource


def test_idle_sample_is_silent():
    src = MockAudioSource()
    s = src.sample()
    assert s.role == "ghost"
    assert s.rms_db < -45
    assert all(v == -96.0 for v in s.spectrum_bins)
    assert all(v == 0.0 for v in s.waveform)


def test_ghost_sample_is_speech_with_energy():
    src = MockAudioSource(seed=1)
    src.set_mode("ghost")
    s = src.sample()
    assert s.role == "ghost"
    assert s.rms_db > -45
    assert len(s.spectrum_bins) == 16
    assert len(s.waveform) == 128


def test_user_sample_is_listening():
    src = MockAudioSource(seed=2)
    src.set_mode("user")
    s = src.sample()
    assert s.role == "user"
    assert len(s.spectrum_bins) == 16
    assert len(s.waveform) == 128


def test_invalid_mode_rejected():
    src = MockAudioSource()
    with pytest.raises(ValueError):
        src.set_mode("bogus")


# -- the live source: the bus is the switch ---------------------------------


def _speech(role: str = "ghost") -> AudioSampleTopic:
    return AudioSampleTopic(
        role=role,
        rms_db=-20.0,
        peak=0.5,
        spectrum_bins=[-40.0] * 16,
        waveform=[0.1] * 128,
    )


class _FakeSubscriber:
    def __init__(self, samples: list[AudioSampleTopic]) -> None:
        self._samples = samples

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def poll_model(self, timeout: float | None = None) -> AudioSampleTopic:
        if self._samples:
            return self._samples.pop(0)
        await asyncio.sleep(0.01)  # a real subscriber waits; don't spin the loop
        raise asyncio.TimeoutError


class _FakeTopics:
    def __init__(self, samples: list[AudioSampleTopic]) -> None:
        self._samples = samples
        self.subscribed = 0

    def subscribe_model(self, model, **kwargs) -> _FakeSubscriber:
        self.subscribed += 1
        return _FakeSubscriber(self._samples)


def test_topic_source_is_silent_before_any_sample():
    src = TopicAudioSource(_FakeTopics([]))
    assert src.sample().rms_db < -45
    assert not src.running


def test_topic_source_latest_wins_and_stales_to_silence():
    src = TopicAudioSource(_FakeTopics([]), idle_timeout=0.05)
    src._latest = _speech()
    src._latest_at = time.monotonic()
    assert src.sample().rms_db > -45

    # speech stopped: the held sample must not keep painting the rain
    src._latest_at -= 1.0
    assert src.sample().rms_db < -45


def test_topic_source_pumps_the_bus_until_stopped():
    topics = _FakeTopics([_speech("user")])
    src = TopicAudioSource(topics)

    async def scenario() -> str:
        await src.start()
        seen = ""
        for _ in range(100):
            await asyncio.sleep(0.01)
            seen = src.sample().role
            if seen == "user":
                break
        assert src.running
        await src.stop()
        return seen

    seen = asyncio.run(scenario())
    assert seen == "user"
    assert topics.subscribed == 1
    assert not src.running
