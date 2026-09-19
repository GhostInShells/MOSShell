import pytest

from ghoshell_screen_manager.audio import MockAudioSource


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
