import numpy as np
import pytest

from ghoshell_moss.contracts.audio import AudioChunk, AudioFrameMeta
from ghoshell_moss.host.speech.capture.miniaudio_capture import (
    _compute_frame_meta,
    pack_chunk,
    unpack_chunk,
)


def _make_sine(duration: float, sample_rate: int, freq: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    """Generate int16 sine wave, shape (n_samples, 1)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * (32767 * amplitude)).astype(np.int16)
    return wave.reshape(-1, 1)


# ── _compute_frame_meta ────────────────────────────────────────────


class TestComputeFrameMeta:
    def test_sine_dbfs_range(self):
        """Full-scale sine: RMS ≈ -3 dBFS (amplitude 1.0 → peak at 0 dBFS)."""
        samples = _make_sine(0.05, 44100, amplitude=1.0)  # full scale
        meta = _compute_frame_meta(samples)
        assert -6 < meta.rms_db <= 0
        assert not meta.is_silent

    def test_silence_is_detected(self):
        """Zero samples → rms_db near -inf, is_silent=True."""
        samples = np.zeros((2205, 1), dtype=np.int16)
        meta = _compute_frame_meta(samples)
        assert meta.rms_db < -50
        assert meta.is_silent

    def test_half_amplitude_lower_dbfs(self):
        """Half amplitude → ~6 dB quieter than full scale."""
        full = _make_sine(0.05, 44100, amplitude=1.0)
        half = _make_sine(0.05, 44100, amplitude=0.5)
        meta_full = _compute_frame_meta(full)
        meta_half = _compute_frame_meta(half)
        assert meta_half.rms_db < meta_full.rms_db

    def test_bands_have_expected_keys(self):
        """Meta bands always contain bass, mid, high."""
        samples = _make_sine(0.05, 44100)
        meta = _compute_frame_meta(samples)
        assert set(meta.bands.keys()) == {"bass", "mid", "high"}
        for v in meta.bands.values():
            assert isinstance(v, float)

    def test_short_input_uses_rms_for_bands(self):
        """When FFT has <6 bins, bands fall back to rms_db."""
        samples = np.array([[100], [200], [300]], dtype=np.int16)  # 3 samples → FFT 2 bins
        meta = _compute_frame_meta(samples)
        assert meta.bands["bass"] == meta.bands["mid"] == meta.bands["high"]

    def test_mono_2d_shape(self):
        """(n, 1) shape input works correctly."""
        samples = _make_sine(0.05, 44100)
        assert samples.ndim == 2 and samples.shape[1] == 1
        meta = _compute_frame_meta(samples)  # should not raise
        assert isinstance(meta, AudioFrameMeta)


# ── pack_chunk / unpack_chunk ───────────────────────────────────────


class TestPackUnpack:
    def test_roundtrip(self):
        """pack → unpack preserves all fields."""
        samples = _make_sine(0.05, 44100).flatten()
        meta = AudioFrameMeta(rms_db=-12.3, bands={"bass": -20, "mid": -12, "high": -8}, is_silent=False)
        chunk = AudioChunk(seq=42, timestamp=1234567890.123, samples=samples, meta=meta)

        packed = pack_chunk(chunk)
        assert isinstance(packed, bytes)

        unpacked = unpack_chunk(packed)
        assert unpacked.seq == 42
        assert unpacked.timestamp == pytest.approx(1234567890.123)
        assert unpacked.meta.rms_db == -12.3
        assert unpacked.meta.bands == {"bass": -20, "mid": -12, "high": -8}
        assert unpacked.meta.is_silent is False
        assert np.array_equal(unpacked.samples, samples)

    def test_binary_header_length(self):
        """First 4 bytes are uint32 BE meta_json length."""
        samples = _make_sine(0.05, 44100).flatten()
        chunk = AudioChunk(seq=0, timestamp=0.0, samples=samples, meta=AudioFrameMeta())
        packed = pack_chunk(chunk)

        import struct
        meta_len = struct.unpack(">I", packed[:4])[0]
        assert meta_len > 0
        assert meta_len < 1024  # meta JSON is small

    def test_silent_chunk_roundtrip(self):
        """Silent meta flag survives roundtrip."""
        samples = np.zeros(100, dtype=np.int16)
        meta = AudioFrameMeta(rms_db=-96, bands={"bass": -96, "mid": -96, "high": -96}, is_silent=True)
        chunk = AudioChunk(seq=0, timestamp=0.0, samples=samples, meta=meta)

        packed = pack_chunk(chunk)
        unpacked = unpack_chunk(packed)
        assert unpacked.meta.is_silent is True
        assert unpacked.meta.rms_db == -96

    def test_pcm_bytes_match_int16_size(self):
        """PCM payload size = n_samples * 2 (int16)."""
        samples = _make_sine(0.05, 44100).flatten()
        chunk = AudioChunk(seq=0, timestamp=0.0, samples=samples, meta=AudioFrameMeta())

        packed = pack_chunk(chunk)

        import struct
        meta_len = struct.unpack(">I", packed[:4])[0]
        pcm_bytes = packed[4 + meta_len:]
        assert len(pcm_bytes) == len(samples) * 2

    def test_high_frequency_energy_in_high_band(self):
        """8 kHz tone → high band should dominate over bass."""
        samples = _make_sine(0.05, 44100, freq=8000.0, amplitude=0.5)
        meta = _compute_frame_meta(samples)
        # High-frequency tone → high band energy > bass band energy
        assert meta.bands["high"] > meta.bands["bass"]

    def test_low_frequency_energy_in_bass_band(self):
        """100 Hz tone → bass band should dominate over high."""
        samples = _make_sine(0.05, 44100, freq=100.0, amplitude=0.5)
        meta = _compute_frame_meta(samples)
        assert meta.bands["bass"] > meta.bands["high"]
