"""Contract tests for AcousticEchoCanceller (pywebrtc impl).

锚定两条契约: (1) 表面自解释采样率与延迟 hint; (2) 无感对齐 —— hint=0 时 delay
estimator 也能自动对齐并抑制回声 (上层不手动对齐).
"""
import numpy as np

from ghoshell_moss.contracts.audio import AcousticEchoCanceller
from ghoshell_moss.host.listener.capture.webrtc_aec import PyWebrtcEchoCanceller

_SR = 16000
_FRAME = int(_SR * 0.01)  # 10ms = 160


def _make_aec(**kw) -> AcousticEchoCanceller:
    params = dict(sample_rate=_SR, stream_delay_ms=0)
    params.update(kw)
    return PyWebrtcEchoCanceller(**params)


class TestSelfDescribing:
    """表面自解释: 采样率与延迟 hint 对上层可读."""

    def test_exposes_rate_and_delay(self):
        aec = _make_aec()
        assert aec.sample_rate == _SR
        assert aec.stream_delay_ms == 0

    def test_exposes_custom_delay_hint(self):
        aec = _make_aec(stream_delay_ms=60)
        assert aec.stream_delay_ms == 60


class TestProcess:
    """单向: push_far 进参考, process 出处理后的 near."""

    def test_process_returns_same_frame_length(self):
        aec = _make_aec()
        aec.push_far(np.zeros(_FRAME, dtype=np.float32))
        out = aec.process(np.zeros(_FRAME, dtype=np.float32))
        assert out.shape == (_FRAME,)
        assert out.dtype == np.float32

    def test_partial_frame_buffers_until_full(self):
        """不足一帧时缓冲, 凑满再产出."""
        aec = _make_aec()
        aec.push_far(np.zeros(_FRAME, dtype=np.float32))
        assert aec.process(np.zeros(_FRAME // 2, dtype=np.float32)).size == 0
        out = aec.process(np.zeros(_FRAME // 2, dtype=np.float32))
        assert out.size == _FRAME


class TestEchoSuppression:
    """无感对齐: hint=0 时 delay estimator 自动对齐, 抑制延迟回声."""

    def test_suppresses_echo_with_zero_hint(self):
        rng = np.random.default_rng(7)
        n = _SR * 2
        far = (rng.standard_normal(n).astype(np.float32) * 0.1)

        # 声学路径: 50ms 延迟 + 衰减.
        delay = int(0.05 * _SR)
        echo = np.zeros(n, dtype=np.float32)
        echo[delay:] = far[: n - delay] * 0.3

        aec = _make_aec()
        out_blocks: list[np.ndarray] = []
        for i in range(0, n - _FRAME, _FRAME):
            aec.push_far(far[i: i + _FRAME])
            out_blocks.append(aec.process(echo[i: i + _FRAME]))
        out = np.concatenate(out_blocks)

        # 后半段 (收敛后) 的回声抑制量.
        half = len(out) // 2

        def _rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(x))))

        before = _rms(echo[half:])
        after = _rms(out[half:])
        erle = 20.0 * np.log10(before / after)
        assert erle > 10.0, f"expected ERLE > 10dB, got {erle:.1f}dB"
