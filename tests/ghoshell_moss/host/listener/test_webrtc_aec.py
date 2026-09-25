"""Contract tests for AcousticEchoCanceller (pywebrtc impl).

锚定两条契约: (1) 表面自解释采样率与延迟 hint; (2) 无感对齐 —— hint=0 时 delay
estimator 也能自动对齐并抑制回声 (上层不手动对齐).
"""
import numpy as np

from ghoshell_moss.contracts.audio import AcousticEchoCanceller
from ghoshell_moss.host.audios.aec.webrtc_aec import PyWebrtcEchoCanceller

_SR = 16000
_FRAME = int(_SR * 0.01)  # 10ms = 160


def _make_aec(**kw) -> AcousticEchoCanceller:
    params = {"sample_rate": _SR, "stream_delay_ms": 0}
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


#: 采集侧一帧的时长 (AudioCaptureConfig.frame_duration_ms) —— process() 一次收到的量.
_CAPTURE_FRAME_MS = 50
#: 播放侧一次写入设备缓冲的片段量级 (player.on_play 的突发粒度).
_PLAY_CHUNK_MS = 250


def _synth_far(n: int, rng: np.random.Generator) -> np.ndarray:
    """类语音的 far 参考: 带限噪声 × 音节包络 (频谱够丰富才喂得动自适应滤波器)."""
    voice = np.convolve(rng.standard_normal(n), 0.72 ** np.arange(64), mode="same")
    env = np.ones(n, dtype=np.float64)
    for start in range(0, n, int(0.4 * _SR)):
        env[start: start + int(0.3 * _SR)] = 0.0
    return (voice / float(np.max(np.abs(voice))) * env).astype(np.float32)


def _room_echo(far: np.ndarray, *, delay_ms: float, gain: float) -> np.ndarray:
    """扬声器→空气→麦克风: 纯延迟 + 增益 + 两个早期反射."""
    echo = np.zeros_like(far)
    for offset_ms, scale in ((delay_ms, 1.0), (12.0, 0.5), (23.0, 0.32)):
        d = int(offset_ms * _SR / 1000)
        if d < len(far):
            echo[d:] += far[: len(far) - d] * gain * scale
    return echo


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12))


def _feed_aligned(aec: AcousticEchoCanceller, far: np.ndarray, near: np.ndarray) -> np.ndarray:
    """理想节拍: 每个 10ms 帧严格对齐地喂 far (离线探针当初验证 16dB 的喂法)."""
    out = np.zeros(len(near), dtype=np.float32)
    for i in range(0, len(near) - _FRAME, _FRAME):
        aec.push_far(far[i: i + _FRAME])
        out[i: i + _FRAME] = aec.process(near[i: i + _FRAME])[:_FRAME]
    return out


def _feed_live_shape(aec: AcousticEchoCanceller, far: np.ndarray, near: np.ndarray,
                     lead_ms: float = 0.0) -> np.ndarray:
    """live 装线的节拍: far 以片段突发写入, near 以采集帧到达.

    player 先写片段再等设备消费 —— 所以 far 在时间上会**领先** near 一段; 采集帧
    50ms 又会被 process 切成 5 个 10ms 块。这两个形状合起来才是 AEC 真实面对的输入.

    ``lead_ms``: far 相对 near 的提前量 (播放设备缓冲 + safety_delay)。实现方式是把
    声学路径的延迟整体缩短同样多 —— 等价于 far 早到了 lead_ms, 而回声仍是 lead_ms
    之后才出现。
    """
    capture_frame = int(_SR * _CAPTURE_FRAME_MS / 1000)
    chunk = int(_SR * _PLAY_CHUNK_MS / 1000)
    out = np.zeros(len(near), dtype=np.float32)
    for chunk_start in range(0, len(near) - capture_frame, chunk):
        aec.push_far(far[chunk_start: chunk_start + chunk])
        for frame_start in range(chunk_start, min(chunk_start + chunk, len(near) - capture_frame), capture_frame):
            block = aec.process(near[frame_start: frame_start + capture_frame])
            out[frame_start: frame_start + capture_frame] = block[:capture_frame]
    return out


def _lead_variant(far: np.ndarray, near: np.ndarray, lead_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """把"far 提前 lead_ms"折算成一组等价的 (near, echo) 输入.

    far 提前 = near 整体后移。于是把 near/echo 往后推 lead_ms、far 保持原位再喂,
    得到的对齐关系与真机里 far 早写 lead_ms 完全一致。
    """
    shift = int(_SR * lead_ms / 1000)
    if shift <= 0:
        return far, near
    return far, np.concatenate([np.zeros(shift, dtype=np.float32), near[: len(near) - shift]])


def _erle(residual: np.ndarray, echo: np.ndarray) -> float:
    """收敛后的回声抑制量 (dB): 后半段 (避开收敛期) 的输入电平 - 输出电平."""
    half = len(residual) // 2
    return _rms_db(echo[half:]) - _rms_db(residual[half:])


class TestEchoSuppressionInLiveFraming:
    """契约: 对齐是机制, 不依赖上层喂帧节拍.

    采集帧 50ms + far 片段突发的真实节拍下, 抑制量必须与严格逐 10ms 对齐喂帧同级 ——
    这正是 live 失效而 offline 探针正常的差别所在.
    """

    def test_bursty_far_and_capture_frames_match_aligned_feeding(self):
        rng = np.random.default_rng(11)
        n = _SR * 4
        far = _synth_far(n, rng)
        echo = _room_echo(far, delay_ms=60.0, gain=0.3)

        aligned_erle = _erle(_feed_aligned(_make_aec(), far, echo), echo)
        live_erle = _erle(_feed_live_shape(_make_aec(), far, echo), echo)

        assert live_erle > 6.0, f"live 节拍下几乎没有抑制: {live_erle:.1f}dB"
        assert live_erle > aligned_erle - 1.0, (
            f"live 节拍 {live_erle:.1f}dB 明显差于对齐喂帧 {aligned_erle:.1f}dB —— 对齐又变成了上层节拍的依赖"
        )
