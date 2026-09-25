"""VoiceProvider — 交错语音总装 InterleavedVoice 的 IoC provider.

singleton, factory 时刻在 matrix bootstrap 之后 (runtime 主动 container.get(Voice)
触发 factory), 因此 con.force_fetch(Matrix) 拿到已 bootstrap 完成的 Matrix.

speech / listener 依赖走 con.fetch (可为 None): 上游 provider 会返回 NullSpeech /
NullASR / None. InterleavedVoice 内部对 None listener 静默降级 (无 controller /
无 channel), 对 NullSpeech (非 TTSSpeech) 跳过说侧桥.
"""
from typing import Iterable, Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.configs import ConfigStore
from ghoshell_moss.contracts.listener import ASRListener
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.contracts.voice import Voice
from ghoshell_moss.core.blueprint.matrix import Matrix

__all__ = ["VoiceProvider"]


class VoiceProvider(Provider[Voice]):

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        return ()

    def factory(self, con: IoCContainer) -> Voice:
        from ghoshell_moss.host.voice.interleaved import InterleavedVoice
        matrix = con.force_fetch(Matrix)
        logger = con.force_fetch(LoggerItf)
        # speech / listener 可缺席 (provider 层已降级或未注册): InterleavedVoice 承担
        # 对 None 的静默降级. fetch 而非 force_fetch — 抛错会阻塞 runtime 启动.
        try:
            speech = con.fetch(Speech)
        except Exception:
            logger.warning("Voice: speech fetch failed — degraded to no speech")
            speech = None
        try:
            listener = con.fetch(ASRListener)
        except Exception:
            logger.warning("Voice: listener fetch failed — degraded to no listen")
            listener = None
        config_store = con.fetch(ConfigStore)
        return InterleavedVoice(
            speech=speech,
            listener=listener,
            matrix=matrix,
            config_store=config_store,
            logger=logger,
        )
