from typing import Iterable, Type

from ghoshell_container import IoCContainer, Provider

from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.audio import AudioCaptureSource
from ghoshell_moss.contracts.listener import ASRListener, Listener
from ghoshell_moss.contracts.logger import LoggerItf

__all__ = ["ListenerProvider"]


class ListenerProvider(Provider[Listener]):
    """listener (耳朵) provider — 单例, 缝合 capture + asr.

    对称 TTSSpeechServiceProvider: capture (单例) 与 asr (非单例) 经 IoC 注入,
    拼成 HostListener. listener 条件持有 capture 生命周期 (is_running 判定),
    始终持有 asr 生命周期.
    """

    def singleton(self) -> bool:
        return True

    def aliases(self) -> Iterable[Type]:
        yield ASRListener

    def factory(self, con: IoCContainer) -> ASRListener:
        from ghoshell_moss.host.listener.listener import HostListener

        capture = con.force_fetch(AudioCaptureSource)
        asr = con.force_fetch(ASR)
        logger = con.force_fetch(LoggerItf)
        return HostListener(capture=capture, asr=asr, logger=logger)
