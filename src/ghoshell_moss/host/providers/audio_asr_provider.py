from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_container import IoCContainer, Provider
from typing import Type

__all__ = ["AudioASRProvider"]


class AudioASRProvider(Provider[ASR]):

    def singleton(self) -> bool:
        return True

    def contract(self) -> Type[ASR]:
        return ASR

    def factory(self, con: IoCContainer) -> ASR:
        from ghoshell_moss.host.listener.volcengine_asr import VolcengineASR, VolcengineASRConfig

        logger = con.force_fetch(LoggerItf)
        config = VolcengineASRConfig().resolve_env()
        return VolcengineASR(config=config, logger=logger)
