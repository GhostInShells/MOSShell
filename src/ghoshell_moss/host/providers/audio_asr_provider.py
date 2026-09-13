from ghoshell_moss.contracts.asr import ASR
from ghoshell_moss.contracts.configs import get_or_create_conf
from ghoshell_moss.contracts.logger import LoggerItf
from ghoshell_container import IoCContainer, Provider
from typing import Type

__all__ = ["AudioASRProvider"]


class AudioASRProvider(Provider[ASR]):

    def singleton(self) -> bool:
        # 非单例: ASR 实例上有每消费者的可变面 (热词/上下文 corpus、error 回调、
        # 关闭状态), 单例会把它们跨消费者串在一起. 耳朵按"一个使用者一个实例"装配.
        return False

    def contract(self) -> Type[ASR]:
        return ASR

    def factory(self, con: IoCContainer) -> ASR:
        from ghoshell_moss.host.listener.volcengine_sauc import (
            VolcengineSaucASR,
            VolcengineSaucConfig,
        )

        logger = con.force_fetch(LoggerItf)
        config = get_or_create_conf(con, VolcengineSaucConfig())
        return VolcengineSaucASR(config=config, logger=logger)
