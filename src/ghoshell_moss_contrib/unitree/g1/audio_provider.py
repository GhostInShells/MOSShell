from ghoshell_container import IoCContainer, Provider
from ghoshell_moss.contracts.speech import StreamAudioPlayer
from ghoshell_common.contracts import LoggerItf

__all__ = ['G1StreamPlayerProvider']


class G1StreamPlayerProvider(Provider[StreamAudioPlayer]):
    """在 IoC 容器中注册 G1StreamPlayer 为 StreamAudioPlayer 的实现。"""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> StreamAudioPlayer:
        from ._sdk import load_unitree_g1_sdk
        # 如果没有环境路径, 启动时抛出异常.
        load_unitree_g1_sdk()

        from .audio_player import G1StreamPlayer
        logger = con.force_fetch(LoggerItf)
        return G1StreamPlayer(
            logger=logger,
        )


