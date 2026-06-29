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
        load_unitree_g1_sdk()

        from ._bootstrap import bootstrap, is_bootstrapped
        if not is_bootstrapped():
            bootstrap(wait_first_frame=False)

        from .audio_player import G1StreamPlayer
        logger = con.force_fetch(LoggerItf)
        return G1StreamPlayer(
            logger=logger,
        )


