"""语音交互：整合 TTS 与音频播放器 | 交互能力 | beta

Example:
    from ghoshell_moss.channels.speech_channel import SpeechChannel
    chan = SpeechChannel(name='speech', description='语音交互通道', speech=tts_speech)
"""

from typing import Optional

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.speech import Speech, TTSSpeech, TTS, StreamAudioPlayer
from ghoshell_moss.core import PyChannel, Channel, ChannelRuntime, ChannelCtx

from ghoshell_moss.core.speech import BaseTTSSpeech, SpeechChannelModule
from ghoshell_moss.message import unique_id

__all__ = ["SpeechChannel", "TTSSpeechChannel"]


class SpeechChannel(Channel):
    """
    实现音频的独立 Channel.
    可以用来整合任何实现了 Speech interface 的模块.
    """

    def __init__(
            self,
            name: str,
            description: str,
            speech: TTSSpeech | Speech,
    ):
        self._speech = speech
        self._uid = unique_id()
        self._name = name
        self._description = description

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._uid

    def description(self) -> str:
        return self._description

    async def say(self, chunks__) -> None:
        """
        使用语音说话的实现.
        :param chunks__: 会转换为语音的自然语言内容. 注意语音播报中使用 tts 等
        """
        task = ChannelCtx.task()
        batch_id = task.cid if task else None
        stream = self._speech.new_stream(batch_id=batch_id)
        await stream.speak(chunks__)

    def materialize(self, container: IoCContainer) -> "ChannelRuntime":
        channel = PyChannel(name=self._name, description=self._description, blocking=True)

        # 注册说话的命令. 可能被覆盖.
        channel.build.command()(self.say)

        # 注册生命周期.
        channel.build.startup(self._speech.start)
        channel.build.close(self._speech.close)

        channel.with_module(
            SpeechChannelModule(register_content_command=True)
        )

        return channel.bootstrap(container=container)


class TTSSpeechChannel(SpeechChannel):
    """
    语法糖, 基于单独的 TTS 和 player 抽象来实现一个 Channel.
    """

    def __init__(
            self,
            *,
            name: str,
            description: str,
            tts: TTS,
            player: StreamAudioPlayer,
    ):
        speech = BaseTTSSpeech(tts=tts, player=player)
        super().__init__(
            name=name,
            description=description,
            speech=speech,
        )
