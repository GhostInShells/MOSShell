import json
from typing import Optional

from ghoshell_container import IoCContainer

from ghoshell_moss.core.contracts.speech import Speech, TTSSpeech, TTS, StreamAudioPlayer
from ghoshell_moss.core import PyChannel, Channel, ChannelRuntime, ChannelCtx
from ghoshell_moss.speech import BaseTTSSpeech
from ghoshell_common.helpers import uuid

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
        self._uid = uuid()
        self._name = name
        self._description = description
        self._runtime: Optional[ChannelRuntime] = None

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

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_running():
            raise RuntimeError(f"{self._name} already running")

        channel = PyChannel(name=self._name, description=self._description, blocking=True)

        # 注册说话的命令. 可能被覆盖.
        channel.build.command()(self.say)

        # 注册生命周期.
        channel.build.startup(self._speech.start)
        channel.build.close(self._speech.close)

        if isinstance(self._speech, TTSSpeech):
            # 注册 tts 原生 command
            for command in self._speech.commands():
                channel.build.add_command(command)

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
