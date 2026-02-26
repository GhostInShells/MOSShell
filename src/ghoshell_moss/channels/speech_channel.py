import json
from typing import Optional

from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.speech import Speech, TTSSpeech, TTS, StreamAudioPlayer
from ghoshell_moss.core import PyChannel, Channel, ChannelRuntime, ChannelCtx
from ghoshell_moss.speech import BaseTTSSpeech
from ghoshell_common.helpers import uuid

__all__ = ["SpeechChannel", "TTSSpeechChannel"]


class SpeechChannel(Channel):
    """
    实现音频的独立 Channel.
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
        async with stream:
            async for chunk in chunks__:
                stream.buffer(chunk)

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_running():
            raise RuntimeError(f"{self._name} already running")

        channel = PyChannel(name=self._name, description=self._description, blocking=True)

        # 注册说话的命令.
        channel.build.command()(self.say)

        # 注册生命周期.
        channel.build.start_up(self._speech.start)
        channel.build.close(self._speech.close)

        if isinstance(self._speech, TTSSpeech):
            tts = self._speech.tts()

            def tone_doc() -> str:
                tts_info = tts.get_info()
                current_tone = tts_info.current_tone
                tones = tts_info.tones
                tone_descriptions = []
                for tone, description in tones.items():
                    tone_descriptions.append(f"  {tone}: {description}")
                descriptions = "\n".join(tone_descriptions)

                docstring = f"可以随时切换你所使用的音色.你的当前音色: {current_tone}可以使用的音色:{descriptions}"
                return docstring

            @channel.build.command(doc=tone_doc)
            async def use_tone(tone: str) -> None:
                tts_info = tts.get_info()
                tones = tts_info.tones
                if tone not in tones:
                    raise ValueError(f"Tone {tone} not in {tones}")
                tts.use_tone(tone)

            def voice_doc() -> str:
                tts_info = tts.get_info()
                schema_str = json.dumps(tts_info.voice_schema)
                return f"可以用来设置你说话的声音.:param json__: schema is {schema_str}"

            @channel.build.command(doc=voice_doc)
            async def set_voice(json__) -> None:
                try:
                    config = json.loads(json__)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON: {json__}")

                tts.set_voice(config)

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
