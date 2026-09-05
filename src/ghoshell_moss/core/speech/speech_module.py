import asyncio
import json

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand
from ghoshell_moss.contracts.speech import PlaybackSample, Speech, SpeechStream, TTSSpeech, split_speech_tokens
from ghoshell_moss.core.speech.null import NullSpeech


# 返回值约定: 正常结束且有真实播放时返回描述播放秒数的字符串; 无播放样本返回 None;
# 被中断时 raise STOPPED(301) 携带进度.
# 尾帧文本由 backend (volcengine/mimo) 在拿不到 text-音频对齐时附到最后一个 PlaybackSample.text
# (见 contracts.speech.speech_tail), 这里直接消费 samples[-1].text; 中英混排按 token 切分.
def _played_seconds(samples: list[PlaybackSample]) -> float:
    return sum((s.duration for s in samples), 0.0)


def played_message(samples: list[PlaybackSample]) -> str | None:
    # 没有任何真实播放样本 (MockSpeech / 尚未出声) 时返回 None, 不报误导性的 0.0s.
    if not samples:
        return None
    return f"played {_played_seconds(samples):.1f}s"


def stopped_message(samples: list[PlaybackSample]) -> str:
    seconds = _played_seconds(samples)
    tail = samples[-1].text.strip() if samples else ""
    if tail:
        words = split_speech_tokens(tail)
        return f"played {seconds:.1f}s, {len(words)} words, stopped at ...{tail}"
    return f"played {seconds:.1f}s, stopped before audible output"


def build_content_command(speech: Speech, name: str = "__content__") -> Command:
    """Build a speech command from a Speech instance (defaults to `__content__`)."""
    return _SpeechCommandFactory(speech).build_content_command(name=name)


class _SpeechCommandFactory:
    """Factory that builds Command objects from a Speech instance.

    Moved the command-building logic from the contracts layer to core/speech.
    """

    def __init__(self, speech: Speech | TTSSpeech):
        self._speech = speech

    def build_content_command(self, name: str = "__content__") -> Command:
        speech = self._speech

        async def _feed_stream(stream: SpeechStream, deltas):
            try:
                if not speech.is_running():
                    return
                has_first_chunk = False
                async for chunk in deltas:
                    if not has_first_chunk and chunk.strip():
                        has_first_chunk = True
                        await stream.start_synthesis()
                    stream.feed(chunk)
                stream.commit()
            except asyncio.CancelledError:
                await stream.close()

        async def _content_partial(chunks__):
            if not speech.is_running():
                return [], {}
            stream = speech.new_stream()
            await stream.start_synthesis()
            _ = asyncio.create_task(_feed_stream(stream, chunks__))
            return [], {"chunks__": stream}

        async def __content__(chunks__) -> str | None:
            """Speak the chunks with your voice. The content becomes spoken audio —
            avoid visually-oriented text (tables, special symbols, markdown) as speech content.
            CDATA chunks if you want to speak xml.
            Returns a short description of seconds played; on interruption raises STOPPED(301) with progress."""
            if not speech.is_running():
                return None
            if not isinstance(chunks__, SpeechStream):
                return None
            samples: list[PlaybackSample] = []
            try:
                await chunks__.play(samples)
            except asyncio.CancelledError:
                CommandUtil.reraise_stopped(stopped_message(samples))
            return played_message(samples)

        return PyCommand(func=__content__, partial=_content_partial, name=name, blocking=True)

    def build_say_command(self) -> Command:
        tts_speech: TTSSpeech = self._speech
        tts = tts_speech.tts()
        tts_info = tts.get_info()
        voice_schema_str = json.dumps(tts_info.voice_schema, ensure_ascii=False, indent=0)

        def say_doc() -> str:
            current_voice = tts.get_voice()
            current_tone = tts.current_tone()
            tones = tts_info.tones
            tone_descriptions = []
            for _tone, description in tones.items():
                tone_descriptions.append(f"`{_tone}`: {description}")
            tone_descriptions_str = ";".join(tone_descriptions)

            return (
                f"Speak with the specified voice state. The content becomes spoken audio — avoid visually-oriented text (tables, special symbols) as speech content.\n"
                f":param voice: Speed, pitch, etc. of the voice. JSON structure, schema is {voice_schema_str}\n"
                f"  Your current voice state is: {json.dumps(current_voice, ensure_ascii=False)}.\n"
                f"  When calling via CTML, voice must be a JSON string, e.g. voice:dict=\"{{'speed': 1.0, 'pitch': 'high'}}\"\n"
                f":param as_default: Make the voice state set in this turn the default.\n"
                f":param chunks__: The text content you speak.\n"
                f":param tone: Switch the voice tone to use. Defaults to the current tone.\n"
                f"  Current tone is `{current_tone}`."
                f"  Available tones: {tone_descriptions_str}\n"
                f"\n"
                f":return: a short description of seconds actually played. On interruption raises a STOPPED error carrying progress.\n"
            )

        async def say_partial(
                chunks__,
                voice: dict | None = None,
                as_default: bool = False,
                tone: str = "",
        ) -> tuple[list, dict]:
            if as_default:
                if voice:
                    tts.set_voice(voice)
                if tone:
                    tts.use_tone(tone)
            batch = tts.new_batch(voice=voice, tone=tone)
            stream = tts_speech.new_tts_stream(batch)

            async def run_tts_batch() -> None:
                try:
                    nonlocal chunks__
                    await stream.start_synthesis()
                    async for chunk in chunks__:
                        if stream.is_closed():
                            return
                        stream.feed(chunk)
                except Exception as e:
                    await stream.fail(e)
                finally:
                    stream.commit()

            _ = asyncio.create_task(run_tts_batch())
            return [], dict(voice=voice, chunks__=stream, as_default=as_default)

        async def say(chunks__, voice: dict | None = None, as_default: bool = False, tone: str = "") -> str | None:
            if not isinstance(chunks__, SpeechStream):
                raise ValueError(f"System error: Chunks is not prepared")
            samples: list[PlaybackSample] = []
            try:
                await chunks__.play(samples)
            except asyncio.CancelledError:
                CommandUtil.reraise_stopped(stopped_message(samples))
            return played_message(samples)

        return PyCommand(
            say,
            doc=say_doc,
            partial=say_partial,
        )


class SpeechChannelModule(ChannelModule):
    """TTS speech capability module.

    The Speech instance is registered to the IoC container externally.
    Fetched via CommandUtil on startup.
    """

    def __init__(self, *, register_content_command: bool = False):
        self._speech: Speech | None = None
        self._own_commands = {}
        self._register_content_command = register_content_command

    def name(self) -> str:
        return "speech"

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands

    async def on_startup(self) -> None:
        if CommandUtil.enabled():
            self._speech = CommandUtil.get_contract(Speech)
        self._speech = self._speech or NullSpeech()
        factory = _SpeechCommandFactory(self._speech)
        commands = {}
        if isinstance(self._speech, TTSSpeech):
            cmd = factory.build_say_command()
        else:
            cmd = factory.build_content_command(name="say")
        commands[cmd.name()] = cmd
        if self._register_content_command:
            cmd = factory.build_content_command()
            commands[cmd.name()] = cmd
        self._own_commands = commands

    async def on_close(self) -> None:
        self._speech = None
