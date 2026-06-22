import asyncio
import json

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand
from ghoshell_moss.contracts.speech import Speech, SpeechStream, TTSSpeech
from ghoshell_moss.core.speech.null import NullSpeech
from ghoshell_moss.topics import SpeechTopic


def build_content_command(speech: Speech) -> Command:
    """构建 __content__ 内核命令。Shell 始终拥有此命令，无论 speech 类型。"""
    return _SpeechCommandFactory(speech).build_content_command()


class _SpeechCommandFactory:
    """从 Speech 实例构建 Command 对象的工厂。

    将 command 构建逻辑从 contracts 层移到 core/speech 层。
    """

    def __init__(
            self,
            speech: Speech | TTSSpeech,
            *,
            role: str = '',
            name: str = '',
    ):
        self._speech = speech
        self._role = role
        self._name = name

    def build_content_command(self) -> Command:
        speech = self._speech

        async def _feed_stream(stream: SpeechStream, deltas):
            publisher = CommandUtil.topic_publisher(SpeechTopic)
            async with publisher:
                content = ''
                try:
                    if not speech.is_running():
                        return
                    has_first_chunk = False
                    async for chunk in deltas:
                        if not has_first_chunk and chunk.strip():
                            has_first_chunk = True
                            await stream.start_synthesis()
                        stream.feed(chunk)
                        content += chunk
                        publisher.pub(SpeechTopic(
                            speaker_name=self._name,
                            role=self._role,
                            text=content,
                        ))
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

        async def __content__(chunks__) -> None:
            """speak chunks with your voice"""
            if not speech.is_running():
                return None
            if not isinstance(chunks__, SpeechStream):
                return None
            try:
                await chunks__.start_synthesis()
                await chunks__.start_play()
                await chunks__.wait_played()
            finally:
                await chunks__.close()

        return PyCommand(func=__content__, partial=_content_partial, name="__content__", blocking=True)

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
                f"使用指定的声音状态说话. 当它在 __main__ channel 时, 默认可以省略. \n"
                f":param voice: 声音的速度, 音调等. json 结构, json schema 是 {voice_schema_str}\n "
                f"  你当前的声音状态是: {json.dumps(current_voice, ensure_ascii=False)}.\n"
                f"  使用 CTML 调用时, voice 必须是 JSON 字符串, 例如: voice:dict=\"{{'speed': 1.0, 'pitch': 'high'}}\"\n"
                f":param as_default: 将本轮设置的声音状态变成默认.\n"
                f":param chunks__: 你说话的文本内容. \n"
                f":param tone: 切换使用的音色. 默认为当前音色\n"
                f"  当前的音色是 `{current_tone}`"
                f"  当前可以使用的音色: {tone_descriptions_str}\n"
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
                publisher = CommandUtil.topic_publisher(SpeechTopic)
                async with publisher:
                    try:
                        nonlocal chunks__
                        await stream.start_synthesis()
                        content = ""
                        async for chunk in chunks__:
                            if stream.is_closed():
                                return
                            stream.feed(chunk)
                            content += chunk
                            publisher.pub(SpeechTopic(
                                speaker_name=self._name,
                                role=self._role,
                                text=content,
                            ))
                    except Exception as e:
                        await stream.fail(e)
                    finally:
                        stream.commit()

            _ = asyncio.create_task(run_tts_batch())
            return [], dict(voice=voice, chunks__=stream, as_default=as_default)

        async def say(chunks__, voice: dict | None = None, as_default: bool = False, tone: str = "") -> None:
            if not isinstance(chunks__, SpeechStream):
                raise ValueError(f"System error: Chunks is not prepared")
            await chunks__.say()

        return PyCommand(
            say,
            doc=say_doc,
            partial=say_partial,
        )


class SpeechChannelModule(ChannelModule):
    """TTS 语音能力模块。

    可配置注册 content：
    - register_content: 注册 __content__ 内核命令（默认 False）

    Speech 实例由外部注册到 IoC 容器。on_startup 时通过 CommandUtil 获取。
    """

    def __init__(
            self,
            *,
            register_content: bool = False,
            role: str = 'ghost',
            name: str = '',
    ):
        self._speech: Speech | None = None
        self._own_commands = {}
        self._register_content = register_content
        self._role = role
        self._name = name

    def name(self) -> str:
        return "speech"

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands

    async def on_startup(self) -> None:
        if CommandUtil.enabled():
            self._speech = CommandUtil.get_contract(Speech)
        self._speech = self._speech or NullSpeech()
        factory = _SpeechCommandFactory(
            self._speech,
            role=self._role,
            name=self._name,
        )
        commands = {}
        if self._register_content:
            cmd = factory.build_content_command()
            commands[cmd.name()] = cmd
        if isinstance(self._speech, TTSSpeech):
            cmd = factory.build_say_command()
            commands[cmd.name()] = cmd
        self._own_commands = commands

    async def on_close(self) -> None:
        self._speech = None
