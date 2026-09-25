import asyncio
import json
from typing import Callable

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.contracts.speech import (
    PlaybackSample,
    Speech,
    SpeechStream,
    TTSSpeech,
    speech_tail,
    split_speech_tokens,
)


# 返回值约定: 正常结束且有真实播放时返回描述播放秒数的字符串; 无播放样本返回 None;
# 被中断时 raise STOPPED(301) 携带进度.
# 中断进度里的文本取自 stream 的 clause 对齐 (SpeechStream.played_text, 只有真的播出声的
# clause 才算); 无对齐能力的后端由 backend 在尾帧附文本 (contracts.speech.speech_tail),
# 这里回落 samples[-1].text. 中英混排按 token 切分.
def _played_seconds(samples: list[PlaybackSample]) -> float:
    return sum((s.duration for s in samples), 0.0)


def played_message(samples: list[PlaybackSample]) -> str | None:
    # 没有任何真实播放样本 (MockSpeech / 尚未出声) 时返回 None, 不报误导性的 0.0s.
    if not samples:
        return None
    return f"played {_played_seconds(samples):.1f}s"


def stopped_message(samples: list[PlaybackSample], played_text: str = "") -> str:
    """中断时的进度: 真实播出的秒数 + 已播出文本的尾部与词数.

    :param played_text: stream 的对齐结果 (SpeechStream.played_text); 拿不到对齐的后端
        传空串, 这里回落 PlaybackSample.text 的尾帧提示. 两者都没有时只报秒数 — 已出声
        就不断言 "没有出声".
    """
    seconds = _played_seconds(samples)
    if not samples:
        return f"played {seconds:.1f}s, stopped before audible output"
    if not played_text:
        played_text = samples[-1].text.strip()
    if not played_text:
        return f"played {seconds:.1f}s"
    words = split_speech_tokens(played_text)
    return f"played {seconds:.1f}s, {len(words)} words, stopped at ...{speech_tail(played_text)}"


def build_content_command(speech: Speech, name: str = "__content__") -> Command:
    """Build a speech command from a Speech instance (defaults to `__content__`)."""
    return _SpeechCommandFactory(speech).build_content_command(name=name)


class _SpeechCommandFactory:
    """Factory that builds Command objects from a Speech instance.

    Moved the command-building logic from the contracts layer to core/speech.
    """

    def __init__(self, speech: Speech | TTSSpeech, is_muted: Callable[[], bool] | None = None):
        self._speech = speech
        self._is_muted = is_muted or (lambda: False)

    def _check_muted(self) -> None:
        """静音闸: mute 时任何说出口的意图都拒绝 — 不建 batch、不出声, 让模型当场意识到. """
        if self._is_muted():
            raise CommandErrorCode.NOT_AVAILABLE.error(
                "muted: speech is off — nothing will be spoken until mute(on=false)"
            )

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
            self._check_muted()
            if not speech.is_running():
                return [], {}
            stream = speech.new_segment()
            await stream.start_synthesis()
            _ = asyncio.create_task(_feed_stream(stream, chunks__))
            return [], {"chunks__": stream}

        async def __content__(chunks__) -> str | None:
            """Speak the chunks with your voice. The content becomes spoken audio —
            avoid visually-oriented text (tables, special symbols, markdown) as speech content.
            CDATA chunks if you want to speak xml.
            Returns a short description of seconds played; on interruption raises STOPPED(301) with progress."""
            self._check_muted()
            if not speech.is_running():
                return None
            if not isinstance(chunks__, SpeechStream):
                return None
            samples: list[PlaybackSample] = []
            try:
                await chunks__.play(samples)
            except asyncio.CancelledError:
                # stream 已随 play 的上下文退出关闭, 但 batch 的 clause 还留着 — 对齐结果仍可读.
                CommandUtil.reraise_stopped(stopped_message(samples, chunks__.played_text()))
            finally:
                # 命令退出/中断都要终结 stream — 否则 feed task 继续喂文本、TTS 继续合成,
                # 无界 _chunks 堆音频导致内存泄漏. close() 幂等.
                await chunks__.close()
            return played_message(samples) or chunks__.played_text() or None

        return PyCommand(func=__content__, partial=_content_partial, name=name, blocking=True)

    def build_say_command(self) -> Command:
        tts_speech: TTSSpeech = self._speech
        tts = tts_speech.tts()
        tts_info = tts.get_info()
        voice_schema_str = json.dumps(tts_info.voice_schema, ensure_ascii=False, indent=0)
        tone_descriptions_str = ";".join(
            f"`{tone}`: {description}" for tone, description in tts_info.tones.items()
        )
        # 命令表面只写契约: schema / tone 目录 / 参数语义, 都是启动期就定的静态内容.
        # 此刻的 voice / tone 是状态, 走 module 的 named notice (见 SpeechChannelModule),
        # 否则文档一变整块命令接口就要重发, 命令 meta 也跟着反复重生成.
        say_doc = (
            f"Speak with the specified voice state. The content becomes spoken audio — avoid visually-oriented text (tables, special symbols) as speech content.\n"
            f":param voice: Speed, pitch, etc. of the voice. JSON structure, schema is {voice_schema_str}\n"
            f"  When calling via CTML, voice must be a JSON string, e.g. voice:dict=\"{{'speed': 1.0, 'pitch': 'high'}}\"\n"
            f":param as_default: Make the voice state set in this turn the default.\n"
            f":param chunks__: The text content you speak.\n"
            f":param tone: Switch the voice tone to use. Defaults to the current tone.\n"
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
            self._check_muted()
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
            self._check_muted()
            if not isinstance(chunks__, SpeechStream):
                raise ValueError(f"System error: Chunks is not prepared")
            samples: list[PlaybackSample] = []
            try:
                await chunks__.play(samples)
            except asyncio.CancelledError:
                CommandUtil.reraise_stopped(stopped_message(samples, chunks__.played_text()))
            finally:
                # 命令退出/中断都要终结 stream (见 __content__ 同理). close() 幂等.
                await chunks__.close()
            return played_message(samples)

        return PyCommand(
            say,
            doc=say_doc,
            partial=say_partial,
        )


class SpeechChannelModule(ChannelModule):
    """TTS speech capability module.

    Speech 来源二选一: 构造时显式传入 ``speech`` 实例, 或 startup 时从 IoC container
    递归取. 两者都拿不到 (或拿到但未 ``is_running``) 时 ``is_available()`` 为 False,
    模块整体从表面下架 — 不挂 say/mute, 也不报 mute notice.
    """

    def __init__(self, *, register_content_command: bool = False, speech: Speech | None = None):
        self._speech: Speech | None = speech
        self._own_commands = {}
        self._register_content_command = register_content_command
        self._muted = False

    def name(self) -> str:
        return "speech"

    def own_commands(self) -> dict[str, Command]:
        return self._own_commands

    def is_available(self) -> bool:
        """Speech 已 resolve 且正在运行.

        命令 / notice / meta 以此为准 — 判定只写在这里, 由 on_startup 与
        get_named_notices 调用, 不各自重抄条件. 语音中途停掉时模块自动下架,
        恢复则自动回来.
        """
        return self._speech is not None and self._speech.is_running()

    async def get_named_notices(self) -> dict[str, str | None]:
        """此刻的状态 — 命令表面写契约, 状态由这里随 meta 刷新下发.

        mute 恒非空 (off/on 都非空), 这样 off→on / on→off 都能被 delta 宣告; 空串表示
        "不变", 模型保留上一次读到的内容. voice/tone 只在 TTS 可用时出现, 缺席即 removed,
        模型收到 ``<voice removed/>`` 墓碑, 不会残留上一次的音色. 两者只报状态, 不重复
        schema / tone 目录 (那些在命令文档里).
        """
        if not self.is_available():
            return {}
        result = {
            "mute": (
                "on — speech is off: say/content will be refused until mute(on=false)"
                if self._muted
                else "off"
            ),
        }
        if isinstance(self._speech, TTSSpeech):
            tts = self._speech.tts()
            result["voice"] = f"Current voice: {json.dumps(tts.get_voice(), ensure_ascii=False)}"
            result["tone"] = f"Current tone: `{tts.current_tone()}`"
        return result

    async def _mute(self, on: bool = True) -> str:
        """Mute or unmute your own voice. While muted, `say` and content commands refuse to
        speak (they raise an error) — you can still think and act, just not make sound.
        Use it to stay quiet until asked (e.g. a meeting or demo where the human introduces you).
        """
        self._muted = on
        return "muted" if on else "unmuted"

    async def on_startup(self) -> None:
        if self._speech is None and CommandUtil.enabled():
            self._speech = CommandUtil.get_contract(Speech)
        if not self.is_available():
            self._own_commands = {}
            return
        factory = _SpeechCommandFactory(self._speech, is_muted=lambda: self._muted)
        commands = {}
        if isinstance(self._speech, TTSSpeech):
            cmd = factory.build_say_command()
        else:
            cmd = factory.build_content_command(name="say")
        commands[cmd.name()] = cmd
        if self._register_content_command:
            cmd = factory.build_content_command()
            commands[cmd.name()] = cmd
        commands["mute"] = PyCommand(self._mute, name="mute")
        self._own_commands = commands

    async def on_close(self) -> None:
        if self._speech:
            await self._speech.clear()
        self._speech = None
