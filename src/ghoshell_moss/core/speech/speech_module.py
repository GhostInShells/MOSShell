import asyncio
import contextlib
import json
from typing import Callable

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.blueprint.states_channel import ChannelModule
from ghoshell_moss.core.concepts.command import Command, PyCommand
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.message import Message
from ghoshell_moss.contracts.speech import (
    PlaybackSample,
    Speech,
    SpeechClause,
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


# 进度 = 我正在说的那几个字。原料是 clause 的对齐结果 (真的播完一句才来), 取它的尾巴:
# 最后一个字就是播放头现在的位置, 前面加省略号说明它后面还有. 不累积, 不数句子.
_PROGRESS_TAIL = 12


def speech_progress(text: str) -> str:
    """一句已播完的 clause 文本 → 一行进度: ``…最后几个字``. 空串 = 这句没字, 不更新."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    if len(text) <= _PROGRESS_TAIL:
        return text
    return "…" + text[-_PROGRESS_TAIL:]


# 进度心跳间隔 (秒). 每秒一次: 再快没有信息增量, 再慢读不到"我这一句拖了多久".
_PROGRESS_BEAT = 1.0


class _SpeechProgress:
    """播报中的进度: clause 只记"说到哪一截", 心跳每秒把"已播秒数 + 那一截"写成一行.

    两个写入者分工固定 — clause 回调跑在音频 worker 线程, 只写字段; 进度只由心跳写。
    否则两边交替写, 秒数刚出来就被字盖掉。
    """

    def __init__(self, task, samples: list[PlaybackSample]):
        self._task = task
        self._samples = samples
        self._tail = ""
        self._beat: asyncio.Task | None = None
        self._dispose_clause: Callable[[], None] = lambda: None

    def attach(self, chunks: SpeechStream) -> None:
        self._dispose_clause = chunks.on_clause(self._on_clause)
        self._beat = asyncio.create_task(self._run())

    def _on_clause(self, clause: SpeechClause) -> None:
        # 这句真的播完了才来, 所以它的尾字就是播放头现在的位置; 空句不覆盖上一截.
        tail = speech_progress(clause.text)
        if tail:
            self._tail = tail

    def _render(self) -> str:
        if not self._tail:
            return ""
        # 已播 = 音频真正写进设备的样本时长之和; 总时长只有播完才知道, 这里不报.
        return f"{_played_seconds(self._samples):.0f}s {self._tail}"

    async def _run(self) -> None:
        try:
            while not self._task.done():
                await asyncio.sleep(_PROGRESS_BEAT)
                if self._task.done():
                    return
                progress = self._render()
                if progress:
                    self._task.set_progress(progress)
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        self._dispose_clause()
        if self._beat is not None and not self._beat.done():
            self._beat.cancel()


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
            progress = _SpeechProgress(CommandUtil.current_task(), samples)
            progress.attach(chunks__)
            try:
                await chunks__.play(samples.append)
            except asyncio.CancelledError:
                # stream 已随 play 的上下文退出关闭, 但 batch 的 clause 还留着 — 对齐结果仍可读.
                CommandUtil.reraise_stopped(stopped_message(samples, chunks__.played_text()))
            finally:
                await progress.close()
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
            f":param replace: drop whatever is still being said and say this instead — one call,\n"
            f"  not clear_speech followed by say. Default false: the line queues behind the rest.\n"
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
                replace: bool = False,
        ) -> tuple[list, dict]:
            self._check_muted()
            # 「换一句」是一个动作, 不是两个. 清退放在 prepare 阶段 (命令还没进队列),
            # 所以它只掀掉此刻已经存在的流, 不会误伤排在这一句后面的东西.
            if replace:
                await self._clear_speech_partial()
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

        async def say(
                chunks__,
                voice: dict | None = None,
                as_default: bool = False,
                tone: str = "",
                replace: bool = False,
        ) -> str | None:
            self._check_muted()
            if not isinstance(chunks__, SpeechStream):
                raise ValueError(f"System error: Chunks is not prepared")
            samples: list[PlaybackSample] = []
            progress = _SpeechProgress(CommandUtil.current_task(), samples)
            progress.attach(chunks__)
            try:
                await chunks__.play(samples.append)
            except asyncio.CancelledError:
                CommandUtil.reraise_stopped(stopped_message(samples, chunks__.played_text()))
            finally:
                await progress.close()
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
        self._dropped = 0
        """clear_speech 在 prepare 阶段清掉的流数, 由执行体报告一次后归零."""

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

    async def get_context_messages(self) -> list[Message]:
        """说话的运行时状态 — 每轮读一次当下, 不说话时不出现.

        notice 是增量下发的: 秒数每轮都在变, 走 delta 就等于每轮推一条状态刷屏.
        context message 每轮本来就要重发一份, 进度一直变不吃额外代价 —— 这个场景下
        它比 delta 好用.

        报的是「还有多少没说」, 不是「说到第几个字」: 我需要的判断是「这段话还剩多长,
        值不值得现在停下」。数字全部来自 speech 契约, 这里只负责排版。
        """
        line = self._speech_state_line()
        if not line:
            return []
        return [Message.new(tag="speech").with_content(line)]

    def _speech_state_line(self) -> str:
        """一行说清「现在还有多少话没出口」; 不在说话就是空串, 整条不出现.

        数据只从契约上取: 活着几条流由 ``Speech.streams()`` 回答, 每条还剩多少由
        ``SpeechStream.unplayed()`` 回答. 某个后端说不清时就不报那个数——报个数是
        为了方便我判断, 不是义务, 更不能拿零去冒充。
        """
        speech = self._speech
        if speech is None:
            return ""
        alive = [stream for stream in speech.streams() if not stream.is_closed()]
        if not alive:
            return ""
        lefts = [stream.unplayed() for stream in alive]
        said = f"{len(alive)} stream(s) speaking"
        if all(left is not None for left in lefts):
            ahead = sum(len(left) for left in lefts if left is not None)
            said += f" · {ahead} chars still to say" if ahead else " · nothing left to say"
        return f"speech: {said} — clear_speech drops it, say(replace=True) swaps it"

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

    async def _clear_speech_partial(self) -> tuple[tuple, dict]:
        """prepare 阶段 (on_compiled) 就把语音清掉.

        命令在这一步还没进通道队列, 所以「清掉此刻存在的流」不会误伤同一轮里排在它后面的
        ``say`` —— 那句那时根本还没被编译出来. 曾经用 ``call_soon`` 想解决同一个问题, 结果是
        清空动作按「同轨低优先级」把后一句一起掀了 (D19 夜实测: clear_speech 之后的 say 被吞).

        清理结果记在实例上, 执行体只负责把话说清楚, 不再碰语音.
        """
        self._dropped = 0
        if self._speech is not None:
            self._dropped = len(await self._speech.clear())
        return (), {}

    async def _clear_speech(self) -> str:
        """Stop talking right now — drop every open speech stream, the one being played and the
        ones still queued behind it.

        The command that was speaking ends by itself, because its stream closes under it; nothing
        else on the shell is touched.

        The dropping happens while the command is being prepared, before it joins the channel
        queue — it sees only the stream that already exists, so a line emitted right after this
        command is not swept away with the old speech.

        Use it when you started saying something and then decided you should not have: clear,
        then say the new line.
        """
        if self._speech is None:
            return "no speech"
        dropped, self._dropped = self._dropped, 0
        return f"dropped {dropped} speech stream(s)"

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
        # 清理放在 partial (prepare / on_compiled) 里做: 那时命令还没进队列, 清得到的只有已经
        # 存在的流; 执行体留空, 只回报清了几条. 不能用 call_soon —— 那是「入队即执行」, 清空
        # 是按同轨低优先级扫的, 会把同一轮里排在它后面的 say 一并掀掉.
        commands["clear_speech"] = PyCommand(
            self._clear_speech, name="clear_speech", partial=self._clear_speech_partial
        )
        self._own_commands = commands

    async def on_close(self) -> None:
        if self._speech:
            await self._speech.clear()
        self._speech = None
