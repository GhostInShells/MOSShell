"""SpeechChannelModule 的表面契约: 命令写静态契约, 此刻的 voice/tone 走 named notice.

状态一旦写回命令文档, say 的 interface 就会随每次换音色/音调重发 (命令 meta 也跟着
被标成 dynamic, 每轮重新生成), 提示词缓存因此失效 —— 这几条测试守的就是这条边界.
"""
from typing import Any

import pytest

from ghoshell_moss.contracts.speech import (
    Speech,
    SpeechStream,
    StreamAudioPlayer,
    TTS,
    TTSBatch,
    TTSInfo,
    TTSSpeech,
)
from ghoshell_moss.core.blueprint.states_channel import new_shell_main_channel
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.ctml.shell.ctml_shell import new_ctml_shell
from ghoshell_moss.core.speech import SpeechChannelModule


class _FakeTTS(TTS):
    """只有 say 表面会读到的几项: get_info / get_voice / current_tone."""

    def __init__(self):
        self._voice: dict[str, Any] = {"speed": 1.0}
        self._tone = "gentle"

    def get_info(self) -> TTSInfo:
        return TTSInfo(
            sample_rate=16000,
            voice_schema={"properties": {"speed": {"type": "number"}}},
            tones={"gentle": "calm", "lively": "bright"},
            current_tone=self._tone,
        )

    def get_voice(self) -> dict[str, Any]:
        return dict(self._voice)

    def current_tone(self) -> str:
        return self._tone

    def set_voice(self, config: dict[str, Any]) -> None:
        self._voice = config

    def use_tone(self, config_key: str) -> None:
        self._tone = config_key

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def clear(self) -> None:
        pass

    def new_batch(
            self,
            batch_id: str = "",
            *,
            callback=None,
            tone: str | None = None,
            voice: dict | None = None,
    ) -> TTSBatch:
        raise NotImplementedError


class _FakeTTSSpeech(TTSSpeech):
    """不做真实合成: 测试只关心 say 的表面, 不关心它怎么出声."""

    def __init__(self, tts: TTS):
        self._tts = tts

    def tts(self) -> TTS:
        return self._tts

    def player(self) -> StreamAudioPlayer:
        raise NotImplementedError

    def new_tts_stream(self, batch: TTSBatch) -> SpeechStream:
        raise NotImplementedError

    def new_segment(self, *, batch_id: str | None = None) -> SpeechStream:
        raise NotImplementedError

    def is_running(self) -> bool:
        return True

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    async def clear(self) -> list[str]:
        return []


def _new_shell(tts: TTS):
    """speech 命令由 module 挂载 — 裸 CTML main channel 不带它 (见 new_moss_main_channel)."""
    main = new_shell_main_channel()
    main.with_module(SpeechChannelModule())
    return new_ctml_shell(main_channel=main, speech=_FakeTTSSpeech(tts))


def _say_interface(shell) -> str:
    metas = shell.channel_metas()[""]
    return next(cmd.interface for cmd in metas.commands if cmd.name == "say")


@pytest.mark.asyncio
async def test_voice_and_tone_state_rides_named_notice():
    """换默认音色/音调后, 只有 notice 片段变, 命令表面一个字符都不动."""
    tts = _FakeTTS()
    shell = _new_shell(tts)
    async with shell:
        await shell.refresh_metas()
        notices = shell.channel_metas()[""].named_notices
        assert notices["tone"] == "Current tone: `gentle`"
        assert notices["voice"] == 'Current voice: {"speed": 1.0}'
        interface = _say_interface(shell)

        tts.use_tone("lively")
        tts.set_voice({"speed": 1.2})
        await shell.refresh_metas()

        notices = shell.channel_metas()[""].named_notices
        assert notices["tone"] == "Current tone: `lively`"
        assert notices["voice"] == 'Current voice: {"speed": 1.2}'
        assert _say_interface(shell) == interface
        assert "1.2" not in interface


@pytest.mark.asyncio
async def test_contract_stays_on_surface():
    """schema 与 tone 目录是契约, 留在命令文档里 — 模型不依赖 notice 就能看懂参数."""
    shell = _new_shell(_FakeTTS())
    async with shell:
        await shell.refresh_metas()
        interface = _say_interface(shell)
        assert "`gentle`: calm" in interface
        assert "`lively`: bright" in interface
        assert "speed" in interface


def _new_runtime(tts: TTS):
    main = new_shell_main_channel()
    main.with_module(SpeechChannelModule())
    main.build.with_binding(Speech, _FakeTTSSpeech(tts))
    return main


@pytest.mark.asyncio
async def test_mute_toggles_and_surfaces_in_notice():
    """mute 是 command 唯一的真相写入点, 状态走 notice (off/on 恒非空, 保证 delta 宣告)."""
    main = _new_runtime(_FakeTTS())
    async with main.bootstrap() as runtime:
        assert runtime.self_meta().named_notices["mute"] == "off"

        await runtime.execute_command("mute", kwargs={"on": True})
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["mute"].startswith("on")

        await runtime.execute_command("mute", kwargs={"on": False})
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["mute"] == "off"


@pytest.mark.asyncio
async def test_muted_say_is_refused_with_not_available():
    """mute 时 say 直接拒绝 (403), 不出声也不建 batch — 模型当场意识到自己不该说话."""
    main = _new_runtime(_FakeTTS())
    async with main.bootstrap() as runtime:
        await runtime.execute_command("mute", kwargs={"on": True})
        say = runtime.get_command("say")
        with pytest.raises(CommandError) as exc:
            await say(None)
        assert exc.value.code == CommandErrorCode.NOT_AVAILABLE
        assert "mute(on=false)" in str(exc.value)


@pytest.mark.asyncio
async def test_module_without_speech_wires_nothing():
    """无 speech (不注入、容器无 Speech) → 模块不装线, 不挂 say/mute."""
    main = new_shell_main_channel()
    main.with_module(SpeechChannelModule())
    async with main.bootstrap() as runtime:
        names = {cmd.name for cmd in runtime.self_meta().commands}
        assert "say" not in names
        assert "mute" not in names
