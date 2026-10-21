"""clear_speech 必须在**话还在播的时候**把它清掉, 又**不许碰到排在它后面的那句**.

它清的是正在播的那一段流, 而它自己也是同一条通道上的一条命令. 两条路都不通:
- 不带 ``call_soon``: 它排在要清的那段话后面, 等轮到自己, 话已经念完 —— 空转;
- 带 ``call_soon`` (blocking): 入队即按"同轨低优先级"清场, 把同一轮里**排在它后面**的
  ``say`` 一起掀掉 —— D19 夜实机就是这么把下一句吞掉的.

所以清理放在 prepare / on_compiled 阶段: 那时命令还没进队列, 清得到的只有已经存在的流.

这里钉的是阶段契约 (清在 prepare、执行体不再清), 不是 TTS 后端能否真的截断音频.
"""

import pytest

from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.speech.speech_module import SpeechChannelModule


@pytest.mark.asyncio
async def test_clear_speech_drops_in_prepare_and_spares_the_next_line():
    """prepare 阶段清掉正在播的流; 执行体不再清 —— 否则后一句 say 陪葬."""
    speech = MockSpeech()
    module = SpeechChannelModule(speech=speech)
    await module.on_startup()
    command = module.own_commands()["clear_speech"]

    playing = speech.new_segment()
    await command.partial()
    assert playing.is_closed(), "prepare 阶段就该把正在播的流清掉, 否则命令永远排在它后面空转"

    # 同一轮里排在 clear_speech 后面的一句: prepare 之后才被建出来.
    queued_behind = speech.new_segment()
    await command()
    assert not queued_behind.is_closed(), "执行体不许再清一次, 否则排在后面的 say 被一并掀掉"
