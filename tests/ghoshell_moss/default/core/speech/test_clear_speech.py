"""clear_speech 必须能在**话还在播的时候**把它清掉.

它清的是正在播的那一段流, 而它自己也是同一条通道上的一条命令: 不带 ``call_soon`` 时它排在
要清的那段话后面, 等轮到自己, 话已经念完 —— 命令成了空转 (D19 夜, 实机上就是这么空转的).

这里只钉调度契约 (入队即执行). "真的把一段正在播的流截断" 没有单测: MockSpeech 的流关掉之后
speak 那个 async for 不会停, 它没法表达中断; 那条性质属于 TTS 后端, 由实机验.
"""

import pytest

from ghoshell_moss.core.speech.mock import MockSpeech
from ghoshell_moss.core.speech.speech_module import SpeechChannelModule


@pytest.mark.asyncio
async def test_clear_speech_is_scheduled_to_run_on_enqueue():
    """调度契约: 入队即执行 (call_soon), 语义对齐 interrupt."""
    module = SpeechChannelModule(speech=MockSpeech())
    await module.on_startup()

    meta = module.own_commands()["clear_speech"].meta()

    assert meta.call_soon is True, "clear_speech 必须在说话途中插队, 否则它永远是空转的"
