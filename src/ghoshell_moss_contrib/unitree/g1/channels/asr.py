"""
G1 远场 ASR Channel — 纯感知通道, 每帧暴露最近 3 条识别结果.

本 channel 不暴露命令, 不发 signal. 机体周围听到的话自动出现在 LLM context
每一帧里, LLM 无需调用任何命令. 结果不进对话历史 (顺行遗忘), 仅作当帧躯体感知.

与蓝牙耳机 ASR (g1.listener) 是独立通道:
- g1.asr    — G1 机体远场麦克风阵列, 听到周围人说的话
- g1.listener — 蓝牙耳机近场 ASR, 听到远程操作者说的话

runtime 生命周期 (start/stop) 由 main.py 管, 本 channel 只读不启停.
"""

import logging

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import asr
from ghoshell_moss_contrib.unitree.g1.channels._utils import check_channel_information

logger = logging.getLogger("moss.g1.channels.asr")

_INSTRUCTION = """\
G1 内置远场麦克风阵列 ASR (与蓝牙耳机 ASR 独立). 识别结果每帧自动进 context, 无需调用命令.

启用: 需在手机 App 开启"唤醒对话模式". 长时间无输入时提示检查.

局限: 中文为主, 英文弱; 3m 外识别率下降; 无声源方位 (angle 固定 0); \
无说话人识别 (speaker_id 固定 0); 情绪字段粗略; 逐字不准, 靠语义理解补偿.\
"""

g1_asr = new_channel(
    name="asr",
    description="G1 远场麦克风 ASR. 机体周围听到的话自动进 context.",
)

g1_asr.build.instruction(_INSTRUCTION)


@g1_asr.build.startup
async def _startup() -> None:
    asr.start()


@g1_asr.build.context_messages
async def _context() -> list[Message]:
    items = asr.peek_window(max_count=3)
    total = asr.peek_total_count()

    if not items:
        h = asr.health()
        if h["last_recv"] == 0:
            return [
                Message.new(tag="g1.asr", attributes={"status": "not_started"})
                .with_content("远场 ASR 未收到任何输入. 需在手机 App 开启唤醒对话模式.")
            ]
        return []

    return [asr.to_message(r) for r in items]


if __name__ == "__main__":
    check_channel_information(g1_asr)
