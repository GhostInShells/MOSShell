"""lipsync — 从 AudioSampleTopic (说侧) 驱动唇形参数.

自动唇动的框架能力: 订阅 ``audio/sample`` (跨进程, 走 Matrix 的 topic 桥), 过滤
``role == "ghost"`` 的采样, 把响度映射到唇形参数. 采样约 5Hz (~200ms), 足够平滑.

同时把"正在说话"这个连续状态报给页面 (``set_speaking``): 模型自带的动作曲线普遍驱动
嘴部参数, 循环待机动作会和唇动抢同一个参数, 所以说话期间待机必须让位.

TopicService 从 channel 运行时的容器里取 (``CommandUtil.force_get_contract``) ——
生命周期函数在 ``ChannelCtx`` 下运行, 与 matrix.session.topics 是同一个实例.

唇形参数来自 ``avatar.lip_param()``: 优先 ``model3.json`` 的 ``Groups.LipSync`` 声明
(KD4), 声明为空时回退到名字像 mouth-open 的参数. 都没有 → 唇动关闭.

"command 即真相" 的例外: 唇动是连续驱动 (像 SDK 的 auto lip-sync), 会覆盖 ``_state``
里的嘴部参数. ghost 想手动控嘴时由它自己决定 (见 NODE.md 的 KD4 硬约束).
"""

from __future__ import annotations

import asyncio

from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.types.topics import AudioSampleTopic

from .avatar import Avatar

# 响度 → 嘴开度 (0..1). peak 是 0..1 峰值振幅, 说侧峰值通常 <0.5, 乘增益后夹到 1.
_GAIN = 2.0
# 连续静默判定: 超过这个时间没新样本, 嘴回闭 (让 TTS 停顿/结束时有自然闭口).
_IDLE_TIMEOUT = 0.35


def mouth_open(sample: AudioSampleTopic) -> float:
    return min(sample.peak * _GAIN, 1.0)


async def run_lip_sync(avatar: Avatar) -> None:
    """订阅说侧采样并驱动唇形. 作为 channel 的 ``build.running`` 生命周期运行."""
    param = avatar.lip_param()
    if param is None:
        avatar.logger.info("avatar %s: no lip param, lip-sync off", avatar.name)
        return

    topics = CommandUtil.force_get_contract(TopicService)
    subscriber = topics.subscribe_model(AudioSampleTopic, maxsize=1)
    avatar.logger.info("avatar %s: lip-sync on %s", avatar.name, param)

    async with subscriber:
        while True:
            try:
                sample = await subscriber.poll_model(timeout=_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                # 说侧停了一段时间 → 闭嘴, 待机动画可以回来了.
                avatar.set_speaking(False)
                # 每 _IDLE_TIMEOUT 重设一次, 冗余但无害 (参数合并).
                if avatar.lip_sync_enabled:
                    avatar.param(param, 0.0)
                continue
            if sample is None:
                continue
            if not avatar.lip_sync_enabled:
                # 模型手动控嘴时暂停自动唇动 (模型输出优先), 只丢弃采样不驱动.
                continue
            speaking = sample.role == "ghost"
            avatar.set_speaking(speaking)
            avatar.param(param, mouth_open(sample) if speaking else 0.0)
