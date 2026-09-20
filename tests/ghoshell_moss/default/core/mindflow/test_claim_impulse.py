"""ClaimImpulse — mindflow 层主动拉取 (claim) 的协议测试.

``Mindflow.claim_impulse`` 是 ghost 刻意自省的显式动作: 把某个 nucleus 当前
持有的 impulse 拉进下一轮思考. 它区别于仲裁路径 (absorb / queued / buffered),
是一个命令式 API, 契约如下:

    - 消费: ``Nucleus.attended`` 物化 stub → full impulse (state 从 nucleus 内部 buffer
      物化, 不是 peek 直接读);
    - 折进观测管线: pending buffer + ``need_observe``, 保证下一帧必读;
    - 不强化当前 attention (无 ``absorb_impulse``);
    - 缓冲在 mindflow 层, attention abort 不清它, 折进下一个 attention 首帧.

这里用 ``InputSignalNucleus`` (真实 nucleus, ``peek`` 返回挑战 stub / ``attended``
物化 full) 在"输掉仲裁后保留"的状态下验证 claim 的物化与管线行为, 走标准
``thinking_loop`` 消费, 不 hack 接口.
"""
import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import (
    Impulse,
    InputSignalMeta,
    Priority,
    Thinking,
)
from ghoshell_moss.core.mindflow import BaseMindflow, InputSignalNucleus
from ghoshell_moss.message import Message

_INPUT_NUCLEUS = InputSignalNucleus.NAME


def _new_mindflow() -> BaseMindflow:
    return BaseMindflow(InputSignalNucleus())


def _defender_impulse() -> Impulse:
    """带保护期的 NOTICE defender, 让同优先级的 input signal 必输 (保留在 nucleus)."""
    impulse = Impulse(
        priority=Priority.NOTICE,
        messages=[Message.new().with_content("defender")],
    )
    impulse.protection_time = 10.0
    return impulse


def _input_signal(text: str):
    return InputSignalMeta().to_signal(
        Message.new().with_content(text),
        priority=Priority.NOTICE,
    )


def _impulse_text(impulse: Impulse) -> str:
    return " ".join(m.to_content_string() for m in impulse.messages)


def _percepts_text(moment) -> str:
    return " ".join(moment.percepts_texts())


async def _first_thinking(mindflow: BaseMindflow) -> Thinking:
    async for thinking in mindflow.thinking_loop():
        return thinking
    raise AssertionError("mindflow.thinking_loop() exited without yielding a thinking")


@pytest.mark.asyncio
async def test_claim_returns_none_for_unknown_or_empty_nucleus():
    """claim 无归属 / 无持有: 未知 nucleus 或空 nucleus 都返回 None, 不抛异常."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        assert mindflow.claim_impulse("no_such_nucleus") is None
        assert mindflow.claim_impulse(_INPUT_NUCLEUS) is None


@pytest.mark.asyncio
async def test_claim_materializes_and_consumes_retained_input():
    """claim 一个输掉仲裁后保留的 input impulse: 物化 full, 消费 nucleus, 强制下一轮.

    契约 (stub/full 二分): ``InputSignalNucleus.peek`` 只给挑战 stub (无 messages),
    ``attended`` 才物化 full. claim 必须拿到物化后的 full impulse 并折进管线,
    而不是把 stub 当载荷.
    """
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_impulse(_defender_impulse())
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            # input signal 输掉 → 保留在 input nucleus (suppress 只停 fire, 不清 cache).
            mindflow.add_signal(_input_signal("held_payload_xyz"))
            await asyncio.sleep(0.2)

            claimed = mindflow.claim_impulse(_INPUT_NUCLEUS)

            # 物化: claim 返回 full impulse (带 messages), 不是 stub.
            assert claimed is not None
            assert "held_payload_xyz" in _impulse_text(claimed)
            # 消费: nucleus buffer 被 attended 清空.
            assert mindflow.nuclei()[_INPUT_NUCLEUS].peek() is None
            # 强制下一轮观察.
            assert mindflow.moments.need_observe() is True
            thinking.abort("test done")


@pytest.mark.asyncio
async def test_claim_folds_payload_into_next_frame_in_same_attention():
    """claim 的载荷折进同一 attention 的回声帧 (need_observe 驱动下一帧).

    契约: claim 后不 abort, need_observe 让当前 attention 再跑一帧, 该帧 percepts
    折进 claim 载荷. 这钉住 claim → pending buffer → 回声帧折叠的完整链路.
    """
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_impulse(_defender_impulse())
        loop = mindflow.thinking_loop()
        first = await asyncio.wait_for(anext(loop), timeout=2.0)
        async with first:
            mindflow.add_signal(_input_signal("held_payload_xyz"))
            await asyncio.sleep(0.2)
            claimed = mindflow.claim_impulse(_INPUT_NUCLEUS)
            assert claimed is not None
            # 不 abort — 退出后 need_observe 驱动同 attention 的回声帧.

        second = await asyncio.wait_for(anext(loop), timeout=2.0)
        async with second:
            # 回声帧 percepts 折进了 claim 载荷.
            assert "held_payload_xyz" in _percepts_text(second.moment)
            second.abort("test done")


@pytest.mark.asyncio
async def test_claim_survives_attention_abort():
    """claim 后立刻 abort: 载荷缓冲在 mindflow 层, 折进下一个 attention 首帧.

    契约 (docstring 的 abort-survival 承诺): pending buffer 挂在 mindflow 上,
    不随 attention abort 消失. 下一个 attention 的首帧折进该载荷.
    """
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_impulse(_defender_impulse())
        loop = mindflow.thinking_loop()
        first = await asyncio.wait_for(anext(loop), timeout=2.0)
        att1 = first.attention
        async with first:
            mindflow.add_signal(_input_signal("held_payload_xyz"))
            await asyncio.sleep(0.2)
            claimed = mindflow.claim_impulse(_INPUT_NUCLEUS)
            assert claimed is not None
            # 立刻 abort — 回声帧不再消费 pending.
            att1.abort("claim then abort")
            await asyncio.wait_for(att1.wait_abort(), timeout=2.0)

        # 新 signal → 新 attention, 首帧折进 claim 载荷.
        mindflow.add_signal(_input_signal("second"))
        second = await asyncio.wait_for(anext(loop), timeout=2.0)
        async with second:
            assert "held_payload_xyz" in _percepts_text(second.moment)
            second.abort("test done")
