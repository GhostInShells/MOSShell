"""Impulse.strength=0 协议承诺单测 — yielded 仲裁路径.

钉住"strength=0 = 绝不竞争"的协议: 在 mindflow._challenge_attention 入口短路,
不分 defender/quiet, 不打任何 mode 分支, 不建 attention, 不 buffer 不 suppress,
fire 'yielded' verdict, 由 nucleus 自然清理缓存.

这是 Zen 静默 attention 心智模型的预留接口 — 调用方未来可组合
complete=False + strength=0 构造 "占住 attention 但不竞争" 的首包.

语义要点 (区分于把 yield 误解为"被挑战者退让"):
    正确理解是**挑战者根本不发起竞争**: strength=0 的 impulse 自己不挑战,
    所以 quiet 系统下也不该创建 attention. 旧实现把 strength=0 短路放进
    "有 defender" 分支, quiet 系统漏到 else 分支创建 attention — 只考虑了
    被挑战者的 yield, 没考虑挑战者不挑战.
    因此短路必须在 defender/quiet 分支之前, 且 verdict 恒为 'yielded'.

API 迁移 (重构 41f0cb63): 用 ``mindflow.thinking_loop()`` 取 Thinking,
``mindflow.project_percepts()`` → ``mindflow.moments.peek().percepts_messages()``.

覆盖路径:
- quiet 系统 (无 defender) + strength=0 → yielded, 不创建 attention
- 有 defender + strength=0 (各 mode: default/silent/notify) → yielded, defender 不动
- strength=0 + FATAL → 仍 yielded (yielded 优先级高于 FATAL 短路)
- yielded verdict 通过 hook 触发
- defender 字段在 yielded 时正确填充 (有 defender 时是 defender impulse, quiet 时是 None)
"""
import asyncio
import pytest

from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, ChallengeVerdict, Impulse, MindflowHook, Priority,
)
from ghoshell_moss.core.mindflow import BaseMindflow
from ghoshell_moss.message import Message


def _new_mindflow() -> BaseMindflow:
    return BaseMindflow()


def _imp(
        *,
        strength: int = 100,
        priority: int = Priority.NOTICE,
        mode: str = '',
        messages: list[Message] | None = None,
) -> Impulse:
    return Impulse(
        strength=strength,
        priority=priority,
        mode=mode,
        messages=messages or [Message.new().with_content('x')],
    )


async def _first_thinking(mindflow: BaseMindflow, timeout: float = 2.0):
    """阻塞拿第一个 Thinking. 生成器 finally 会重置 _is_looping_thinking."""
    async for thinking in mindflow.thinking_loop():
        return thinking
    raise AssertionError("mindflow.thinking_loop() exited without yielding")


async def _setup_defender(mindflow: BaseMindflow):
    """注入一个 NOTICE defender, 拿到它的 Thinking (调用方用 .attention)."""
    mindflow.add_impulse(_imp(priority=Priority.NOTICE))
    return await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)


class _VerdictCapture(MindflowHook):
    """收集 verdict 用于断言. 一个 challenger 一条记录."""

    def __init__(self):
        self.records: list[tuple[Impulse, Impulse | None, ChallengeVerdict]] = []

    def name(self) -> str:
        return 'verdict_capture'

    def on_impulse_challenged(self, challenger, defender, verdict):
        self.records.append((challenger, defender, verdict))


# ============================================================
# quiet 系统 (无 defender)
# ============================================================

@pytest.mark.asyncio
async def test_strength_zero_yields_in_quiet_system():
    """quiet + strength=0 → yielded, 不创建 attention (挑战者不挑战)."""
    mindflow = _new_mindflow()
    hook = _VerdictCapture()
    mindflow.with_hook(hook)
    async with mindflow:
        await mindflow.wait_started()
        yielded = _imp(strength=0)
        mindflow.add_impulse(yielded)
        # 给 consume loop 时间.
        await asyncio.sleep(0.2)
        # 协议命题 1: hook 收到 yielded verdict, defender 为 None (quiet).
        assert len(hook.records) == 1
        c, d, v = hook.records[0]
        assert v == 'yielded'
        assert d is None
        assert c is yielded
        # 协议命题 2: 挑战者不竞争 → 不创建 attention.
        assert mindflow.attention() is None


# ============================================================
# 有 defender, 各 mode 都应礼让
# ============================================================

@pytest.mark.asyncio
async def test_strength_zero_yields_with_default_mode_defender():
    """有 defender + strength=0 (default mode) → yielded, defender 不动."""
    mindflow = _new_mindflow()
    hook = _VerdictCapture()
    mindflow.with_hook(hook)
    async with mindflow:
        await mindflow.wait_started()
        defender_think = await _setup_defender(mindflow)
        async with defender_think:
            defender_imp = defender_think.attention.draw_from()
            hook.records.clear()  # 忽略 initial verdict.

            yielded = _imp(strength=0)
            mindflow.add_impulse(yielded)
            await asyncio.sleep(0.2)

            # 协议: yielded verdict + defender 字段正确.
            yielded_records = [r for r in hook.records if r[2] == 'yielded']
            assert len(yielded_records) == 1
            _, d, _ = yielded_records[0]
            assert d is defender_imp
            # defender 没被打扰.
            assert not defender_think.is_aborted()
            defender_think.abort('test done')


@pytest.mark.asyncio
async def test_strength_zero_with_silent_mode_still_yields_not_buffer():
    """strength=0 + mode=silent → yielded (strength=0 短路在 mode 分支之前).
    协议: silent 的 buffer 偏离不应被触发, messages 不进 mindflow buffer."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        defender_think = await _setup_defender(mindflow)
        async with defender_think:
            mindflow.add_impulse(_imp(
                strength=0,
                mode=ChallengeMode.silent.value,
                messages=[Message.new().with_content('yielded_msg')],
            ))
            await asyncio.sleep(0.2)
            # messages 没进 buffer (yielded 不走 buffer 分支).
            buffered = mindflow.moments.peek().percepts_messages()
            texts = [c['text'] for m in buffered for c in m.contents if 'text' in c]
            assert 'yielded_msg' not in texts
            defender_think.abort('test done')


@pytest.mark.asyncio
async def test_strength_zero_with_notify_mode_still_yields_not_buffer():
    """strength=0 + mode=notify → yielded (同上, mode 分支不触发)."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        defender_think = await _setup_defender(mindflow)
        async with defender_think:
            mindflow.add_impulse(_imp(
                strength=0,
                mode=ChallengeMode.notify.value,
                messages=[Message.new().with_content('yielded_msg')],
            ))
            await asyncio.sleep(0.2)
            buffered = mindflow.moments.peek().percepts_messages()
            texts = [c['text'] for m in buffered for c in m.contents if 'text' in c]
            assert 'yielded_msg' not in texts
            defender_think.abort('test done')


# ============================================================
# 优先级 — strength=0 短路高于 FATAL 短路
# ============================================================

@pytest.mark.asyncio
async def test_strength_zero_overrides_fatal_short_circuit():
    """协议立场: 即便 priority=FATAL, strength=0 仍 yielded.
    用例: 调用方矛盾意图 (FATAL 必胜 vs strength=0 礼让), yielded 是更明确的退出意图."""
    mindflow = _new_mindflow()
    hook = _VerdictCapture()
    mindflow.with_hook(hook)
    async with mindflow:
        await mindflow.wait_started()
        defender_think = await _setup_defender(mindflow)
        async with defender_think:
            hook.records.clear()
            mindflow.add_impulse(_imp(strength=0, priority=Priority.FATAL))
            await asyncio.sleep(0.2)
            # defender 没被 FATAL 抢占.
            assert not defender_think.is_aborted()
            # verdict 是 yielded, 不是 preempted.
            yielded_records = [r for r in hook.records if r[2] == 'yielded']
            assert len(yielded_records) == 1
            defender_think.abort('test done')
