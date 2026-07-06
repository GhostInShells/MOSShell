"""
_fsm_sen_state_transitions — story_202607_fsm 依赖无关场景测试.

═══════════════════════════════════════════════════════════════════════════════
一句话场景
═══════════════════════════════════════════════════════════════════════════════

模拟 G1 遥控器按键 + FSM 变化, 覆盖 story_202607_fsm 的核心行为:
根阀门 / 授权递增归零 / 下游按键分发 / sport_mode 变化 / warrant 三态 (正常/内部中断/入场快速失败).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════

    .venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._fsm_sen_state_transitions

═══════════════════════════════════════════════════════════════════════════════
前置
═══════════════════════════════════════════════════════════════════════════════

不需要 G1, 不需要 PC2, 不需要 unitree_sdk2py 运行时行为. **macOS 亦可跑**.
只依赖 python stdlib + pydantic + ghoshell_moss.

用 control_pad._configure_for_testing + fsm._configure_for_testing 跳过 sdk 注册,
用 control_pad._dispatch_press_for_testing 注入按键, fsm._inject_sport_mode_for_testing 注入 FSM.

═══════════════════════════════════════════════════════════════════════════════
预期
═══════════════════════════════════════════════════════════════════════════════

全部 assert 通过, 退出码 0. 每个 scenario 打印 `PASS: <name>`.
任一失败 assert / exception → 退出码非 0.

═══════════════════════════════════════════════════════════════════════════════
安全要点
═══════════════════════════════════════════════════════════════════════════════

无 — 无实机, 无网络, 无 subprocess.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ghoshell_moss.core.blueprint.channel_builder import ObserveError

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad
from ghoshell_moss_contrib.unitree.g1.runtime import story_202607_fsm as fsm
from ghoshell_moss_contrib.unitree.g1.sdk import FsmMode


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scenario")


DEBOUNCE_CLEAR = 0.25  # > control_pad global_min_debounce_sec (0.05) 也 > default (0.20)


def _reset_all():
    """每个 scenario 前重置状态, 清屏."""
    control_pad._configure_for_testing()
    fsm._configure_for_testing()


def scenario_1_root_gate():
    """L1+Start 进 AI, L1+Select 退 AI. 三元组同步更新, change listener 被通知."""
    print("\n─── scenario 1: root gate ─────────────────────────")
    _reset_all()

    events: list[fsm.StateSnapshot] = []
    handle = fsm.register_change_callback(lambda snap: events.append(snap))

    assert fsm.read() == (False, FsmMode.UNKNOWN, 0), "initial state"

    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    assert fsm.get_ai_mode() is True, "L1+Start should enter AI mode"
    assert fsm.get_auth_level() == 0, "auth starts at 0"

    control_pad._dispatch_press_for_testing("select", {"l1", "select"})
    assert fsm.get_ai_mode() is False, "L1+Select should exit AI mode"

    fsm.unregister_change_callback(handle)
    assert len(events) >= 2, f"expected ≥2 change events, got {len(events)}"
    print(f"  received {len(events)} change events, transitions OK.")
    print("PASS: scenario_1_root_gate")


def scenario_2_auth_transitions():
    """AI 模式内 L1+方向直选授权档: 上=0, 右=1, 下=2, 左=3."""
    print("\n─── scenario 2: auth direct-select ────────────────")
    _reset_all()

    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    assert fsm.get_auth_level() == 0

    # 方向 → 期望档位
    sequence = [
        ("right", {"l1", "right"}, 1),
        ("down",  {"l1", "down"},  2),
        ("left",  {"l1", "left"},  3),
        ("up",    {"l1", "up"},    0),
        ("left",  {"l1", "left"},  3),
        ("up",    {"l1", "up"},    0),
    ]
    for i, (btn, keys, exp) in enumerate(sequence):
        time.sleep(DEBOUNCE_CLEAR)
        control_pad._dispatch_press_for_testing(btn, keys)
        assert fsm.get_auth_level() == exp, \
            f"step #{i+1} ({btn}): expected {exp}, got {fsm.get_auth_level()}"

    # 归零后, AI 模式仍在
    assert fsm.get_ai_mode() is True, "auth=0 not exit AI mode"

    print("  R→1, D→2, L→3, U→0 直选 OK; AI mode 保持.")
    print("PASS: scenario_2_auth_transitions")


def scenario_3_button_dispatch():
    """AI 模式下 X/A/Y → 下游 button listener 按语义名分发."""
    print("\n─── scenario 3: button dispatch ───────────────────")
    _reset_all()

    control_pad._dispatch_press_for_testing("start", {"l1", "start"})

    received: list[str] = []
    handle = fsm.register_button_callback(lambda name: received.append(name))

    for btn in ["x", "a", "y"]:
        time.sleep(DEBOUNCE_CLEAR)
        control_pad._dispatch_press_for_testing(btn, {btn})

    fsm.unregister_button_callback(handle)
    assert set(received) == {"interrupt", "trigger", "audio_toggle"}, \
        f"expected {{interrupt, trigger, audio_toggle}}, got {set(received)}"
    print(f"  received: {received}")
    print("PASS: scenario_3_button_dispatch")


def scenario_4_button_dispatch_off_when_not_ai():
    """AI 模式外, X/A/Y binding 常驻但 _dispatch_button 按 _ai_mode 关闸, 下游 listener 不触发."""
    print("\n─── scenario 4: buttons silent off-mode ───────────")
    _reset_all()

    received: list[str] = []
    handle = fsm.register_button_callback(lambda name: received.append(name))

    # AI 模式外按 X — binding 常驻但 _dispatch_button 按 _ai_mode 关闸, 下游 listener 不触发
    control_pad._dispatch_press_for_testing("x", {"x"})
    assert len(received) == 0, f"AI off-mode X should not dispatch, got {received}"

    # 进 AI 后按 X — 应触发
    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("x", {"x"})
    assert received == ["interrupt"], f"expected [interrupt], got {received}"

    # 退 AI 后再按 X — binding 仍在, 但 _dispatch_button 关闸, 下游不触发
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("select", {"l1", "select"})
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("x", {"x"})
    assert received == ["interrupt"], f"AI off-mode again, X should not dispatch, got {received}"

    fsm.unregister_button_callback(handle)
    print("  AI-mode dispatch gating verified (bindings 常驻, gate 在 _dispatch_button).")
    print("PASS: scenario_4_button_dispatch_off_when_not_ai")


def scenario_5_sport_mode_injection():
    """sport_mode 变化触发 change listener, need_fsm_state 立即感知."""
    print("\n─── scenario 5: sport_mode injection ──────────────")
    _reset_all()

    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("right", {"l1", "right"})  # auth=1

    events: list[fsm.StateSnapshot] = []
    handle = fsm.register_change_callback(lambda snap: events.append(snap))

    # 初始 UNKNOWN, 注入 STAND
    fsm._inject_sport_mode_for_testing(int(FsmMode.STAND))
    assert fsm.get_sport_mode() == FsmMode.STAND
    assert fsm.need_fsm_state([([FsmMode.STAND], [1])]) is True
    assert fsm.need_fsm_state([([FsmMode.WALK_RUN], [1])]) is False

    # 再切 WALK_RUN
    fsm._inject_sport_mode_for_testing(int(FsmMode.WALK_RUN))
    assert fsm.get_sport_mode() == FsmMode.WALK_RUN
    assert fsm.need_fsm_state([([FsmMode.WALK_RUN], [1, 2])])

    fsm.unregister_change_callback(handle)
    assert len(events) >= 2, f"expected ≥2 sport_mode changes, got {len(events)}"
    print(f"  UNKNOWN → STAND → WALK_RUN OK; need_fsm_state 感知一致.")
    print("PASS: scenario_5_sport_mode_injection")


def scenario_6_need_fsm_state_gate():
    """AI 模式 off 时 need_fsm_state 无条件 False."""
    print("\n─── scenario 6: need_fsm_state gate ───────────────")
    _reset_all()
    # AI off, sport_mode=UNKNOWN, auth=L0
    assert not fsm.get_ai_mode()
    assert fsm.need_fsm_state([([FsmMode.UNKNOWN], [0])]) is False, \
        "AI off 时任何 requirement 应返回 False"
    print("  AI-off 无条件 False OK.")
    print("PASS: scenario_6_need_fsm_state_gate")


async def scenario_7_warrant_normal_completion():
    """Warrant 内动作跑完, 状态不变 → 无中断, 正常 return."""
    print("\n─── scenario 7: warrant normal completion ─────────")
    _reset_all()
    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    fsm._inject_sport_mode_for_testing(int(FsmMode.STAND))
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("right", {"l1", "right"})

    guard = lambda: fsm.need_fsm_state([([FsmMode.STAND], [1])])
    assert guard(), "guard should hold before warrant"

    completed = False
    async with fsm.warrant(guard):
        await asyncio.sleep(0.05)
        completed = True
    assert completed, "warrant body should run to completion"
    print("  action ran to completion, no ObserveError.")
    print("PASS: scenario_7_warrant_normal_completion")


async def scenario_8_warrant_internal_cancel():
    """Warrant 内状态变化 → guard fails → CancelledError → __aexit__ 转 ObserveError."""
    print("\n─── scenario 8: warrant internal cancel ───────────")
    _reset_all()
    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    fsm._inject_sport_mode_for_testing(int(FsmMode.STAND))
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("right", {"l1", "right"})  # auth=1

    guard = lambda: fsm.need_fsm_state([([FsmMode.STAND], [1])])

    async def do_work() -> str:
        async with fsm.warrant(guard):
            await asyncio.sleep(1.0)  # 将被 warrant cancel
        return "completed_no_cancel"

    task = asyncio.create_task(do_work())
    await asyncio.sleep(0.1)  # 让 warrant 进入 aenter

    # 触发状态变化: 1 → 0, guard fails
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("up", {"l1", "up"})

    try:
        result = await task
        raise AssertionError(f"expected ObserveError, got result={result!r}")
    except ObserveError as e:
        msg = str(e)
        assert "state change" in msg or "auth=0" in msg, f"observe msg unexpected: {msg}"
        print(f"  ObserveError raised as expected: {msg}")
    print("PASS: scenario_8_warrant_internal_cancel")


async def scenario_9_warrant_fast_fail():
    """Warrant __aenter__ 时 guard 已 False → 立刻 raise ObserveError, 不进入 body."""
    print("\n─── scenario 9: warrant fast-fail on entry ────────")
    _reset_all()
    # 未进 AI 模式 → guard always False
    guard = lambda: fsm.need_fsm_state([([FsmMode.STAND], [1])])
    assert not guard()

    body_entered = False
    try:
        async with fsm.warrant(guard):
            body_entered = True
    except ObserveError as e:
        assert not body_entered, "body should not be entered on fast fail"
        print(f"  fast fail OK: {e}")
    else:
        raise AssertionError("expected ObserveError on fast fail")
    print("PASS: scenario_9_warrant_fast_fail")


async def scenario_10_warrant_external_cancel_passthrough():
    """外部 cancel (非 warrant 触发) 应原样传播 CancelledError, warrant 不吃."""
    print("\n─── scenario 10: warrant external cancel ──────────")
    _reset_all()
    control_pad._dispatch_press_for_testing("start", {"l1", "start"})
    fsm._inject_sport_mode_for_testing(int(FsmMode.STAND))
    time.sleep(DEBOUNCE_CLEAR)
    control_pad._dispatch_press_for_testing("right", {"l1", "right"})

    guard = lambda: fsm.need_fsm_state([([FsmMode.STAND], [1])])

    async def do_work():
        async with fsm.warrant(guard):
            await asyncio.sleep(1.0)
        return "completed"

    task = asyncio.create_task(do_work())
    await asyncio.sleep(0.1)
    task.cancel()  # 模拟外部 shell 的 cancel

    try:
        await task
    except asyncio.CancelledError:
        print("  external CancelledError passed through cleanly.")
    except ObserveError as e:
        raise AssertionError(f"external cancel should not become ObserveError, got {e}")
    print("PASS: scenario_10_warrant_external_cancel_passthrough")


def main():
    scenario_1_root_gate()
    scenario_2_auth_transitions()
    scenario_3_button_dispatch()
    scenario_4_button_dispatch_off_when_not_ai()
    scenario_5_sport_mode_injection()
    scenario_6_need_fsm_state_gate()

    asyncio.run(scenario_7_warrant_normal_completion())
    asyncio.run(scenario_8_warrant_internal_cancel())
    asyncio.run(scenario_9_warrant_fast_fail())
    asyncio.run(scenario_10_warrant_external_cancel_passthrough())

    print("\n═══════════════════════════════════════════════════")
    print("ALL SCENARIOS PASSED.")
    print("═══════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
