#!/usr/bin/env python3
"""
Channel 行为实验: 连续命令平滑过渡。

命题: channel 连续发出两个 Move 命令时，第二个是打断第一个还是排队？
      如果命令是 blocking=True，Damp() 能否在 Move 运行时插队？

SDK 层 Move() 调 SetVelocity，第二个调用直接覆盖前一个（DDS 机制）。
Channel 层引入 blocking 后，队列语义改变了这一行为。
本脚本对比 blocking vs nonblocking 两种 channel 设计下的连续 Move 表现。

前置: G1 开机 + RPC 服务运行
用法: python 01_sequential_transition.py <networkInterface> [--blocking|--nonblocking]
"""
import sys
import asyncio
import time

from ghoshell_moss.core.blueprint.channel_builder import new_channel


def build_loco_channel(client, blocking: bool = True):
    """构建运动控制 channel — 单命令模式，聚焦 Move + Damp 行为。"""
    chan = new_channel(
        name="loco_test",
        description=f"sequential transition test (blocking={blocking})",
    )

    @chan.build.command(blocking=blocking)
    async def move(vx: float, vy: float, vyaw: float, continuous: bool = False) -> str:
        """移动: vx 前后, vy 横向, vyaw 旋转。continuous=True 持续移动。"""
        code = client.Move(vx, vy, vyaw, continous_move=continuous)
        return "ok" if code == 0 else f"err:{code}"

    @chan.build.command(blocking=blocking, call_soon=True)
    async def damp() -> str:
        """急停 — call_soon=True 确保优先级。"""
        code = client.Damp()
        return "ok" if code == 0 else f"err:{code}"

    @chan.build.command(blocking=False)
    async def stop_move() -> str:
        code = client.StopMove()
        return "ok" if code == 0 else f"err:{code}"

    return chan


async def test_sequential_move(chan, label: str):
    """发送两个连续 Move，观察过渡。"""
    print(f"\n{'='*50}")
    print(f"[{label}] 连续 Move: 前进 0.3 m/s -> 后退 0.2 m/s")
    print(f"{'='*50}")

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()

        t0 = time.monotonic()
        task1 = runtime.execute_command("move", kwargs={"vx": 0.3, "vy": 0.0, "vyaw": 0.0})
        await asyncio.sleep(0.5)  # 让第一个命令执行 0.5s
        t1 = time.monotonic()
        task2 = runtime.execute_command("move", kwargs={"vx": -0.2, "vy": 0.0, "vyaw": 0.0})
        t2 = time.monotonic()

        r1 = await task1
        r2 = await task2

        print(f"Move(+0.3) 发出: {t1 - t0:.3f}s, 完成: {time.monotonic() - t0:.3f}s, 结果: {r1}")
        print(f"Move(-0.2) 发出: {t2 - t0:.3f}s, 完成: {time.monotonic() - t0:.3f}s, 结果: {r2}")


async def test_damp_interrupt(chan, label: str):
    """Move 中途发 Damp，观察中断行为。"""
    print(f"\n{'='*50}")
    print(f"[{label}] Move 中途 Damp 中断")
    print(f"{'='*50}")

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()

        t0 = time.monotonic()
        move_task = runtime.execute_command("move", kwargs={"vx": 0.3, "vy": 0.0, "vyaw": 0.0, "continuous": True})
        await asyncio.sleep(0.5)
        damp_task = runtime.execute_command("damp")
        t1 = time.monotonic()

        r_move = await move_task
        r_damp = await damp_task

        print(f"Move(continuous) 发出 -> 0.5s -> Damp 发出")
        print(f"Damp 发出: {t1 - t0:.3f}s")
        print(f"Move 结果: {r_move}")
        print(f"Damp 结果: {r_damp}")


async def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <networkInterface> [--blocking|--nonblocking]")
        sys.exit(1)

    nic = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--blocking"
    blocking = mode == "--blocking"

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("LocoClient 就绪")

    chan = build_loco_channel(loco, blocking=blocking)

    await test_sequential_move(chan, f"blocking={blocking}")
    await test_damp_interrupt(chan, f"blocking={blocking}")

    print("\n── 观察要点 ──")
    if blocking:
        print("blocking=True: Move 排队执行。第二个 Move 等待第一个完成后执行。")
        print("Damp(call_soon=True): 应清空队列后立刻执行 — 这才是正确的急停行为。")
    else:
        print("blocking=False: Move 并发。第二个 Move 立即覆盖前一个的 DDS topic。")
        print("急停在 nonblocking 模式下依赖 call_soon + priority 抢占。")
    print("两个模式对比回答: 急停应该用 blocking+call_soon 还是 nonblocking+priority?")
    print("连续 Move 场景下哪种更接近预期行为? 这些答案直接决定 channel 设计。")


if __name__ == "__main__":
    asyncio.run(main())
