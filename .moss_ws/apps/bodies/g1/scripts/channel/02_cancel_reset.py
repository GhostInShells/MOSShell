#!/usr/bin/env python3
"""
Channel 行为实验: 命令取消与状态复位。

命题: Task.cancel() 后 channel 的状态是什么？被取消的 SDK 动作是否自动复位？
      Damp() 是否可以作为"安全复位原语"在 finally 中调用？

Channel 命令如果是 async，支持 asyncio.CancelledError 生命周期。
sync 命令需要用 CommandUtil.is_task_done() 轮询取消状态。
本脚本测试两种实现方式在取消时的复位行为差异。

前置: G1 开机 + RPC 服务运行
用法: python 02_cancel_reset.py <networkInterface>
"""
import sys
import asyncio

from ghoshell_moss.core.blueprint.channel_builder import new_channel


def build_cancel_test_channel(client):
    """两个命令: async 版支持 cancel，sync 版不支持。对比复位行为。"""
    chan = new_channel(name="cancel_test", description="cancel + reset behavior test")

    @chan.build.command(blocking=True)
    async def move_async(vx: float, vy: float, vyaw: float) -> str:
        """async Move — 支持 CancelledError，finally 中 Damp 复位。"""
        try:
            code = client.Move(vx, vy, vyaw, continous_move=True)
            await asyncio.sleep(0.1)  # simulate async work
            return "ok" if code == 0 else f"err:{code}"
        except asyncio.CancelledError:
            print("  [move_async] CancelledError caught — Damp() 复位")
            client.Damp()
            return "cancelled+damp"

    @chan.build.command(blocking=True)
    def move_sync(vx: float, vy: float, vyaw: float) -> str:
        """sync Move — 通过 CommandUtil.is_task_done 检测取消。取消后不复位。"""
        from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

        code = client.Move(vx, vy, vyaw, continous_move=True)
        if CommandUtil.is_task_done():
            print("  [move_sync] task done detected — 但 sync 无法 finally Damp")
            return "cancelled_no_reset"
        return "ok" if code == 0 else f"err:{code}"

    @chan.build.command(blocking=True, call_soon=True)
    async def damp() -> str:
        """复位原语。"""
        code = client.Damp()
        return "ok" if code == 0 else f"err:{code}"

    return chan


async def test_async_cancel(runtime):
    """async 命令被 cancel 时，finally 中 Damp 是否执行？"""
    print("\n── async 命令 cancel ──")
    task = runtime.create_task("move_async", kwargs={"vx": 0.2, "vy": 0.0, "vyaw": 0.0})
    await asyncio.sleep(0.3)
    print("  发出 cancel...")
    task.cancel()
    try:
        result = await task
        print(f"  结果: {result}")
    except asyncio.CancelledError:
        print("  结果: CancelledError (未捕获 — finally 未执行?)")


async def test_sync_cancel(runtime):
    """sync 命令被 cancel 时，is_task_done 轮询是否生效？"""
    print("\n── sync 命令 cancel ──")
    task = runtime.create_task("move_sync", kwargs={"vx": 0.2, "vy": 0.0, "vyaw": 0.0})
    await asyncio.sleep(0.3)
    print("  发出 cancel...")
    task.cancel()
    try:
        result = await task
        print(f"  结果: {result}")
    except asyncio.CancelledError:
        print("  结果: CancelledError — sync 函数无法捕获 cancel")


async def test_damp_after_cancel(runtime):
    """cancel 后显式发 Damp — 能否复位？这是最安全的手动模式。"""
    print("\n── cancel + 显式 Damp ──")
    task = runtime.create_task("move_async", kwargs={"vx": 0.2, "vy": 0.0, "vyaw": 0.0})
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    result = await runtime.execute_command("damp")
    print(f"  Damp 结果: {result}")


async def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <networkInterface>")
        sys.exit(1)

    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("LocoClient 就绪")

    chan = build_cancel_test_channel(loco)

    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await test_async_cancel(runtime)
        await test_sync_cancel(runtime)
        await test_damp_after_cancel(runtime)

    print("\n── 观察要点 ──")
    print("1. async 命令的 CancelledError + finally Damp 是否可靠？")
    print("2. sync 命令被 cancel 后能否检测到？is_task_done 的时机正确吗？")
    print("3. cancel + 显式 Damp 是否为最安全的复位模式？")
    print(
        "结论决定: channel 命令统一用 async + finally Damp，"
        "还是 sync + 显式 Damp 由调用方负责。"
    )


if __name__ == "__main__":
    asyncio.run(main())
