#!/usr/bin/env python3
"""
G1 Channel + SDK 第一案例 — channel runtime 独立运行验证。

本脚本对应"第一个 channel + sdk 实现案例":
  通过 chan.bootstrap() 让 g1_channel 独立运行 (不走 main.py / Matrix / MCP)，
  按 test_py_channel 范式直接调 runtime.execute_command()，
  验证 channel layer 的 CTML 入口契约 + SDK 调用链路全通。

验证序列 (audio + arm 混合):
  1. bodies_g1.audio:led_control(0, 0, 255)     — 蓝灯 (开始)
  2. bodies_g1.audio:get_volume()                — observe: 当前音量
  3. bodies_g1.audio:set_volume(50)              — 调音量
  4. bodies_g1.arm:list_actions()                — observe: 可用动作清单 (RPC 返回数据)
  5. bodies_g1.arm:execute_action("face wave")   — 实际动作 (需 G1 落座)
  6. wait 3s
  7. bodies_g1.arm:release_arm()                 — 复位
  8. bodies_g1.audio:say("channel 链路打通")     — 音频确认
  9. bodies_g1.audio:led_control(0, 255, 0)      — 绿灯 (成功)
  10. wait 3s
  11. bodies_g1.audio:led_control(0, 0, 0)       — 灯灭 (收尾)

安全:
  G1 必须处于 Sit (落座) 模式。手臂周围无遮挡。
  人类在场 + 遥控器握 L2+B 急停权。

用法: python 15_channel_action.py <networkInterface>
"""
import asyncio
import sys


async def run(nic: str) -> int:
    # ── SDK 初始化 ──
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    print("初始化 SDK clients...")
    loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
    arm = G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
    audio = AudioClient(); audio.SetTimeout(10.0); audio.Init()
    print("OK\n")

    # ── 构建 channel ──
    from g1_channel import build_g1_channel
    chan = build_g1_channel(
        loco_client=loco,
        arm_client=arm,
        audio_client=audio,
    )
    print(f"Channel 构建完成: id={chan.id()}\n")

    # ── 启动 runtime ──
    async with chan.bootstrap() as runtime:
        assert runtime.is_running(), "channel runtime 未启动"
        print(f"Runtime 启动 OK | 子 channel: {list(runtime.sub_channels().keys())}")
        print(f"已注册 metas: {len(runtime.metas())}\n")

        # ── 1. LED 蓝 ──
        print("=" * 50)
        print("[1] audio:led_control(0, 0, 255) — 蓝灯")
        r = await runtime.execute_command("audio:led_control", args=(0, 0, 255))
        print(f"    返回: {r}")
        await asyncio.sleep(2)

        # ── 2. 读音量 ──
        print("\n[2] audio:get_volume() — 读取音量 (observe)")
        r = await runtime.execute_command("audio:get_volume")
        print(f"    返回: {r}")

        # ── 3. 设音量 ──
        print("\n[3] audio:set_volume(50)")
        r = await runtime.execute_command("audio:set_volume", args=(50,))
        print(f"    返回: {r}")
        await asyncio.sleep(1)

        # ── 4. 列出手臂动作 ──
        print("\n[4] arm:list_actions() — RPC 返回数据 (observe)")
        r = await runtime.execute_command("arm:list_actions")
        print(f"    返回: {r}")

        # ── 安全门: 人类确认落座 ──
        print("\n" + "=" * 50)
        print("⚠ 下一步执行手臂动作。请确认:")
        print("  - G1 已落座 (Sit 模式)")
        print("  - 手臂周围 1m 内无任何遮挡")
        print("  - 遥控器在手，L2+B 急停就绪")
        ans = input("满足以上条件? (yes/no): ").strip().lower()
        if ans != "yes":
            print("已跳过手臂动作。直接进入收尾。")
            skip_arm = True
        else:
            skip_arm = False

        if not skip_arm:
            # ── 5. 执行手臂动作 ──
            print("\n[5] arm:execute_action('face wave')")
            print("    3 秒后执行...")
            for i in range(3, 0, -1):
                print(f"      {i}...")
                await asyncio.sleep(1)
            r = await runtime.execute_command("arm:execute_action", args=("face wave",))
            print(f"    返回: {r}")

            # ── 6. 等动作完成 ──
            print("\n[6] 等待 3s 让动作完成...")
            await asyncio.sleep(3)

            # ── 7. 复位 ──
            print("\n[7] arm:release_arm() — 复位")
            r = await runtime.execute_command("arm:release_arm")
            print(f"    返回: {r}")
            await asyncio.sleep(2)

        # ── 8. 音频确认 ──
        print("\n[8] audio:say('channel 链路打通')")
        r = await runtime.execute_command(
            "audio:say",
            args=("channel 链路打通", 0),
        )
        print(f"    返回: {r}")
        await asyncio.sleep(4)

        # ── 9. LED 绿 ──
        print("\n[9] audio:led_control(0, 255, 0) — 绿灯成功")
        r = await runtime.execute_command("audio:led_control", args=(0, 255, 0))
        print(f"    返回: {r}")
        await asyncio.sleep(3)

        # ── 10. LED 灭 ──
        print("\n[10] audio:led_control(0, 0, 0) — 收尾")
        r = await runtime.execute_command("audio:led_control", args=(0, 0, 0))
        print(f"    返回: {r}")

        print("\n" + "=" * 50)
        print("Channel 第一案例完成。")
        print("\n验证结论:")
        print("  [ ] LED 蓝/绿/灭 三色按序变化？")
        print("  [ ] get_volume 是否返回数字 (而非 dict 字面量)？")
        print("  [ ] list_actions 是否返回非空动作清单？")
        print("  [ ] face wave 是否在 G1 上实际执行？")
        print("  [ ] say 文本是否被播放 (TTS 质量已知低，能听到即可)？")
        print("  [ ] 链路全通 → 可以进入 Matrix.provide_channel 接 MCP 阶段")

    return 0


def main():
    if len(sys.argv) < 2:
        print("用法: python 15_channel_action.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    code = asyncio.run(run(nic))
    sys.exit(code)


if __name__ == "__main__":
    main()