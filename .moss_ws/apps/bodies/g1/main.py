"""
G1 Body App — 进程入口。

由 Circus 管理生命周期。构造 DDS 连接 → 创建 SDK clients → 构建 channel → Matrix 注册。
连接失败抛异常 → 进程退出 → Circus 重启。
"""

import os

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    nic = os.environ.get("UNITREE_G1_NIC", "eth0")

    # DDS + monitor + AudioClient (由 contrib 管理单例)
    from ghoshell_moss_contrib.unitree.g1 import bootstrap, get_audio_client
    bootstrap(nic)

    # Loco + Arm clients
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()

    audio = get_audio_client()

    from ghoshell_moss_contrib.unitree.g1.channel import build_g1_channel
    channel = build_g1_channel(loco_client=loco, arm_client=arm, audio_client=audio)

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    matrix = Matrix.discover()
    matrix.run(main)