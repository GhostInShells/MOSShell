"""
G1 Body App — 进程入口。

由 Circus 管理生命周期。当前阶段 A（云端文档摸底），channel 仅提供 instruction
声明开发进度，无实际命令。随阶段推进逐步添加命令。
"""

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix


def build_g1_channel():
    channel = new_channel(
        name="bodies_g1",
        description="Unitree G1 人形机器人身体控制",
    )

    channel.build.instruction(
        "G1 channel 开发中。当前阶段: 文档摸底 + 源码分析。"
        "无可调用命令。详细进度和架构决策见 docs/index.md。"
        "下一步阶段 B: 代码仓库摸底（读 SDK 源码）。"
    )

    return channel


async def main(matrix: Matrix):
    channel = build_g1_channel()
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    matrix = Matrix.discover()
    matrix.run(main)
