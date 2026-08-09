"""G1 整机身体 node — 装配完整 channel 树作为 cell 膜.

装配内容 = 旧 unitree_g1 mode 的 channels.py 拓扑 (sdk.bootstrap + 组树),
换成 node 形态: 直接 provide g1_root 作为本 cell 的唯一膜.

macOS 不可测试 (cyclonedds 不编译). 等价代码在 G1 真机 (PC2) 验证:
    moss nodes run nodes/unitree/g1/control
"""

from ghoshell_moss.core.blueprint.matrix import Matrix

from ghoshell_moss_contrib.unitree.g1 import sdk

sdk.bootstrap()  # 整机基建: DDS + 三 client + monitor (import 路径顶部一次性)

from ghoshell_moss_contrib.unitree.g1.channels.g1_root import g1_root
from ghoshell_moss_contrib.unitree.g1.channels.fsm import g1_fsm
from ghoshell_moss_contrib.unitree.g1.channels.face_led import face_led
from ghoshell_moss_contrib.unitree.g1.channels.listener import listener_channel
from ghoshell_moss_contrib.unitree.g1.channels.asr import g1_asr
from ghoshell_moss_contrib.unitree.g1.channels.locomotion import locomotion_channel
from ghoshell_moss_contrib.unitree.g1.channels.arms import arms_channel


async def main(matrix: Matrix) -> None:
    g1_root.import_channels(g1_fsm)
    g1_root.import_channels(face_led)
    g1_root.import_channels(listener_channel)
    g1_root.import_channels(g1_asr)
    g1_root.import_channels(locomotion_channel)
    g1_root.import_channels(arms_channel)
    await matrix.provide_channel(g1_root)


if __name__ == "__main__":
    Matrix.discover().run(main)
