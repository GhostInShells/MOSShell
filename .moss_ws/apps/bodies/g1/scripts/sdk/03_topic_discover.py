#!/usr/bin/env python3
"""
调查: DDS topic 发现 — 当前活跃的 topic 列表
决策: 对比 docs/sdk-topics.md 清单，确认哪些 topic 实际在 DDS 总线上可见

SDK 路径: src/unitree_sdk2_python/
对应文件:
  unitree_sdk2py/core/channel.py              — ChannelFactory, ChannelSubscriber
  unitree_sdk2py/idl/unitree_hg/msg/dds_.py   — G1 消息类型
  unitree_sdk2py/idl/unitree_go/msg/dds_.py   — Go2 共享类型

前置:
  G1 开机 + 交换机通电 + 以太网连接 PC2
  source .venv/bin/activate
  python 00_import_verify.py  # 必须先通过

用法:
  python 03_topic_discover.py <networkInterface>
  python 03_topic_discover.py eth0
"""
import sys
import time

print("=== DDS Topic 发现 ===\n")

if len(sys.argv) < 2:
    print("用法: python 03_topic_discover.py <networkInterface>")
    print("示例: python 03_topic_discover.py eth0")
    sys.exit(1)

nic = sys.argv[1]

# 初始化 DDS — unitree_sdk2py/core/channel.py: ChannelFactoryInitialize()
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK: DDS 初始化成功\n")
except Exception as e:
    print(f"FAIL: DDS 初始化失败: {e}")
    print("检查: 网卡名是否正确？G1 是否开机？以太网连接是否正常？")
    sys.exit(1)

# 用 cyclonedds 内置发现 — cyclonedds.core
try:
    from cyclonedds.domain import DomainParticipant
    dp = DomainParticipant(0)
    time.sleep(3)
    print("已创建临时 DomainParticipant(0)，等待 3s 发现...")
    print("(cyclonedds Python API 对 builtin topic 的支持有限)\n")
except ImportError:
    print("无法直接访问 cyclonedds builtin discovery\n")

print("=== 已知 Topic 清单 (来自 docs/sdk-topics.md) ===\n")

known_topics = [
    ("rt/lowstate", "hg", "LowState_", "底层反馈 IMU+电机+遥控器"),
    ("rt/lf/lowstate", "hg", "LowState_", "底层反馈-低频"),
    ("rt/lowcmd", "hg", "LowCmd_", "底层控制命令"),
    ("rt/arm_sdk", "hg", "LowCmd_", "上肢+腰 DDS 控制"),
    ("rt/dex3/left/state", "hg", "HandState_", "左灵巧手状态"),
    ("rt/dex3/left/cmd", "hg", "HandCmd_", "左灵巧手控制"),
    ("rt/dex3/right/state", "hg", "HandState_", "右灵巧手状态"),
    ("rt/dex3/right/cmd", "hg", "HandCmd_", "右灵巧手控制"),
    ("rt/lf/bmsstate", "hg", "BmsState_", "电池"),
    ("rt/lf/mainboardstate", "hg", "MainBoardState_", "主板"),
    ("rt/odommodestate", "go2", "IMUState_", "里程计"),
    ("rt/lf/odommodestate", "go2", "IMUState_", "里程计-低频"),
    ("rt/secondary_imu", "hg", "IMUState_", "机身 IMU"),
    ("rt/lf/secondary_imu", "hg", "IMUState_", "机身 IMU-低频"),
    ("rt/sportmodestate", "go2", "SportModeState_", "运动模式状态"),
    ("rt/audio_msg", "?", "String_ (JSON)", "ASR 结果"),
    ("rt/utlidar/cloud_livox_mid360", "?", "PointCloud2_", "LiDAR 点云"),
    ("rt/utlidar/imu_livox_mid360", "?", "Imu_", "LiDAR IMU"),
]

print(f"{'Topic':<35} {'组':<5} {'类型':<20} 说明")
print("-" * 85)
for topic, group, msg_type, desc in known_topics:
    print(f"{topic:<35} {group:<5} {msg_type:<20} {desc}")

print(f"\n共 {len(known_topics)} 个已知 topic\n")

print("=== 验证建议 ===")
print("""
cyclonedds Python API 对 builtin topic 的访问有限制。
建议的验证方式:

1. 用 SDK ChannelSubscriber 逐个尝试订阅已知 topic (纯读, 低风险)
2. 订阅成功 = topic 存在
3. 订阅失败 = topic 不存在或类型不匹配

下一步: 写 04_topic_subscription_test.py 做逐个订阅验证。
""")
