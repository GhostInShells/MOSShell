#!/usr/bin/env python3
"""
验证: 所有 SDK 核心模块可 import
决策: 任一 import 失败 → SDK 安装不完整，阻塞后续所有 SDK 实验

SDK 路径: src/unitree_sdk2_python/
对应模块:
  unitree_sdk2py/core/      — DDS Channel 基础设施
  unitree_sdk2py/rpc/        — RPC Client 基类
  unitree_sdk2py/g1/loco/    — LocoClient 运动控制
  unitree_sdk2py/g1/arm/     — G1ArmActionClient 手臂预设
  unitree_sdk2py/g1/audio/   — AudioClient 音频
  unitree_sdk2py/comm/       — MotionSwitcherClient
  unitree_sdk2py/b2/         — RobotStateClient
  unitree_sdk2py/idl/        — DDS 消息类型定义
  unitree_sdk2py/utils/      — CRC/线程 工具

前置:
  cd .moss_ws/apps/bodies/g1
  source .venv/bin/activate
  # 确认 cyclonedds 已安装: pip list | grep cyclonedds
  # 确认 unitree_sdk2py 可 import: python -c "import unitree_sdk2py"
"""
import sys

modules = {
    # 核心基础设施 — unitree_sdk2py/core/
    "core.channel": "unitree_sdk2py.core.channel",
    "core.channel_config": "unitree_sdk2py.core.channel_config",
    "core.channel_name": "unitree_sdk2py.core.channel_name",

    # RPC — unitree_sdk2py/rpc/
    "rpc.client": "unitree_sdk2py.rpc.client",

    # G1 loco — unitree_sdk2py/g1/loco/
    "g1.loco.g1_loco_client": "unitree_sdk2py.g1.loco.g1_loco_client",
    "g1.loco.g1_loco_api": "unitree_sdk2py.g1.loco.g1_loco_api",

    # G1 arm — unitree_sdk2py/g1/arm/
    "g1.arm.g1_arm_action_client": "unitree_sdk2py.g1.arm.g1_arm_action_client",
    "g1.arm.g1_arm_action_api": "unitree_sdk2py.g1.arm.g1_arm_action_api",

    # G1 audio — unitree_sdk2py/g1/audio/
    "g1.audio.g1_audio_client": "unitree_sdk2py.g1.audio.g1_audio_client",
    "g1.audio.g1_audio_api": "unitree_sdk2py.g1.audio.g1_audio_api",

    # 跨机器人共享 — unitree_sdk2py/comm/, unitree_sdk2py/b2/
    "comm.motion_switcher.motion_switcher_client": "unitree_sdk2py.comm.motion_switcher.motion_switcher_client",
    "b2.robot_state.robot_state_client": "unitree_sdk2py.b2.robot_state.robot_state_client",

    # IDL — unitree_sdk2py/idl/
    "idl.unitree_hg.msg.dds_": "unitree_sdk2py.idl.unitree_hg.msg.dds_",
    "idl.unitree_go.msg.dds_": "unitree_sdk2py.idl.unitree_go.msg.dds_",

    # 工具 — unitree_sdk2py/utils/
    "utils.crc": "unitree_sdk2py.utils.crc",
    "utils.thread": "unitree_sdk2py.utils.thread",
}

print("=== SDK Import 验证 ===\n")

passed = 0
failed = 0
for name, path in modules.items():
    try:
        __import__(path)
        print(f"  OK  {name}")
        passed += 1
    except ImportError as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

print(f"\n结果: {passed}/{passed+failed} passed, {failed} failed")

if failed > 0:
    print("\n阻塞: SDK 安装不完整。检查 uv sync 是否包含 unitree_sdk2py。")
    sys.exit(1)
else:
    print("OK: 所有模块可 import。")
