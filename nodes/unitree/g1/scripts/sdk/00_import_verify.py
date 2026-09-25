#!/usr/bin/env python3
"""
验证: 两层 import sanity check
  第一段: unitree_sdk2py 各核心模块可 import (SDK 本身完整性)
  第二段: ghoshell_moss_contrib.unitree.g1.sdk 子包及暴露符号可 import (我方封装完整性)

两段都是纯 import, 不 bootstrap, 不连 DDS — 不需要 G1 上电即可跑.
任一段 fail → 阻塞后续所有 SDK 实验.

范围限定: 本脚本只验证 g1.sdk 层. runtime/channels/providers 由各自子包的脚本验证.

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

Contrib 路径: src/ghoshell_moss_contrib/unitree/g1/sdk/
  _sdk        — UNITREE_G1_SDK_PATH + nic 解析
  _bootstrap  — ChannelFactoryInit + 三 client 单例 + dump_state
  _monitor    — DDS subscriber 路由到 state
  _buttons    — 按键 callback 注册 (跑在 reader 线程)
  state       — 6 frozen dataclass + 模块级原子 getter

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

print(f"\nSDK 段结果: {passed}/{passed+failed} passed, {failed} failed")
if failed > 0:
    print("阻塞: SDK 安装不完整。检查 uv sync 是否包含 unitree_sdk2py。")


# ── 第二段: contrib g1.sdk 子包 import 验证 ────────────────────────────────

print("\n=== Contrib g1.sdk 子包 Import 验证 ===\n")

contrib_submodules = [
    "ghoshell_moss_contrib.unitree.g1",          # 空伞, 应可 import 但无 re-export
    "ghoshell_moss_contrib.unitree.g1.sdk",      # L1 子包入口
    "ghoshell_moss_contrib.unitree.g1.sdk._sdk",
    "ghoshell_moss_contrib.unitree.g1.sdk._bootstrap",
    "ghoshell_moss_contrib.unitree.g1.sdk._monitor",
    "ghoshell_moss_contrib.unitree.g1.sdk._buttons",
    "ghoshell_moss_contrib.unitree.g1.sdk.state",
]

# g1.sdk 包顶层应暴露的符号 (源头: sdk/__init__.py)
sdk_expected_symbols = [
    # bootstrap 生命周期 + 客户端取用
    "bootstrap", "is_bootstrapped",
    "get_audio_client", "get_loco_client", "get_arm_client",
    "get_network_interface", "dump_state",
    # state 类型
    "MotionState", "JointState", "JointsState", "IMUState",
    "RemoteState", "BatteryState", "HealthState",
    # state 原子读取 + 健康
    "motion", "joints", "imu", "remote", "battery", "health",
    "last_update", "is_started",
    # buttons callback
    "CallbackHandle",
    "register_button_callback", "unregister_button_callback",
    # env / sdk 路径
    "load_unitree_g1_sdk",
]

contrib_passed = 0
contrib_failed = 0

# 1) 子模块可 import
for path in contrib_submodules:
    try:
        __import__(path)
        print(f"  OK   import  {path}")
        contrib_passed += 1
    except Exception as e:  # 不限 ImportError — __init__ 内可能 raise 其他类型
        print(f"  FAIL import  {path}: {type(e).__name__}: {e}")
        contrib_failed += 1

# 2) g1.sdk 顶层暴露符号检查 (sdk/__init__.py 契约)
try:
    sdk_mod = __import__("ghoshell_moss_contrib.unitree.g1.sdk", fromlist=["*"])
    for name in sdk_expected_symbols:
        if hasattr(sdk_mod, name):
            print(f"  OK   symbol  g1.sdk.{name}")
            contrib_passed += 1
        else:
            print(f"  FAIL symbol  g1.sdk.{name} (missing in sdk/__init__.py)")
            contrib_failed += 1
except Exception as e:
    print(f"  FAIL symbols g1.sdk 包 import 失败: {type(e).__name__}: {e}")
    contrib_failed += len(sdk_expected_symbols)

# 3) 空伞 sanity: 顶层 g1/__init__.py 不应有任何 re-export
#    (重构纪律: 外部入口必须走 g1.sdk.* / g1.runtime.* 子路径)
try:
    g1_root = __import__("ghoshell_moss_contrib.unitree.g1", fromlist=["*"])
    forbidden_at_root = [
        "bootstrap", "state", "warrant",
        "G1StreamPlayerProvider", "build_g1_channel",
        "register_button_callback", "motion", "remote",
    ]
    leaked = [n for n in forbidden_at_root if hasattr(g1_root, n)]
    if leaked:
        print(f"  FAIL umbrella g1/__init__.py 泄漏了符号 (应为空伞): {leaked}")
        contrib_failed += 1
    else:
        print("  OK   umbrella g1/__init__.py 是空伞 (无 re-export)")
        contrib_passed += 1
except Exception as e:
    print(f"  FAIL umbrella g1 包 import 失败: {type(e).__name__}: {e}")
    contrib_failed += 1

print(f"\nContrib 段结果: {contrib_passed}/{contrib_passed+contrib_failed} passed, {contrib_failed} failed")


# ── 统一退出 ───────────────────────────────────────────────────────────────

total_failed = failed + contrib_failed
if total_failed > 0:
    print(f"\n总计 FAIL: {total_failed} 项. 上机部署前必须全部修复.")
    sys.exit(1)
else:
    print("\nALL OK — unitree_sdk2py + g1.sdk 子包全部可 import.")
