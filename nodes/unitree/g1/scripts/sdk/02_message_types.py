#!/usr/bin/env python3
"""
反射: DDS 消息类型的完整字段结构
决策: 确认 docs/sdk-topics.md 中的类型路径正确，字段名和类型与文档一致

SDK 路径: src/unitree_sdk2_python/
对应文件:
  unitree_sdk2py/idl/unitree_hg/msg/dds_.py   — G1/H1-2 消息类型 (LowState_, LowCmd_ 等)
  unitree_sdk2py/idl/unitree_go/msg/dds_.py   — Go2 共享类型 (SportModeState_, IMUState_)
  unitree_sdk2py/idl/default.py               — 默认导出

前置:
  source .venv/bin/activate
  python 00_import_verify.py  # 必须先通过
"""
import sys

print("=== DDS 消息类型字段反射 ===\n")
print("对照: docs/sdk-topics.md 类型验证清单\n")

# G1/H1-2 类型 — unitree_sdk2py/idl/unitree_hg/msg/dds_.py
try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    print("## LowState_  (rt/lowstate, rt/lf/lowstate)")
    print("   路径: unitree_sdk2py/idl/unitree_hg/msg/dds_.py")
    for f in dir(LowState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"FAIL: LowState_ — {e}\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    print("## LowCmd_  (rt/lowcmd, rt/arm_sdk)")
    print("   路径: unitree_sdk2py/idl/unitree_hg/msg/dds_.py")
    for f in dir(LowCmd_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"FAIL: LowCmd_ — {e}\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_
    print("## MotorCmd_  (LowCmd_.motor_cmd[])")
    print("   路径: unitree_sdk2py/idl/unitree_hg/msg/dds_.py (可能为嵌套类型)")
    for f in dir(MotorCmd_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except Exception as e:
    print(f"INFO: MotorCmd_ 可能是嵌套类型: {e} — 尝试从 LowCmd_ 实例访问...\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorState_
    print("## MotorState_  (LowState_.motor_state[])")
    for f in dir(MotorState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except Exception as e:
    print(f"INFO: MotorState_ 可能是嵌套类型: {e}\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_
    print("## IMUState_ (hg)  (LowState_.imu_state, rt/secondary_imu)")
    print("   路径: unitree_sdk2py/idl/unitree_hg/msg/dds_.py")
    for f in dir(IMUState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"FAIL: IMUState_ (hg) — {e}\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
    print("## BmsState_  (rt/lf/bmsstate)")
    for f in dir(BmsState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"WARN: BmsState_ — {e} (可能仅在 default 导出中)\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MainBoardState_
    print("## MainBoardState_  (rt/lf/mainboardstate)")
    for f in dir(MainBoardState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"WARN: MainBoardState_ — {e} (可能仅在 default 导出中)\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    print("## HandState_  (rt/dex3/*/state)")
    for f in dir(HandState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"WARN: HandState_ — {e}\n")

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
    print("## HandCmd_  (rt/dex3/*/cmd)")
    for f in dir(HandCmd_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"WARN: HandCmd_ — {e}\n")

# Go2 共享类型 — unitree_sdk2py/idl/unitree_go/msg/dds_.py
try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    print("## SportModeState_  (rt/sportmodestate)")
    print("   路径: unitree_sdk2py/idl/unitree_go/msg/dds_.py")
    for f in dir(SportModeState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"WARN: SportModeState_ — {e}\n")

try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_ as GoIMUState_
    print("## IMUState_ (go2)  (rt/odommodestate)")
    print("   路径: unitree_sdk2py/idl/unitree_go/msg/dds_.py")
    for f in dir(GoIMUState_):
        if not f.startswith('_'):
            print(f"  .{f}")
    print()
except ImportError as e:
    print(f"INFO: IMUState_ (go2) — {e}\n")

print("=== 对照 docs/sdk-topics.md ===")
print("""
验证清单:
  [ ] LowState_     — hg, rt/lowstate
  [ ] LowCmd_       — hg, rt/lowcmd + rt/arm_sdk
  [ ] HandState_    — hg, rt/dex3/*/state
  [ ] HandCmd_      — hg, rt/dex3/*/cmd
  [ ] BmsState_     — hg, rt/lf/bmsstate
  [ ] MainBoardState_ — hg, rt/lf/mainboardstate
  [ ] IMUState_ (hg)  — hg, rt/secondary_imu
  [ ] IMUState_ (go2) — go2, rt/odommodestate
  [ ] SportModeState_ — go2, rt/sportmodestate
""")
