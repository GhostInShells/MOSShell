#!/usr/bin/env python3
"""
反射: 五个 RPC 客户端的完整 API surface
决策: 对比 docs/index.md 和 docs/sdk-topics.md 中的文档记录，确认 API 存在性和签名一致性

SDK 路径: src/unitree_sdk2_python/
对应文件:
  unitree_sdk2py/g1/loco/g1_loco_client.py     — LocoClient
  unitree_sdk2py/g1/arm/g1_arm_action_client.py — G1ArmActionClient + action_map
  unitree_sdk2py/g1/audio/g1_audio_client.py   — AudioClient
  unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py — MotionSwitcherClient
  unitree_sdk2py/b2/robot_state/robot_state_client.py — RobotStateClient

前置:
  source .venv/bin/activate
  python 00_import_verify.py  # 必须先通过
"""
import inspect
import sys

clients = {}

# LocoClient — unitree_sdk2py/g1/loco/g1_loco_client.py
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
clients["LocoClient"] = LocoClient
# G1ArmActionClient — unitree_sdk2py/g1/arm/g1_arm_action_client.py
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
clients["G1ArmActionClient (action_map)"] = ("action_map", action_map)
# AudioClient — unitree_sdk2py/g1/audio/g1_audio_client.py
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
clients["AudioClient"] = AudioClient
# MotionSwitcherClient — unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
clients["MotionSwitcherClient"] = MotionSwitcherClient
# RobotStateClient — unitree_sdk2py/b2/robot_state/robot_state_client.py
from unitree_sdk2py.b2.robot_state.robot_state_client import RobotStateClient
clients["RobotStateClient"] = RobotStateClient


print("=== RPC Client API Surface ===\n")

for name, obj in clients.items():
    if isinstance(obj, tuple):
        label, data = obj
        print(f"## {label}\n")
        print(f"  类型: {type(data).__name__}")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"    {k}: {v}")
        print()
        continue

    cls = obj
    print(f"## {name}\n")
    print(f"  基类: {[b.__name__ for b in cls.__mro__][:3]}")
    print()

    # 公开方法 (排除 _ 开头)
    public_methods = []
    for m_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if m_name.startswith('_'):
            continue
        try:
            sig = inspect.signature(method)
            public_methods.append((m_name, str(sig)))
        except (ValueError, TypeError):
            public_methods.append((m_name, "(无法获取签名)"))

    if public_methods:
        print("  公开方法:")
        for m_name, sig in public_methods:
            doc = inspect.getdoc(method) or ""
            doc_first = doc.split('\n')[0][:80] if doc else ""
            print(f"    {m_name}{sig}")
            if doc_first:
                print(f"        {doc_first}")
        print()

    # 检查 Init() 注册的 API
    try:
        instance = cls()
        if hasattr(instance, 'Init'):
            print(f"  Init() 方法存在")
        else:
            print(f"  警告: 无 Init() 方法")
    except Exception as e:
        print(f"  警告: 实例化失败: {e}")
    print()

print("=== 文档对照检查 ===")
print("""
对照 docs/index.md (阶段 A 文档摸底):
1. LocoClient — 是否有 Squat(), ContinuousGait(), GetFsmMode(), StandUp()?
   → 答案: 源码中无这些方法。Sit()=SetFsmId(3), Move(continuous=True)替代,
     SetFsmId()替代StandUp(), GetFsmId/GetFsmMode 需自行封装
2. G1ArmActionClient — action_map 键数是否与文档一致?
   → 文档描述 15 种，action_map 实际 17 种
3. AudioClient — ASR 方法是否存在?
   → API ID 1002 已注册但无封装方法 (与文档吻合)
""")
