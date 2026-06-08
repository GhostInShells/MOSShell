#!/usr/bin/env python3
"""
订阅 G1 底层状态并解析遥控器按键。
验证: LowState_ 字段与文档一致，wireless_remote[40] 解析正确。

SDK 参考:
  example/wireless_controller/wireless_controller.py  — 遥控器解析逻辑
  unitree_sdk2py/idl/unitree_hg/msg/dds_.py           — LowState_ 类型定义
  src/unitree_sdk2_python/

前置:
  G1 开机 + 以太网连接 PC2 + DDS 环境就绪
  source .venv/bin/activate
  python 00_import_verify.py  # 必须先通过

用法: python 04_lowstate_sub.py <networkInterface>
"""
import sys
import time
import struct
from collections import OrderedDict

# ── 遥控器解析 (来自 example/wireless_controller/wireless_controller.py) ──

class RemoteController:
    """解析 LowState_.wireless_remote[40] 字节"""
    def __init__(self):
        self.Lx = self.Rx = self.Ry = self.Ly = 0.0
        self.L1 = self.L2 = self.R1 = self.R2 = 0
        self.A = self.B = self.X = self.Y = 0
        self.Up = self.Down = self.Left = self.Right = 0
        self.Select = self.F1 = self.F3 = self.Start = 0

    def parse_buttons(self, data1, data2):
        self.R1     = (data1 >> 0) & 1
        self.L1     = (data1 >> 1) & 1
        self.Start  = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2     = (data1 >> 4) & 1
        self.L2     = (data1 >> 5) & 1
        self.F1     = (data1 >> 6) & 1
        self.F3     = (data1 >> 7) & 1
        self.A      = (data2 >> 0) & 1
        self.B      = (data2 >> 1) & 1
        self.X      = (data2 >> 2) & 1
        self.Y      = (data2 >> 3) & 1
        self.Up     = (data2 >> 4) & 1
        self.Right  = (data2 >> 5) & 1
        self.Down   = (data2 >> 6) & 1
        self.Left   = (data2 >> 7) & 1

    def parse_keys(self, data):
        self.Lx = struct.unpack('<f', data[4:8])[0]
        self.Rx = struct.unpack('<f', data[8:12])[0]
        self.Ry = struct.unpack('<f', data[12:16])[0]
        self.Ly = struct.unpack('<f', data[20:24])[0]

    def parse(self, remote_data):
        self.parse_keys(remote_data)
        self.parse_buttons(remote_data[2], remote_data[3])

    def is_estop(self):
        """L2+B 同时按下 = 急停"""
        return self.L2 == 1 and self.B == 1

    def summary(self):
        btns = []
        for name in ['A','B','X','Y','L1','L2','R1','R2','Up','Down','Left','Right','Select','Start','F1','F3']:
            if getattr(self, name):
                btns.append(name)
        return (f"Lx={self.Lx:+.3f} Ly={self.Ly:+.3f} "
                f"Rx={self.Rx:+.3f} Ry={self.Ry:+.3f} "
                f"buttons=[{','.join(btns) if btns else 'none'}]")

# ── LowState 订阅 ──

MOTOR_NAMES = [
    "L_HipPitch","L_HipRoll","L_HipYaw","L_Knee","L_AnkleP","L_AnkleR",
    "R_HipPitch","R_HipRoll","R_HipYaw","R_Knee","R_AnkleP","R_AnkleR",
    "WaistYaw","WaistRoll","WaistPitch",
    "L_ShldPitch","L_ShldRoll","L_ShldYaw","L_Elbow","L_WristRoll","L_WristPitch","L_WristYaw",
    "R_ShldPitch","R_ShldRoll","R_ShldYaw","R_Elbow","R_WristRoll","R_WristPitch","R_WristYaw",
    "Weight(kNotUsed)"
]

def print_lowstate(msg):
    remote = RemoteController()
    remote.parse(bytes(msg.wireless_remote))

    print(f"\n{'='*60}")
    print(f"tick={msg.tick}  mode_pr={msg.mode_pr}  mode_machine={msg.mode_machine}")
    print(f"遥控器: {remote.summary()}")
    if remote.is_estop():
        print("  *** L2+B 急停! ***")

    # IMU 摘要
    imu = msg.imu_state
    print(f"IMU rpy: [{imu.rpy[0]:.2f}, {imu.rpy[1]:.2f}, {imu.rpy[2]:.2f}]")

    # 电机摘要 (名称: 位置)
    print("电机 (name: q  mode):")
    for i in range(min(29, len(msg.motor_state))):
        ms = msg.motor_state[i]
        if i < len(MOTOR_NAMES):
            print(f"  [{i:2d}] {MOTOR_NAMES[i]:<16s} q={ms.q:+7.3f}  dq={ms.dq:+7.3f}  mode={ms.mode}", end="")
        else:
            print(f"  [{i:2d}] motor_{i:<13s} q={ms.q:+7.3f}  dq={ms.dq:+7.3f}  mode={ms.mode}", end="")
        if hasattr(ms, 'tau_est'):
            print(f"  tau_est={ms.tau_est:+.2f}")
        else:
            print()

def main():
    if len(sys.argv) < 2:
        print("用法: python 04_lowstate_sub.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    # 使用低频 topic 减少带宽
    sub = ChannelSubscriber("rt/lf/lowstate", LowState_)
    sub.Init()

    print("订阅 rt/lf/lowstate ... (Ctrl+C 停止)")
    print("请人类操作遥控器: 推动摇杆、按下各按键、L2+B\n")
    print("电机索引 0-11=腿 12-14=腰 15-21=左臂 22-28=右臂 29=weight")

    count = 0
    try:
        while True:
            msg = sub.Read(timeout=5000)
            if msg is not None:
                count += 1
                print_lowstate(msg)
                if count >= 20:
                    print("\n已采集 20 帧。完成。")
                    break
            else:
                print("超时: 未收到 LowState 数据。检查 G1 是否开机、DDS 是否正常。")
                break
    except KeyboardInterrupt:
        print("\n停止。")

    sub.Close()
    print("验证结论:")
    print("  [ ] wireless_remote 解析是否与遥控器实际按键一致？")
    print("  [ ] motor_state 结构和限位是否与文档一致？")
    print("  [ ] imu_state 数据是否合理？")

if __name__ == "__main__":
    main()
