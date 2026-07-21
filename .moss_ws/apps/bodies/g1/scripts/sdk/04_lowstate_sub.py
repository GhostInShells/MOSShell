#!/usr/bin/env python3
"""
订阅 G1 底层状态并解析遥控器按键。
验证: LowState_ 字段与文档一致，wireless_remote[40] 解析正确。

SDK 参考:
  example/g1/low_level/g1_low_level_example.py  — 使用 rt/lowstate (高频)
  example/wireless_controller/wireless_controller.py  — 遥控器解析逻辑
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_LowState_.py  — LowState_ 类型定义

订阅 canonical topic rt/lowstate (高频)。LowState_.motor_state 是 35 槽，G1 23-DoF
实际占用 0-28 (0-11 腿 / 12-14 腰 / 15-21 左臂 / 22-28 右臂)，29-34 为保留。

前置:
  G1 开机 + 以太网连接 PC2 + DDS 环境就绪
  source .venv/bin/activate
  python 00_import_verify.py  # 必须先通过

用法: python 04_lowstate_sub.py <networkInterface>
"""
import sys
import time
import struct


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
        return self.L2 == 1 and self.B == 1

    def summary(self):
        btns = []
        for name in ['A','B','X','Y','L1','L2','R1','R2','Up','Down','Left','Right','Select','Start','F1','F3']:
            if getattr(self, name):
                btns.append(name)
        return (f"Lx={self.Lx:+.3f} Ly={self.Ly:+.3f} "
                f"Rx={self.Rx:+.3f} Ry={self.Ry:+.3f} "
                f"buttons=[{','.join(btns) if btns else 'none'}]")


# G1 23-DoF 实际占用槽位 0-28；35 槽数组的 29-34 保留
MOTOR_NAMES = [
    "L_HipPitch","L_HipRoll","L_HipYaw","L_Knee","L_AnkleP","L_AnkleR",
    "R_HipPitch","R_HipRoll","R_HipYaw","R_Knee","R_AnkleP","R_AnkleR",
    "WaistYaw","WaistRoll","WaistPitch",
    "L_ShldPitch","L_ShldRoll","L_ShldYaw","L_Elbow","L_WristRoll","L_WristPitch","L_WristYaw",
    "R_ShldPitch","R_ShldRoll","R_ShldYaw","R_Elbow","R_WristRoll","R_WristPitch","R_WristYaw",
]


def print_lowstate(msg):
    remote = RemoteController()
    remote.parse(bytes(msg.wireless_remote))

    print(f"\n{'='*60}")
    print(f"tick={msg.tick}  mode_pr={msg.mode_pr}  mode_machine={msg.mode_machine}")
    print(f"遥控器: {remote.summary()}")
    if remote.is_estop():
        print("  *** L2+B 急停! ***")

    imu = msg.imu_state
    print(f"IMU rpy: [{imu.rpy[0]:+.3f}, {imu.rpy[1]:+.3f}, {imu.rpy[2]:+.3f}]  "
          f"gyro: [{imu.gyroscope[0]:+.3f}, {imu.gyroscope[1]:+.3f}, {imu.gyroscope[2]:+.3f}]")

    n_motors = len(msg.motor_state)
    print(f"电机 ({n_motors} 槽, G1 23-DoF 占 0-28):")
    for i in range(n_motors):
        ms = msg.motor_state[i]
        if i < len(MOTOR_NAMES):
            label = MOTOR_NAMES[i]
        elif i == n_motors - 1:
            label = "(reserved/weight?)"
        else:
            label = f"motor_{i}"
        tau = getattr(ms, 'tau_est', 0.0)
        print(f"  [{i:2d}] {label:<18s} q={ms.q:+7.3f}  dq={ms.dq:+7.3f}  tau={tau:+6.2f}  mode={ms.mode}")


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

    # canonical: g1 example 用的就是 rt/lowstate
    # 若需低频版可改 rt/lf/lowstate (前任 session 验证存在)
    topic = "rt/lowstate"
    sub = ChannelSubscriber(topic, LowState_)
    sub.Init()

    print(f"订阅 {topic} ...")
    print("请人类操作遥控器: 推动摇杆、按下各按键、L2+B (急停测试请到 12_estop_verify.py)\n")

    count = 0
    try:
        while count < 20:
            msg = sub.Read(timeout=3000)
            if msg is not None:
                count += 1
                print_lowstate(msg)
            else:
                print(f"超时: {topic} 未收到 LowState 数据。检查 G1 是否开机、DDS 是否正常 (ufw 分片?)。")
                break
    except KeyboardInterrupt:
        print("\n停止。")

    sub.Close()
    print(f"\n采集帧数: {count}")
    print("验证结论:")
    print("  [ ] wireless_remote 解析是否与遥控器实际按键一致？")
    print("  [ ] motor_state 0-28 的 q 值是否与肢体姿态吻合？29-34 是否为 0/无效？")
    print("  [ ] imu_state.rpy 数据是否合理？")

if __name__ == "__main__":
    main()