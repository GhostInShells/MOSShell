#!/usr/bin/env python3
"""
订阅 G1 运动模式状态。
验证: SportModeState_ 字段与文档一致，fsm_id/fsm_mode 与 FSM 模式 ID 表吻合。

SDK 参考:
  example/g1/high_level/g1_loco_client_example.py  — SportModeState 订阅模式
  unitree_sdk2py/idl/unitree_go/msg/dds_.py         — SportModeState_ 类型定义
  src/unitree_sdk2_python/

前置: 同 04_lowstate_sub.py
用法: python 05_sportmode_sub.py <networkInterface>
"""
import sys
import time

# FSM 模式 ID 对照表 (来自 docs/index.md)
FSM_TABLE = {
    0:   "零力矩",
    1:   "阻尼",
    2:   "位控下蹲",
    3:   "位控落座",
    4:   "锁定站立",
    500: "常规运控",
    501: "常规运控-3Dof-waist",
    702: "躺起",
    706: "平衡下蹲/蹲起",
    801: "走跑运控",
    802: "走跑运控",
}

FSM_MODE_TABLE = {
    0: "静态(可切换模式)",
    1: "动态(不可切换，仅可切到阻尼)",
}

def print_sportmode(msg):
    fsm_name = FSM_TABLE.get(msg.fsm_id, f"未知({msg.fsm_id})")
    mode_name = FSM_MODE_TABLE.get(msg.fsm_mode, f"未知({msg.fsm_mode})")
    print(f"fsm_id={msg.fsm_id} ({fsm_name})  "
          f"fsm_mode={msg.fsm_mode} ({mode_name})  "
          f"task_id={msg.task_id}  "
          f"task_time={msg.task_time:.2f}s")

def main():
    if len(sys.argv) < 2:
        print("用法: python 05_sportmode_sub.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)
    print("OK\n")

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init()

    print("订阅 rt/sportmodestate ... (10 帧后自动停止)")
    print("请人类在 G1 的运控模式间切换 (Damp → Sit → Start)\n")

    count = 0
    last_fsm = None
    try:
        while count < 10:
            msg = sub.Read(timeout=5000)
            if msg is not None:
                if msg.fsm_id != last_fsm:
                    print_sportmode(msg)
                    if msg.fsm_id != last_fsm and last_fsm is not None:
                        print("  ^ 模式变更!")
                    last_fsm = msg.fsm_id
                    count += 1
            else:
                print("超时: 未收到 SportModeState。")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n停止。")

    sub.Close()
    print("\n验证结论:")
    print("  [ ] fsm_id 是否与实际运控模式一致？")
    print("  [ ] 模式切换时 fsm_mode 是否从 1(动态) 变为 0(静态)？")
    print("  [ ] fsm_id 值与 docs/index.md 的 FSM 表是否吻合？")

if __name__ == "__main__":
    main()
