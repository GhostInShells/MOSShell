#!/usr/bin/env python3
"""
SportModeState 发布探测 — 验证 G1 是否在 rt/sportmodestate 上发布数据。

SDK 现实:
  SportModeState_ 是 Go2 类型 (unitree_go.msg.dds_)。G1 example
  g1_loco_client_example.py 虽然 import 了此类型但从未订阅 — 即官方代码本身
  并不依赖 G1 发布该 topic。前任 2026-06-15 session 实测"05 无数据"。
  本脚本用短超时确认/推翻这一观察，不阻塞整轮。

如果 G1 不发 SportModeState，FSM 状态需从其他途径获取:
  - LocoClient 的 GET_FSM_ID API 已 _RegistApi 但无 Python wrapper (需自行封装)
  - LowState.mode_pr / mode_machine 字段 (但语义未确认)
  - MotionSwitcherClient.CheckMode() (调试模式下的 motion mode, 非 FSM id)

用法: python 05_sportmode_sub.py <networkInterface>
"""
import sys
import time


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

    print("探测命题: G1 是否在 rt/sportmodestate 上发布 SportModeState_?")
    print("方法: 短超时连续 3 次尝试 (每次 2s)，全部超时则结论为'不发布'。\n")

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init()

    received = 0
    last_fsm = None
    try:
        for attempt in range(1, 4):
            print(f"[尝试 {attempt}/3] 等待 2s...")
            msg = sub.Read(timeout=2000)
            if msg is None:
                print(f"  超时")
                continue
            received += 1
            fsm_name = FSM_TABLE.get(getattr(msg, 'fsm_id', -1), f"未知({getattr(msg, 'fsm_id', '?')})")
            print(f"  收到! fsm_id={getattr(msg, 'fsm_id', '?')} ({fsm_name})  "
                  f"fsm_mode={getattr(msg, 'fsm_mode', '?')}  "
                  f"task_id={getattr(msg, 'task_id', '?')}")
            last_fsm = getattr(msg, 'fsm_id', None)
    except KeyboardInterrupt:
        print("\n中断。")

    sub.Close()
    print(f"\n{'='*50}")
    print(f"探测结果: 3 次尝试收到 {received} 帧")
    if received == 0:
        print("结论: G1 不在 rt/sportmodestate 发布 SportModeState_。")
        print("  下一步: FSM 状态需通过 LocoClient.GetFsmId() 自行封装 RPC，")
        print("  或解读 LowState.mode_pr/mode_machine 字段。")
    else:
        print("结论: G1 确实发布 SportModeState_。可作为 FSM 状态来源。")
        print("  建议: 后续脚本人类切换运控模式时再跑一次确认 fsm_id 变更。")


if __name__ == "__main__":
    main()