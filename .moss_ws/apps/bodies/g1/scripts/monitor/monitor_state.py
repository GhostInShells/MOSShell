#!/usr/bin/env python3
"""
monitor_state — 监控 G1 FSM 状态变更

订阅 rt/sportmodestate, 实时打印 fsm_id / fsm_mode 的变化.
Read() 用 queue 做 Python 层超时保护 (cyclonedds Read 在无 matched publisher
时可能永久阻塞, timeout 参数不生效).

用法:
  python monitor_state.py <networkInterface>
  python monitor_state.py eth0
"""

import sys
import time
import threading
import queue


def main():
    if len(sys.argv) < 2:
        print("用法: python monitor_state.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    print(f"初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init()
    print("订阅 rt/sportmodestate 就绪")
    print()

    _last_id = -1
    _last_mode = -1
    _msg_queue: queue.Queue = queue.Queue(maxsize=1)
    running = True

    def _reader():
        """跑在独立线程, Read() 可能永久阻塞, 不怕."""
        while running:
            msg = sub.Read(timeout=500)
            if not running:
                break
            if msg is not None:
                # 丢旧帧取最新
                try:
                    while True:
                        _msg_queue.get_nowait()
                except queue.Empty:
                    pass
                _msg_queue.put(msg)

    _thread = threading.Thread(target=_reader, daemon=True)
    _thread.start()

    print("等待首帧...")
    print("按 Ctrl+C 退出")
    print()

    try:
        while True:
            try:
                msg = _msg_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            fsm_id = msg.fsm_id
            fsm_mode = msg.fsm_mode

            if fsm_id != _last_id or fsm_mode != _last_mode:
                mode_str = "静态" if fsm_mode == 0 else ("动态" if fsm_mode == 1 else f"未知({fsm_mode})")
                arrow = "→" if _last_id >= 0 else "="
                prev_str = str(_last_id) if _last_id >= 0 else "?"
                ts = time.strftime('%H:%M:%S')
                print(f"  *** [{ts}] fsm_id {prev_str} {arrow} {fsm_id}  "
                      f"fsm_mode={fsm_mode}({mode_str}) ***")
                _last_id = fsm_id
                _last_mode = fsm_mode

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        running = False
        _thread.join(timeout=2)
        sub.Close()


if __name__ == "__main__":
    main()
