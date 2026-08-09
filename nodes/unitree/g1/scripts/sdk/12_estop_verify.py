#!/usr/bin/env python3
"""
急停验证: 踏步中遥控器 L2+B → 软件立即 Damp()。
核心命题: 遥控器急停在 MOSS 控制链路上的感知延迟和响应效果。

SDK 参考:
  example/wireless_controller/wireless_controller.py  — 遥控器解析
  example/g1/high_level/g1_loco_client_example.py      — LocoClient 调用模式
  unitree_sdk2py/g1/loco/g1_loco_client.py              — LocoClient.Damp() = SetFsmId(1)
  unitree_sdk2py/idl/unitree_hg/msg/dds_.py             — LowState_.wireless_remote[40]
  src/unitree_sdk2_python/

安全设计:
  硬件层: L2+B 不可绕过 — G1 的 FSM 直接进入阻尼。这是真正的安全底线。
  软件层: MOSS 从 wireless_remote 感知急停 → 立即 Damp() — 体验层，不是安全底线。
  不走 ZeroTorque (危险 — 电机会失能瘫倒)。

前置:
  G1 开机 + 非调试模式 + Start(站立运控) + 周围清空 + 遥控器就绪
  source .venv/bin/activate
  python 00_import_verify.py

用法: python 12_estop_verify.py <networkInterface>
"""
import sys
import time
import struct
import threading
from pathlib import Path

# ── 急停状态: 文件标记 + 内存标记 ──
# 使用临时文件作为跨脚本可见的急停信号
# 只要文件存在 → 任何 command 脚本应拒绝发送新指令
ESTOP_FLAG_FILE = Path("/tmp/g1_moss_estop")

class EstopState:
    """急停状态管理 — 双路标记 (文件 + 内存)"""
    def __init__(self):
        self.active = False
        self.trigger_time = 0.0
        self.trigger_source = ""

    def set(self, source: str):
        if not self.active:
            self.active = True
            self.trigger_time = time.time()
            self.trigger_source = source
            ESTOP_FLAG_FILE.touch()
            print(f"\n*** 急停触发! source={source} time={self.trigger_time:.3f} ***\n")

    def clear(self):
        self.active = False
        if ESTOP_FLAG_FILE.exists():
            ESTOP_FLAG_FILE.unlink()

# ── 遥控器解析 ──

class RemoteMonitor:
    """后台线程: 订阅 LowState 并检测 L2+B"""

    def __init__(self, estop: EstopState):
        self.estop = estop
        self.latest_l2 = 0
        self.latest_b = 0
        self.running = False
        self._thread = None

    def parse_wireless(self, data):
        """解析 wireless_remote[40] 字节数组"""
        l2 = (data[2] >> 5) & 1
        b  = (data[3] >> 1) & 1
        self.latest_l2 = l2
        self.latest_b = b
        return l2, b

    def _handler(self, msg):
        l2, b = self.parse_wireless(bytes(msg.wireless_remote))
        if l2 and b and not self.estop.active:
            self.estop.set("L2+B")

    def start(self, subscriber):
        """启动后台监控"""
        self.running = True

        def _poll():
            print("[Monitor] 后台急停监控已启动 — 等待 L2+B...")
            while self.running and not self.estop.active:
                sample = subscriber.Read(timeout=500)
                if sample is not None:
                    self._handler(sample)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

# ── 主流程 ──

def main():
    if len(sys.argv) < 2:
        print("用法: python 12_estop_verify.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    # 清理上次残留的急停标记
    estop = EstopState()
    estop.clear()

    print("=" * 60)
    print("G1 急停验证 — 踏步中遥控器 L2+B")
    print()
    print("验证命题:")
    print("  1. MOSS 能否在 300ms 内感知到 L2+B？")
    print("  2. Damp() 能否在踏步运动中立即生效？")
    print("  3. 硬件 L2+B 先发生还是软件 Damp() 先响应？")
    print()
    print("安全:")
    print("  遥控器 L2+B 是硬件底线 — 无论脚本做什么都会进入阻尼")
    print("  本脚本使用极慢速踏步 (Move 0.05 m/s continuous)")
    print("  不走 ZeroTorque。任何时候可松开 L2+B")
    print("=" * 60)
    input("按 Enter 继续...")

    # ── 初始化 ──
    print(f"\n初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    # LowState 订阅 — 用高频 rt/lowstate (g1 example canonical)
    # 急停延迟测量必须用高频版，低频版会拉高 L2+B 检测时间
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    print("OK: LowState 订阅就绪 (rt/lowstate 高频)")

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("OK: LocoClient 就绪\n")

    # ── 启动急停监控 ──
    monitor = RemoteMonitor(estop)
    monitor.start(sub)

    time.sleep(1)

    # ── 确认当前模式 ──
    print("进入 Start (站立运控) 模式...")
    code = loco.Start()
    if code != 0:
        print(f"FAIL: Start code={code}")
        monitor.stop()
        return
    print("OK: Start 完成 — G1 已进入站立运控")

    time.sleep(3)

    # ── 测试: 踏步 + L2+B 急停 ──
    print("\n" + "=" * 40)
    print("测试: 极慢速原地踏步")
    print("  将在 3 秒后启动连续低速 Move")
    print("  人类在任何一个时刻按下遥控器 L2+B")
    print("  脚本检测到后立即 Damp()")
    print("=" * 40)

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n启动连续移动 (vx=0.05 m/s)...")
    loco.Move(0.05, 0, 0, True)
    t_start = time.time()

    # 等待急停触发 (最多 30 秒)
    while not estop.active and (time.time() - t_start) < 30:
        elapsed = time.time() - t_start
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            # 每 5 秒提示
            print(f"  踏步中... ({int(elapsed)}s) L2: {monitor.latest_l2} B: {monitor.latest_b}")
            time.sleep(1)

    if estop.active:
        # 急停响应
        t_detect = estop.trigger_time - t_start
        print(f"\n急停检测延迟: {t_detect:.3f}s  (从踏步开始算)")
        print("发送 Damp()...")
        t_before_damp = time.time()
        code = loco.Damp()
        t_after_damp = time.time()
        damp_latency = t_after_damp - t_before_damp
        print(f"Damp() 调用延迟: {damp_latency*1000:.0f}ms")

        print("\n请人类判断:")
        print("  - G1 是否进入了阻尼状态？(LED 橙色)")
        print("  - 踏步是否立即停止？")
        print("  - 是先感觉到硬急停还是软件 Damp？")

        monitor.stop()
        time.sleep(2)
    else:
        print("\n超时 — 未检测到 L2+B。停止移动。")
        loco.StopMove()
        monitor.stop()

    # 清理
    estop.clear()

    print("\n验证结论:")
    print("  [ ] wireless_remote 后台监控是否持续运行？")
    print("  [ ] L2+B 检测延迟是否 < 500ms？")
    print("  [ ] Damp() 调用是否立即生效？")
    print("  [ ] 硬件急停 (L2+B) 和软件急停 (Damp) 哪个先到？")
    print(f"\n急停标记文件: {ESTOP_FLAG_FILE}")
    print("  存在 = 急停中。后续脚本应先检查此文件再发任何运动指令。")

if __name__ == "__main__":
    main()
