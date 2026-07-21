"""
_imu_sen_listen_and_drain — IMU 定时采样体验 + Enter drain.

场景:
  IMU 持续 2Hz 采样进 ring buffer. listener 后台打印 (throttled — 每 5 帧打印 1 帧,
  避免刷屏), 主线程按 Enter 触发 drain 看 batch 形态 (current + 历史 + 折叠表).

  使用者主动扰动 G1 机身: 轻推 / 让 G1 摆姿势 / 在 Sport 模式下原地小晃,
  观察姿态采样的历史窗口和数值变化.

  这是 channel 真实使用 scenario 的最小模拟:
    "周期性 drain 拿历史轨迹喂进 context_messages, 让模型理解'刚才发生了什么'."

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._imu_sen_listen_and_drain <nic>

前置:
  - G1 已开机
  - 任何模式都可以. Damp 下手推扰动最直观 (机身可被推动)
  - 在 Sport 模式下做小动作或保持静止都能看到效果

预期:
  [imu#1 sampled] roll=+0.3° pitch=-1.2° yaw=+87.5° |a|=9.79 |ω|=0.012
  [imu#6 sampled] roll=+0.5° pitch=-1.0° yaw=+87.5° |a|=9.81 |ω|=0.008
  ... (every 5th 才打印)
  >>> press Enter to drain >>>
  [drain] samples=10 window=5.0s rate=2.0Hz
    current: roll=+0.3° pitch=-1.2° yaw=+87.5° |accel|=9.79 |gyro|=0.012
    recent samples (yaw≈+87.5° throughout):
      t       roll     pitch    |accel|  |gyro|
       -5.0s  +0.3°    -1.2°    9.79     0.012
       ...
  Ctrl+C 退出.

实测样本 (2026-07-01, Damp 模式, G1 被推拉后静止):
  [drain] samples=10 window=15.4s rate=2.0Hz
    current: roll=-0.7° pitch=-6.6° yaw=-6.4° |accel|=9.84 |gyro|=0.006
    recent samples:
      t       roll     pitch    yaw      |accel|  |gyro|
       -4.9s    -1.3°    -6.1°   -10.6°   9.94    0.091
       -4.4s    -0.9°    -6.5°    -7.4°   9.91    0.126
       -3.9s    -0.7°    -6.8°    -6.3°  10.00    0.027
       -3.4s    -0.7°    -6.6°    -6.3°   9.96    0.014
       ...
  静止后 roll/pitch/yaw 收敛到稳定值, |gyro| < 0.01.
  Damp 模式的初始姿态: roll≈-0.7°, pitch≈-6.6°, yaw 随置放位置变化.
"""
from __future__ import annotations

import sys

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import imu
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


_sample_count = 0
_print_every = 5  # 2Hz × 5 = 每 2.5s 打印一行, 不刷屏


def _on_sample(s: imu.ImuSample) -> None:
    """每帧 IMU 触发. 跑在 sampler 线程. throttle 打印."""
    global _sample_count
    _sample_count += 1
    if _sample_count % _print_every != 0:
        return

    import math
    roll_d = s.roll_rad * 180.0 / math.pi
    pitch_d = s.pitch_rad * 180.0 / math.pi
    yaw_d = s.yaw_rad * 180.0 / math.pi
    gx, gy, gz = s.gyro_xyz
    ax, ay, az = s.accel_xyz
    gmag = math.sqrt(gx * gx + gy * gy + gz * gz)
    amag = math.sqrt(ax * ax + ay * ay + az * az)
    print(
        f"[imu#{_sample_count} sampled] "
        f"roll={roll_d:+.1f}° pitch={pitch_d:+.1f}° yaw={yaw_d:+.1f}° "
        f"|a|={amag:.2f} |ω|={gmag:.3f}"
    )


def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/3] imu.start(sample_rate_hz=2.0, buffer_size=10) ...")
    imu.start(sample_rate_hz=2.0, buffer_size=10)
    handle = imu.register_listener(_on_sample)
    print(f"      listener handle = {handle}  (每 {_print_every} 帧打印 1 帧)")

    print()
    print("=" * 64)
    print(" 扰动 G1 看 IMU 变化:")
    print("   - Damp 模式: 轻推机身, 看 roll/pitch 变化")
    print("   - Sport 模式: 让 G1 原地小动作或保持静止")
    print(" 按 Enter   → drain 当前 buffer (含 helper 表格化输出)")
    print(" Ctrl+C    → 干净退出")
    print("=" * 64)
    print()

    session: PromptSession = PromptSession()
    drain_count = 0
    try:
        with patch_stdout.patch_stdout(raw=True):
            while True:
                try:
                    session.prompt(">>> press Enter to drain >>> ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                drain_count += 1
                batch = imu.drain()
                print(f"[drain] samples={len(batch.samples)} "
                      f"window={batch.window_seconds:.1f}s "
                      f"rate={batch.sample_rate_hz:.1f}Hz")
                print(imu.batch_to_xml(batch))
                print(f"[health] {imu.health()}\n")
    finally:
        print(f"\n[3/3] imu.stop() ...")
        imu.unregister_listener(handle)
        imu.stop()
        print()
        print("=" * 64)
        print(f" 摘要: sampler 触发 {_sample_count} 次, drain {drain_count} 次.")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
