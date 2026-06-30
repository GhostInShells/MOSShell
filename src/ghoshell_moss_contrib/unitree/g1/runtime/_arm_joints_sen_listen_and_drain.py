"""
_arm_joints_sen_listen_and_drain — 双臂关节定时采样体验 + Enter drain.

场景:
  arm_joints 2Hz 采样 10 关节进 ring buffer. listener 后台 throttle 打印
  "运动中关节列表" (静止时静默), 主线程 Enter drain 看 batch (current + 历史
  + 折叠静止关节表).

  使用者扰动方式 (按现场情况选一):
    a) Damp 模式 — 手动摆 G1 手臂, 看采样轨迹
    b) Sport 模式 — 让 G1 跑出厂动作 (ExecuteAction 11/clap 等), 看双臂关节
       变化历史
    c) 直接静置 — 看静止关节折叠 ("all 10 joints stationary")

  这是 channel 真实使用 scenario 的最小模拟:
    "arms keyframe 编辑前, channel 把最近 N 秒采样喂进 context, 让模型
     看到当前 pose 和'刚才动了哪些关节', 作为新动画的起点参考."

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._arm_joints_sen_listen_and_drain <nic>

前置:
  - G1 已开机, 任何模式
  - Damp 模式下手臂可被手推 (motors free); Sport 模式下手臂锁定 (motors engaged)
  - 不进调试模式

预期:
  (静止时静默 — listener 默认不打印)
  [arm_joints#12 moving] left_shoulder_pitch, left_elbow
  ...
  >>> press Enter to drain >>>
  [drain] samples=6 window=3.0s rate=2.0Hz
    current pose:
      left  shoulder_pitch=-0.12 shoulder_roll=+0.08 ...
      right shoulder_pitch=-0.12 shoulder_roll=-0.08 ...
      motors: 10/10 engaged
    recent samples (joints with delta > 0.05 rad shown; others stationary):
      t        left_shoulder_pitch     left_elbow
      -3.0s    -0.12                   -0.95
      ...
    note: rad zero & positive-direction NOT calibrated; use deltas, not absolutes.
"""
from __future__ import annotations

import sys

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import arm_joints
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


_sample_count = 0
_print_every = 4  # 仅作 throttle 用; listener 只在"运动中"才打印


def _on_sample(s: arm_joints.ArmJointsSample) -> None:
    """每帧 arm_joints 触发. 只在有关节"在动"时打印, 静止时静默."""
    global _sample_count
    _sample_count += 1

    moving = [name for name, v in s.dq.items() if abs(v) > 0.05]
    if not moving:
        return  # 静止时不打印, 避免刷屏
    if _sample_count % _print_every != 0:
        return  # 运动中也 throttle (避免动作期间太密)

    names_short = [n.replace("shoulder_", "sh_") for n in moving]
    print(f"[arm_joints#{_sample_count} moving] {', '.join(names_short)}")


def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)

    print("[2/3] arm_joints.start(sample_rate_hz=2.0, buffer_size=6) ...")
    arm_joints.start(sample_rate_hz=2.0, buffer_size=6)
    handle = arm_joints.register_listener(_on_sample)
    print(f"      listener handle = {handle}")
    print(f"      静止时 listener 静默, 仅在关节运动 (|dq| > 0.05 rad/s) 时打印.")

    print()
    print("=" * 64)
    print(" 扰动手臂看采样:")
    print("   a) Damp 模式: 手动摆 G1 双臂, 看采样轨迹")
    print("   b) Sport 模式: 让 G1 跑出厂动作, 看双臂动作历史")
    print("   c) 静置: 看 helper 折叠 'all 10 joints stationary'")
    print(" 按 Enter   → drain 当前 buffer (current + 折叠静止关节表)")
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
                batch = arm_joints.drain()
                print(f"[drain] samples={len(batch.samples)} "
                      f"window={batch.window_seconds:.1f}s "
                      f"rate={batch.sample_rate_hz:.1f}Hz")
                print(arm_joints.batch_to_xml(batch))
                print(f"[health] {arm_joints.health()}\n")
    finally:
        print(f"\n[3/3] arm_joints.stop() ...")
        arm_joints.unregister_listener(handle)
        arm_joints.stop()
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
