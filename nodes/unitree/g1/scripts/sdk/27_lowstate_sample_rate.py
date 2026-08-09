#!/usr/bin/env python3
"""
27_lowstate_sample_rate — LowState 真实到达频率 + 处理上限测量

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

monitor 用 cyclonedds callback 写 state.py — 每帧 LowState 触发回调.
G1 标称 500Hz 高频包. 实际到达频率是多少? 我们用 queueLen=1 + 每帧构造
frozen dataclass 时, 单线程能处理多少 Hz 不掉帧?

这两个数决定:
  - sensors.joints / sensors.trajectory 的采样率上限
  - state.py monitor 的负载 — 如果接近 100% 单核, 需要节流
  - 急停 callback 的响应延迟基线

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + 任何模式 (这测的是 LowState 频率, 跟动作无关)
  2. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

测试流程:
  阶段 1: 30s 高频接收测量 — 看实际到达 Hz
  阶段 2: 30s 模拟"每帧做工作量" — 在 handler 里构造完整 frozen dataclass,
          看处理 Hz 是否仍跟得上
  阶段 3: 30s 模拟"重 handler" — 加 sleep(2ms), 看是否开始丢帧

每阶段汇总: 总收到帧数 / 平均 Hz / 时间间隔分布 (min/max/p95).

风险:
  无运动, 无副作用. 纯订阅测量.
"""
import sys
import time
import threading
import statistics
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _JointSnapshot:
    """模拟 monitor 在 handler 里构造的 frozen dataclass."""
    q: float
    dq: float
    tau: float
    mode: int


def measure(subscriber, duration: float, handler_work: str = "none") -> dict:
    """订阅 duration 秒, 在 handler 中按 handler_work 类型做工作量. 返回统计."""
    arrival_times: list[float] = []
    count = 0
    snapshots_constructed = 0

    def _handler(msg):
        nonlocal count, snapshots_constructed
        t = time.monotonic()
        arrival_times.append(t)
        count += 1

        if handler_work == "construct":
            # 模拟 state.py 真实工作: 构造 35 个 JointSnapshot
            try:
                snaps = tuple(
                    _JointSnapshot(
                        q=getattr(m, 'q', 0.0),
                        dq=getattr(m, 'dq', 0.0),
                        tau=getattr(m, 'tau_est', 0.0),
                        mode=getattr(m, 'mode', 0),
                    )
                    for m in msg.motor_state
                )
                snapshots_constructed += len(snaps)
            except Exception:
                pass
        elif handler_work == "heavy":
            # 重 handler — 加 2ms sleep
            time.sleep(0.002)

    # 重订阅(因为前一次可能仍有 callback 在跑)
    subscriber.Close()
    subscriber.Init(_handler, 1)

    t_start = time.monotonic()
    time.sleep(duration)
    t_end = time.monotonic()

    actual_dur = t_end - t_start
    hz = count / actual_dur if actual_dur > 0 else 0

    # 帧间隔统计
    intervals_ms = []
    for i in range(1, len(arrival_times)):
        intervals_ms.append((arrival_times[i] - arrival_times[i-1]) * 1000)

    stats = {
        'duration': actual_dur,
        'count': count,
        'hz': hz,
        'snapshots_constructed': snapshots_constructed,
    }
    if intervals_ms:
        stats['interval_min_ms'] = min(intervals_ms)
        stats['interval_max_ms'] = max(intervals_ms)
        stats['interval_mean_ms'] = statistics.mean(intervals_ms)
        stats['interval_median_ms'] = statistics.median(intervals_ms)
        try:
            stats['interval_p95_ms'] = statistics.quantiles(intervals_ms, n=20)[18]
        except statistics.StatisticsError:
            stats['interval_p95_ms'] = max(intervals_ms)
    return stats


def print_stats(label: str, s: dict):
    print(f"\n  {label}:")
    print(f"    时长: {s['duration']:.2f}s   收到 {s['count']} 帧   {s['hz']:.1f} Hz")
    if s.get('snapshots_constructed', 0) > 0:
        print(f"    构造了 {s['snapshots_constructed']} 个 JointSnapshot")
    if 'interval_mean_ms' in s:
        print(f"    帧间隔 (ms): mean={s['interval_mean_ms']:.2f}  median={s['interval_median_ms']:.2f}")
        print(f"                 min={s['interval_min_ms']:.2f}  max={s['interval_max_ms']:.2f}  p95={s['interval_p95_ms']:.2f}")


def main():
    if len(sys.argv) < 2:
        print("用法: python 27_lowstate_sample_rate.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    print("=" * 70)
    print("27_lowstate_sample_rate — LowState 频率 + 处理上限")
    print("=" * 70)
    print()
    input("准备好了按 Enter 开始 (~2 分钟) >>> ")

    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    # 初次 Init 不需要 handler, measure 会重新 Init
    sub.Init()

    # ── 阶段 1: 空 handler 看真实到达频率 ──
    print("\n阶段 1: 空 handler 30s (基线真实频率)")
    s1 = measure(sub, duration=30.0, handler_work="none")
    print_stats("阶段 1 (空 handler)", s1)

    # ── 阶段 2: 模拟真实工作量 ──
    print("\n阶段 2: 模拟 monitor handler (构造 frozen dataclass) 30s")
    s2 = measure(sub, duration=30.0, handler_work="construct")
    print_stats("阶段 2 (真实工作量)", s2)

    # ── 阶段 3: 重 handler 看丢帧上限 ──
    print("\n阶段 3: 重 handler (每帧 +2ms sleep) 30s — 看丢帧上限")
    s3 = measure(sub, duration=30.0, handler_work="heavy")
    print_stats("阶段 3 (重 handler)", s3)

    sub.Close()

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("LowState 采样率汇总")
    print("=" * 70)
    print(f"  阶段 1 基线频率:        {s1['hz']:.1f} Hz")
    print(f"  阶段 2 真实工作下:      {s2['hz']:.1f} Hz  (差 {s1['hz'] - s2['hz']:+.1f})")
    print(f"  阶段 3 重 handler 下:   {s3['hz']:.1f} Hz  (差 {s1['hz'] - s3['hz']:+.1f})")
    print()
    print("反馈给模型:")
    print(f"  - state.py monitor 实际能跑多高 Hz (取阶段 2)")
    print(f"  - 是否需要节流: 阶段 2 < 阶段 1 显著 → 需要")
    print(f"  - sensors 关节采样建议: 阶段 2 / N 作为采样上限")
    print(f"  - 急停 callback 响应延迟基线: median interval / 2")
    print()
    print("实测记录:")
    print("  2026-06-29: LowState 真实 ~1052 Hz (非 500Hz).")
    print("    frozen dataclass 构造开销可忽略 (差 0.2 Hz).")
    print("    结论: _monitor.py 当前设计吃满 1kHz 没问题, 不需要节流.")


if __name__ == "__main__":
    main()
