#!/usr/bin/env python3
"""
17_remote_keys_passthrough — 调试/AI 模式下遥控器按键和摇杆是否仍透传到 wireless_remote[40]

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本(给实机执行人/未来实例)
═══════════════════════════════════════════════════════════════════════════════

整个 channel 体系的范式决策是"遥控器在调试模式下被 G1 主板忽略, MOSS 把字节捡起来
当自己的输入设备". 这个反转的全部地基就在这一条:
  调试/AI 模式下, 推动摇杆 + 按下按键, wireless_remote[40] 字节内容是否仍然反映?

如果不反映(G1 主板把字节清零或固定值), 整个 warrant 的中断信号来源、state DAG 的授权键、
sensors 之外的人机协作通道, 全部塌掉, 设计要回炉.

如果反映, 我们立刻可以决定:
  - 哪些按键可以绑授权(state DAG 边)
  - 哪些按键可以绑 warrant scope 急停
  - 摇杆是否还能作为额外信号

═══════════════════════════════════════════════════════════════════════════════
执行人指引 — 你不需要动脑, 按步骤做即可
═══════════════════════════════════════════════════════════════════════════════

前置环境:
  1. G1 已开机, 正常 DDS 连接已就绪(本 session 前面的 03/04 应已通过)
  2. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate
  3. 周围空旷(本脚本不发任何运动指令, 但稳妥起见)

执行流程(脚本会一步步引导你, 每一步都会暂停等你做):
  阶段 1 — Sport 模式基线: 验证脚本本身的按键解析能跑通
  阶段 2 — 切到调试模式(L2+R2): 这是真正要测的状态
  阶段 3 — 按指引依次按每个按键, 看终端是否报告"按键变化"
  阶段 4 — 推每个摇杆方向, 看终端是否报告摇杆值变化
  阶段 5 — 切回 Damp 收尾

终端输出说明:
  每帧 LowState 到来时, 脚本会打印当前 fsm_mode + 16 按键的位 + 4 摇杆值.
  按键按下/松开时会高亮显示状态变化.
  你只需要照 prompt 操作, 然后口头(或截屏)反馈观察结果.

记录方法:
  脚本完成后会生成一个表格, 列出 "按键 X 在调试模式下是否报告变化".
  把这个表格反馈给模型实例, 模型会更新到 docs/sdk-topics.md + design 文档.

风险:
  本脚本不发任何运动指令. 唯一的"动作"是初始切到 Damp + 中间需要你按 L2+R2 进调试.
  L2+B 永远是硬件急停后备.
"""
import sys
import time
import struct
import threading
from pathlib import Path


# ── wireless_remote[40] 字节布局 (来自 04 脚本 + SDK 验证) ──
#
# bytes[0..1]    保留
# bytes[2..3]    按键 bitfield (data1, data2)
# bytes[4..7]    Lx (float32)
# bytes[8..11]   Rx (float32)
# bytes[12..15]  Ry (float32)
# bytes[16..19]  保留
# bytes[20..23]  Ly (float32)
# bytes[24..39]  保留

KEY_BITS_DATA1 = [
    ('R1',     0),
    ('L1',     1),
    ('Start',  2),
    ('Select', 3),
    ('R2',     4),
    ('L2',     5),
    ('F1',     6),
    ('F3',     7),
]

KEY_BITS_DATA2 = [
    ('A',     0),
    ('B',     1),
    ('X',     2),
    ('Y',     3),
    ('Up',    4),
    ('Right', 5),
    ('Down',  6),
    ('Left',  7),
]


class RemoteSnapshot:
    """单帧遥控器状态."""
    __slots__ = ('keys', 'lx', 'ly', 'rx', 'ry', 'tick')

    def __init__(self):
        self.keys = {name: 0 for name, _ in KEY_BITS_DATA1 + KEY_BITS_DATA2}
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.tick = 0

    @classmethod
    def parse(cls, data: bytes, tick: int = 0) -> 'RemoteSnapshot':
        snap = cls()
        snap.tick = tick
        data1, data2 = data[2], data[3]
        for name, bit in KEY_BITS_DATA1:
            snap.keys[name] = (data1 >> bit) & 1
        for name, bit in KEY_BITS_DATA2:
            snap.keys[name] = (data2 >> bit) & 1
        snap.lx = struct.unpack('<f', data[4:8])[0]
        snap.rx = struct.unpack('<f', data[8:12])[0]
        snap.ry = struct.unpack('<f', data[12:16])[0]
        snap.ly = struct.unpack('<f', data[20:24])[0]
        return snap

    def active_keys(self) -> list[str]:
        return [name for name, val in self.keys.items() if val]


# ── 结果记录: 每个按键是否在指定模式下被观察到按下 ──

class ObservationLog:
    def __init__(self):
        # mode_label -> key/axis -> observed_change: bool
        self.records: dict[str, dict[str, bool]] = {}

    def mode(self, label: str) -> dict[str, bool]:
        if label not in self.records:
            self.records[label] = {}
        return self.records[label]

    def mark(self, mode_label: str, name: str) -> None:
        self.mode(mode_label)[name] = True


# ── 监控线程: 实时打印帧 + 跟踪按键变化 ──

class MonitorThread:
    def __init__(self, subscriber, log: ObservationLog):
        self.sub = subscriber
        self.log = log
        self.running = False
        self.mode_label: str = ''
        self.fsm_mode: int = -1
        self.last_snapshot: RemoteSnapshot | None = None
        self._thread: threading.Thread | None = None
        self._verbose = True
        self._verbose_counter = 0

    def set_mode_label(self, label: str) -> None:
        self.mode_label = label
        print(f"\n>>> 当前记录模式标签: {label}\n")

    def silence(self) -> None:
        self._verbose = False

    def verbose(self) -> None:
        self._verbose = True
        self._verbose_counter = 0

    def _print_frame_periodic(self, snap: RemoteSnapshot) -> None:
        # 每 60 帧(约 0.12s @ 500Hz) 打一次基线状态, 避免刷屏
        self._verbose_counter += 1
        if self._verbose_counter % 60 != 0:
            return
        if not self._verbose:
            return
        ak = snap.active_keys()
        print(f"  [{snap.tick}] fsm={self.fsm_mode} "
              f"lx={snap.lx:+.3f} ly={snap.ly:+.3f} "
              f"rx={snap.rx:+.3f} ry={snap.ry:+.3f} "
              f"keys=[{','.join(ak) if ak else '-'}]")

    def _on_change(self, prev: RemoteSnapshot, curr: RemoteSnapshot) -> None:
        # 按键边沿
        for name in curr.keys:
            if prev.keys[name] != curr.keys[name]:
                edge = '↓ down' if curr.keys[name] else '↑ up'
                print(f"    *** key {name:<6} {edge}  fsm={self.fsm_mode} ***")
                if curr.keys[name] and self.mode_label:
                    self.log.mark(self.mode_label, name)

        # 摇杆: 任一轴绝对值 > 0.15 死区算"动了"
        AXIS_DEAD = 0.15
        for axis, prev_v, curr_v in [
            ('Lx', prev.lx, curr.lx),
            ('Ly', prev.ly, curr.ly),
            ('Rx', prev.rx, curr.rx),
            ('Ry', prev.ry, curr.ry),
        ]:
            if abs(curr_v) > AXIS_DEAD and abs(prev_v) <= AXIS_DEAD:
                print(f"    *** axis {axis:<3} active {curr_v:+.3f}  fsm={self.fsm_mode} ***")
                if self.mode_label:
                    self.log.mark(self.mode_label, axis)

    def start(self):
        self.running = True

        def _poll():
            while self.running:
                msg = self.sub.Read(timeout=500)
                if msg is None:
                    continue
                self.fsm_mode = getattr(msg, 'mode_machine', -1)
                snap = RemoteSnapshot.parse(bytes(msg.wireless_remote), msg.tick)
                if self.last_snapshot is not None:
                    self._on_change(self.last_snapshot, snap)
                self.last_snapshot = snap
                self._print_frame_periodic(snap)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2)


# ── 引导式步骤 ──

ALL_KEYS = [name for name, _ in KEY_BITS_DATA1 + KEY_BITS_DATA2]
ALL_AXES = ['Lx', 'Ly', 'Rx', 'Ry']

# L2+B 是硬件急停, 单独拎出来 — 任何模式都"应该"看得见
SAFETY_KEYS = ['L2', 'B']
# L2+R2 是进调试模式的组合键 — 在 Sport/Sit 模式下被消化, 调试模式下空闲
MODE_SWITCH_COMBO = ['L2', 'R2']


def prompt(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def run_keys_test(monitor: MonitorThread, mode_label: str) -> None:
    """引导执行人按每个按键, 摇杆, 让 monitor 记录."""
    monitor.set_mode_label(mode_label)

    print(f"\n=========== 阶段: 在 {mode_label} 下逐个测试 ===========")
    print("接下来会让你依次按每个按键、推每个摇杆方向.")
    print("每次操作后稍等 1 秒, 让监控记录到状态变化.\n")

    for key in ALL_KEYS:
        if key == 'B':
            # B 是 L2+B 急停的一半, 单独按 B(不按 L2) 不触发急停
            print(f"  [测试 {key}] 单独按一下 B 键(不要同时按 L2)")
        elif key == 'L2':
            print(f"  [测试 {key}] 单独按一下 L2 键(不要同时按 B 或 R2!)")
        elif key == 'R2':
            print(f"  [测试 {key}] 单独按一下 R2 键(不要同时按 L2!)")
        else:
            print(f"  [测试 {key}] 按一下 {key} 键")
        time.sleep(0.5)
        input("    按下并松开, 然后回车 >>> ")

    print("\n  -- 摇杆轴 --")
    for axis in ALL_AXES:
        stick, direction = axis[0], axis[1]
        side = {'L': '左', 'R': '右'}[stick]
        axis_dir = {'x': '左右', 'y': '前后'}[direction]
        print(f"  [测试 {axis}] 推 {side} 摇杆 {axis_dir} 方向(任一边都行)")
        time.sleep(0.5)
        input("    操作后回车 >>> ")


def print_observation_table(log: ObservationLog) -> None:
    print("\n" + "=" * 70)
    print("观测汇总表")
    print("=" * 70)

    modes = list(log.records.keys())
    if not modes:
        print("(无记录)")
        return

    header = f"{'按键/轴':<10} " + " ".join(f"{m:<14}" for m in modes)
    print(header)
    print("-" * len(header))

    for name in ALL_KEYS + ALL_AXES:
        row = f"{name:<10} "
        for m in modes:
            seen = log.mode(m).get(name, False)
            mark = "  ✓ 透传" if seen else "  ✗ 未观察"
            row += f"{mark:<14} "
        print(row)

    print()
    print("解读:")
    print("  ✓ 透传 = 该模式下推/按该输入时, wireless_remote 字节内容反映了它")
    print("  ✗ 未观察 = 没看到变化(可能 G1 主板清零, 也可能你没按到, 重测)")
    print()
    print("把这张表反馈给模型实例 — 模型会决定哪些键可以用作授权 / scope 急停.")


# ── 主流程 ──

def main():
    if len(sys.argv) < 2:
        print("用法: python 17_remote_keys_passthrough.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print("=" * 70)
    print("17_remote_keys_passthrough — 遥控器按键/摇杆透传验证")
    print("=" * 70)
    print()
    print("命题: 在调试/AI 模式下, 非 L2+B 的按键和摇杆是否仍透传到 LowState.")
    print()
    print("流程:")
    print("  阶段 1: Sport 模式基线 — 验证脚本本身能解析按键")
    print("  阶段 2: 切到调试模式(L2+R2) — 真正要测的状态")
    print("  阶段 3: 在调试模式下逐键 + 摇杆测试")
    print("  阶段 4: 切回 Damp 收尾")
    print()
    print("执行风格: 脚本一步步引导你, 不需要动脑.")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    # ── 初始化 ──
    print(f"\n初始化 DDS (domain=0, interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    print("OK: LowState 订阅就绪")

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("OK: LocoClient 就绪\n")

    log = ObservationLog()
    monitor = MonitorThread(sub, log)
    monitor.start()
    time.sleep(1)

    # ── 阶段 1: Sport 模式基线 ──
    print("\n" + "=" * 70)
    print("阶段 1: Sport 模式基线")
    print("=" * 70)
    print()
    print("先确认 G1 当前在 Sport 模式(站立运控). 看终端 fsm 应当 = 6.")
    print("如果不在 Sport, 你需要手动按遥控器进 Sport(通常 L2+A 或类似组合).")
    prompt("看终端报告 fsm_mode = 6 后, 回车继续(如果不是 6, 处理好再回车)")

    if monitor.fsm_mode != 6:
        print(f"!! 警告: 当前 fsm_mode = {monitor.fsm_mode}, 不是 Sport. 继续可能影响判断.")
        prompt("确认继续? (Ctrl+C 退出)")

    run_keys_test(monitor, mode_label="Sport(基线)")

    # ── 阶段 2: 切到调试模式 ──
    print("\n" + "=" * 70)
    print("阶段 2: 切到调试模式")
    print("=" * 70)
    print()
    print("步骤(在遥控器上人手操作):")
    print("  1. 长按 L2+A 切到 Damp(阻尼)")
    print("     -> 看终端 fsm_mode 变为 0")
    print("  2. 长按 L2+R2 切到调试模式")
    print("     -> 此后遥控器对机体的控制语义会被 G1 主板挂起")
    print("     -> 但 wireless_remote 字节理论上仍透传(本阶段要验证的)")
    print()
    print("调试模式下的 fsm_mode 值预期未知 — 可能仍是 0, 可能是别的. 记下来即可.")

    prompt("操作完毕 + 看终端 fsm_mode 变化后, 回车继续")
    print(f"  调试模式下 fsm_mode = {monitor.fsm_mode}")

    # ── 阶段 3: 调试模式逐键测试 ──
    print("\n" + "=" * 70)
    print("阶段 3: 调试模式下的按键 + 摇杆测试")
    print("=" * 70)
    print()
    print("这是核心阶段. 接下来每按一个键, 关注终端的 '*** key X ↓ down ***' 是否出现.")
    print("出现 = 透传成立. 不出现 = G1 主板把这个键消化了, 我们读不到.")
    print()

    run_keys_test(monitor, mode_label=f"调试(fsm={monitor.fsm_mode})")

    # ── 阶段 4: 收尾 ──
    print("\n" + "=" * 70)
    print("阶段 4: 收尾")
    print("=" * 70)
    print()
    print("按 L2+A 切回 Damp, 然后等待终端 fsm_mode = 0.")
    prompt("操作完毕回车")

    monitor.silence()
    time.sleep(1)
    monitor.stop()
    sub.Close()

    # ── 汇总 ──
    print_observation_table(log)

    print("\n下一步:")
    print("  把这张表完整截屏 / 复制给模型实例, 让模型:")
    print("    1. 更新 docs/sdk-topics.md 的遥控器条目")
    print("    2. 决定哪些键可以用作 state DAG 授权 / warrant scope 急停")
    print("    3. 更新 design/2026-06-28_channel_architecture.md 的'授权键分配'部分")
    print()
    print("特别关注:")
    print("  - 调试模式下 L2+B 是否仍触发硬件 Damp(应当, 这是安全底线)")
    print("  - 调试模式下哪些键完全静默 — 那些键肯定是 G1 主板消化掉的, 不可用")
    print("  - 摇杆是否仍报告值 — 决定 sensors 是否能用摇杆作为额外信号")


if __name__ == "__main__":
    main()
