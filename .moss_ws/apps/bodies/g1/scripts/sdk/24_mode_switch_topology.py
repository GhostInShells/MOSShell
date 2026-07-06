#!/usr/bin/env python3
"""
24_mode_switch_topology — FSM 模式完整可达图实测 (遥控器路径)

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

20 脚本测了 Sit / Stand / Sport 路径中的关键边. 24 是它的补完 —
通过遥控器组合键触发模式切换, 用 rt/sportmodestate 读取真实的 fsm_id,
得到完整的 "遥控器按键 → FSM ID" 映射 和 各边可达性/耗时.

state DAG 设计的 "具体边定义" 依赖这张图.

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机, 悬挂在吊架上
  2. 周围有足够空间 (部分切换 G1 会站起/坐下/摆姿态)
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

方法:
  本脚本 **不使用 SetFsmId API (高危)**. 全部通过遥控器组合键切换.
  脚本引导你依次按各组合键, 然后订阅 rt/sportmodestate 读取真实的 fsm_id.

风险:
  状态切换有体姿变化. 任何不稳 L2+B 急停.
  **阻尼模式脱力, 必须有悬挂.** 不要在无悬挂状态下切阻尼.

遥控器组合键清单 (2026-06-29 实机确认):
  从零力矩:
    L2 + Y = 零力矩
    L2 + B = 阻尼
    L2 + 上 = 锁定站立
    L2 + X = 躺→站
    L2 + 左 (长按) = 落座
    L2 + A (长按) = 蹲↔站

  从运控:
    R2 + A = 走跑运控
    R1 + X = 常规运控 (单腰)
    R1 + Y = 常规运控 (三腰)
    R2 + B = 越障运控
    R1 + B = 舞蹈运控

  调速:
    R2 + 上/下 = 速度高低

  调试:
    L2 + R2 = 诊断/调试模式 (仅从阻尼或零力矩)

已知 FSM ID 表 (来自文档 index.md):
  0 = ZeroTorque      1 = Damp           2 = 位控下蹲
  3 = Sit             4 = 锁定站立       500 = 常规运控 (单腰)
  501 = 常规运控 (三腰)  702 = 躺起         706 = 蹲起
  801/802 = 走跑运控
"""
import sys
import time
import threading
from typing import Optional


# ── 遥控器按键 → (目标模式描述, 预期 FSM ID 候选) ──
# FSM ID 为 None 表示待实测确认.

KEY_COMBOS = [
    # (组合键描述, 目标描述, 预期 FSM ID, 风险)
    # ─ 从零力矩出发 ─
    ("L2 + Y",       "零力矩",          0,   "无"),
    ("L2 + B",       "阻尼 (急停)",      1,   "脱力! 需悬挂"),
    ("L2 + 上",      "锁定站立",         4,   "站起"),
    ("L2 + X",       "躺→站",           702, "从躺到站"),
    ("L2 + 左 (长按)", "落座",           3,   "坐下"),
    ("L2 + A (长按)",  "蹲↔站",         706, "姿态切换"),

    # ─ 从运控出发 ─
    ("R1 + X",       "常规运控 (单腰)",   500, "站起+平衡"),
    ("R1 + Y",       "常规运控 (三腰)",   501, "站起+平衡+三腰"),
    ("R2 + A",       "走跑运控",         None, "801 或 802? 待实测"),
    ("R2 + B",       "越障运控",         None, "ID 未知"),
    ("R1 + B",       "舞蹈运控",         None, "ID 未知"),

    # ─ 调试入口 ─
    ("L2 + R2",      "诊断/调试模式",     None, "仅从阻尼或零力矩"),

    # ─ 调速 (不切换模式, 但在运控内生效) ─
    ("R2 + 上",      "高速",             None, "走跑运控内调速"),
    ("R2 + 下",      "低速",             None, "走跑运控内调速"),
]


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def grade_transition() -> tuple[int, str]:
    print()
    print("  打分:")
    print("    1 = 顺利完成, 终态稳定")
    print("    2 = 完成但有小不稳")
    print("    3 = 完成但抖动明显")
    print("    4 = 被拒 / 危险 / 不可达")
    while True:
        ans = prompt("输入 1-4")
        if ans in {'1', '2', '3', '4'}:
            note = prompt("简短补充")
            return int(ans), note
        print("  请输入 1-4")


def main():
    if len(sys.argv) < 2:
        print("用法: python 24_mode_switch_topology.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    print("=" * 70)
    print("24_mode_switch_topology — FSM 可达图 (遥控器路径)")
    print("=" * 70)
    print()
    print("本脚本通过遥控器组合键触发模式切换, 用 rt/sportmodestate")
    print("读取真实的 fsm_id. 不使用 SetFsmId API.")
    print()
    print("重要约束:")
    print("  - 阻尼模式脱力, 必须在吊架下运行!")
    print("  - 调试模式仅从阻尼或零力矩进入 (不从运控直接进)")
    print("  - 运控模式之间切换可能需要先回阻尼")
    print("=" * 70)
    input("\n准备好了按 Enter 开始 >>> ")

    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init()
    print("OK: rt/sportmodestate 订阅就绪")

    # ── 共享状态: 最新 SportModeState ──
    _lock = threading.Lock()
    _latest: dict = {'fsm_id': -1, 'fsm_mode': -1}

    running = True

    def _read_sport_mode():
        while running:
            msg = sub.Read(timeout=500)
            if msg is None:
                continue
            with _lock:
                _latest['fsm_id'] = msg.fsm_id
                _latest['fsm_mode'] = msg.fsm_mode  # 0=静态, 1=动态

    _thread = threading.Thread(target=_read_sport_mode, daemon=True)
    _thread.start()
    time.sleep(1)

    def current_fsm() -> tuple[int, int]:
        with _lock:
            return _latest['fsm_id'], _latest['fsm_mode']

    def current_fsm_str() -> str:
        fsm_id, fsm_mode = current_fsm()
        mode_str = "静态" if fsm_mode == 0 else ("动态" if fsm_mode == 1 else f"?{fsm_mode}")
        return f"fsm_id={fsm_id}  fsm_mode={fsm_mode}({mode_str})"

    print(f"\n当前状态: {current_fsm_str()}")

    # ── 逐按键测试 ──
    results = []

    for (combo, desc, expected_id, risk) in KEY_COMBOS:
        print("\n" + "=" * 70)
        print(f"组合键: {combo}")
        print(f"目标: {desc}")
        if expected_id is not None:
            print(f"预期 FSM ID: {expected_id}")
        else:
            print(f"预期 FSM ID: 未知 (本实验要确认)")
        if risk != "无":
            print(f"⚠️  风险: {risk}")
        print(f"当前状态: {current_fsm_str()}")
        print("=" * 70)

        ans = prompt("准备好了按 Enter, 输入 's' 跳过, Ctrl+C 退出 > ")
        if ans.lower() == 's':
            print(f"  跳过 {combo}")
            continue

        fsm_before, mode_before = current_fsm()
        t_before = time.monotonic()

        print(f"  请按 {combo} ...")
        prompt_continue("操作完毕回车")

        # 等 fsm_id 变化 或 超时
        changed = False
        t_start = time.monotonic()
        while time.monotonic() - t_start < 10.0:
            fsm_after, mode_after = current_fsm()
            if fsm_after != fsm_before:
                changed = True
                break
            time.sleep(0.1)

        elapsed = time.monotonic() - t_before if changed else -1
        fsm_after, mode_after = current_fsm()

        if changed:
            print(f"\n  fsm_id 变化: {fsm_before} → {fsm_after}  用时 {elapsed:.2f}s")
            print(f"  当前: {current_fsm_str()}")
        else:
            print(f"\n  fsm_id 未变化 (仍 {fsm_after}) — 可能不可达 或 按键未触发")
            print(f"  当前: {current_fsm_str()}")

        grade, note_text = grade_transition()

        results.append({
            'combo': combo,
            'desc': desc,
            'expected': expected_id,
            'before': fsm_before,
            'after': fsm_after,
            'mode_after': mode_after,
            'changed': changed,
            'elapsed': elapsed,
            'grade': grade,
            'note': note_text,
        })

    # ── 收尾 ──
    running = False
    _thread.join(timeout=2)
    sub.Close()

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("FSM 可达图汇总")
    print("=" * 70)
    print()
    print(f"{'组合键':<16} {'目标':<16} {'预期':<6} {'before':<6} {'after':<6} {'变化':<5} {'用时':<8} {'分':<4} 备注")
    print("-" * 90)
    for r in results:
        chg = "✓" if r['changed'] else "✗"
        elapsed_s = f"{r['elapsed']:.2f}s" if r['elapsed'] >= 0 else "—"
        expected_s = str(r['expected']) if r['expected'] is not None else "?"
        print(f"{r['combo']:<16} {r['desc']:<16} {expected_s:<6} {r['before']:<6} "
              f"{r['after']:<6} {chg:<5} {elapsed_s:<8} {r['grade']:<4} {r['note']}")
    print()
    print("反馈给模型: 模型据此构建遥控器按键 → FSM ID 映射 + 可达边表.")


if __name__ == "__main__":
    main()
