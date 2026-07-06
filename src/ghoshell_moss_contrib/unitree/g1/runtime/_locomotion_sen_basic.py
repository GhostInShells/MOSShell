"""
_locomotion_sen_basic — locomotion runtime 的基础场景验证脚本.

场景:
  你站在 G1 前方 (吊架可有可无, 但 0.25 m/s 慢速直行 + 1 米活动半径相对安全).
  G1 在 Sport 模式 (FSM 6 - "走跑运控"), MOSS 通过 LocoClient 协控. 跑本脚本,
  在 prompt 里输入命令, 观察 G1 的物理反应 + Observe 返回文本.

  双工体验 (主线程 prompt, 命令是 async 直 await). 关键验证场景是**抢占切换**:
  跑 walk_forward 5 后, 第 2 秒立刻输 turn_left 1, 看 G1 是否平滑切换不抽搐,
  以及第一行 Observe 是否 "preempted_by:turn_left".

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._locomotion_sen_basic <nic>

  <nic>: 网络接口名 (PC2 上一般 eth0 或具体网卡名). 同 sdk.bootstrap 的 nic 参数.

前置:
  - G1 开机, FSM 处于运动模式 (Sport, fsm_id=500/801/802). 遥控器 R2+A 进运控.
  - 吊架/站立环境安全, 周围 1-2 米清场.
  - PC2 已 uv sync, unitree_sdk2_python 已 clone 到 .moss_ws/apps/bodies/g1/src/.
  - **如 G1 在 Sit/Damp/调试模式, LocoClient.Move 会被静默拒收或报错** — 切到 Sport 再跑.

预期 (跑通时人看到):
  > 输入: f 0.5
  [walk_forward duration after 0.50s]
  > 输入: f 2
  > 输入: tl 0.5 medium     <- 立刻输 (抢占测试)
  [walk_forward preempted_by:turn_left_medium after 0.63s]
  [turn_left_medium duration after 0.50s]
  > 输入: s                 <- 中途强停
  [stopped walk_forward]
  > 输入: q

  G1 物理反应预期:
  - 前进/后退/横移: 平稳走 duration 秒后立即停 (StopMove → 回 stand idle).
  - 转身: 原地 yaw 旋转 duration 秒后停.
  - 抢占切换 (walk → turn): 不应有"突然 stop 再启动"的抽搐 — version 化的 StopMove
    会被新 session 挡掉, G1 直接从 walk 速度过渡到 turn 速度. 实测如发现抽搐,
    说明 G1 主板对 Move(0,0,0) 的过渡处理跟我们预期不同, 需另谋方案.
  - 强停: 立即静止. 0.5s 内 Observe 返回.

调试:
  - `h` 打印 health() (current_command / version / elapsed_sec).
  - locomotion logger 是 "moss.g1.runtime.locomotion", 想看详情把 root logger
    设 DEBUG 或单独配 handler.

退出:
  - 输入 `q` (或 Ctrl+D / Ctrl+C) → stop_runtime() 兜底 StopMove + G1 回 stand.
  - 不要直接 kill -9, 否则 finally 不跑, G1 可能保持最后一帧速度 (或 Sport 自停).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from ghoshell_moss_contrib.unitree.g1.runtime import locomotion
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap

# ── 命令字符串解析 ──────────────────────────────────────────────────────

_HELP_TEXT = """
命令格式 (空格分隔):
  f <duration>            walk_forward
  b <duration>            walk_backward
  l <duration>            strafe_left
  r <duration>            strafe_right
  tl <duration> [speed]   turn_left  (speed: low/medium/high, 默认 low)
  tr <duration> [speed]   turn_right
  s                       stop (独立接口, 立即生效)
  h                       打印 health
  ?                       这份帮助
  q                       退出 (=Ctrl+D / Ctrl+C)

抢占测试: 输完前一条立即输下一条, 观察 Observe 的 reason.
""".strip()


async def _dispatch(raw: str) -> Optional[str]:
    """解析单行输入, 返回 Observe 文本或 None (帮助/health 类已自打印)."""
    parts = raw.strip().split()
    if not parts:
        return None
    cmd = parts[0].lower()

    if cmd in ("q", "quit", "exit"):
        return "__QUIT__"
    if cmd in ("?", "help"):
        print(_HELP_TEXT)
        return None
    if cmd == "h":
        print(f"[health] {locomotion.health()}")
        return None
    if cmd == "s":
        return await locomotion.stop()

    # 需要 duration 的命令
    if len(parts) < 2:
        print(f"[error] cmd '{cmd}' 需要 duration 参数. 输 ? 看帮助.")
        return None
    try:
        duration = float(parts[1])
    except ValueError:
        print(f"[error] duration '{parts[1]}' 不是合法数字.")
        return None

    if cmd == "f":
        return await locomotion.walk_forward(duration)
    if cmd == "b":
        return await locomotion.walk_backward(duration)
    if cmd == "l":
        return await locomotion.strafe_left(duration)
    if cmd == "r":
        return await locomotion.strafe_right(duration)
    if cmd in ("tl", "tr"):
        speed = parts[2].lower() if len(parts) >= 3 else "low"
        if speed not in ("low", "medium", "high"):
            print(f"[error] speed '{speed}' 必须是 low/medium/high.")
            return None
        if cmd == "tl":
            return await locomotion.turn_left(duration, speed)  # type: ignore[arg-type]
        return await locomotion.turn_right(duration, speed)  # type: ignore[arg-type]

    print(f"[error] 未知命令 '{cmd}'. 输 ? 看帮助.")
    return None


# ── 主循环 ──────────────────────────────────────────────────────────────

async def _main_loop() -> int:
    """主交互循环.

    并发模型: 移动命令 (f/b/l/r/tl/tr) 用 asyncio.create_task 后台跑, 主线程立刻
    回去接下一行 — 这样在第一条命令还没返回时, 你就能输第二条触发抢占测试.
    控制类命令 (s/h/?/q) 直接 await, 它们都是瞬时的, 没必要后台化.

    pending list 只是 GC 屏障 + 完成结果回显; 实际"哪个命令是 current"由 locomotion
    runtime 的 _current_version 决定, 主循环不管.
    """
    pending: list[asyncio.Task] = []

    print()
    print("=" * 60)
    print("G1 locomotion 实机验证. 输 ? 看命令, 输 q 退出.")
    print("=" * 60)

    loop = asyncio.get_running_loop()

    while True:
        # input() 是同步 blocking — 放到 executor 里, 不阻塞 event loop.
        try:
            raw = await loop.run_in_executor(None, input, "> 输入: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[ctrl-d/c] 退出中...")
            return 0
        if not raw.strip():
            continue

        # 先清理已完成的 pending (防止列表无限增长 + 打印它们的结果)
        for t in list(pending):
            if t.done():
                try:
                    result = t.result()
                    if result is not None and result != "__QUIT__":
                        print(f"[done] {result}")
                except Exception as e:
                    print(f"[done with exception] {e}")
                pending.remove(t)

        # 特例: stop / h / ? / q 直接 await (不需要并发)
        first = raw.strip().split()[0].lower()
        if first in ("s", "h", "?", "help", "q", "quit", "exit"):
            try:
                result = await _dispatch(raw)
            except Exception as e:
                print(f"[exception] {e}")
                continue
            if result == "__QUIT__":
                return 0
            if result is not None:
                print(f"[result] {result}")
            continue

        # 移动命令: create_task 后台跑, 主线程立刻回去接下一行 → 支持抢占测试
        task = asyncio.create_task(_dispatch(raw), name=f"cmd:{raw}")

        def _on_done(t: asyncio.Task, raw=raw):
            try:
                result = t.result()
                if result is not None:
                    print(f"\n[{raw}] → {result}\n> 输入: ", end="", flush=True)
            except Exception as e:
                print(f"\n[{raw}] EXC → {e}\n> 输入: ", end="", flush=True)
        task.add_done_callback(_on_done)
        pending.append(task)


def main() -> int:
    ap = argparse.ArgumentParser(prog="locomotion_sen_basic", description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("nic", help="DDS 网络接口名 (PC2 上一般 eth0 或具体网卡)")
    ap.add_argument("--log-level", default="INFO", help="root logger 级别. 默认 INFO.")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    print(f"[bootstrap] DDS nic={args.nic}, 连接 G1...")
    try:
        bootstrap(args.nic)
    except Exception as e:
        print(f"[error] bootstrap 失败: {e}", file=sys.stderr)
        return 1

    locomotion.start()
    print("[ok] locomotion runtime started.")
    print("[warn] 请确认 G1 在 Sport 模式 (R2+A 进运控). Sit/Damp 下 LocoClient 命令无效.")

    try:
        return asyncio.run(_main_loop())
    finally:
        print("[cleanup] stop_runtime + StopMove 兜底...")
        try:
            locomotion.stop_runtime(timeout=1.0)
        except Exception:
            logging.exception("stop_runtime 抛异常 (忽略)")
        print("[bye] G1 应回 stand idle.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
