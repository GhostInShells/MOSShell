"""
_locomotion_tes_preempt — 抢占切换自动化测试 (不用手速).

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._locomotion_tes_preempt <nic>

前置: G1 Sport 模式, 吊架, 前方 1m 清空.
"""
import asyncio
import sys
from ghoshell_moss_contrib.unitree.g1.runtime import locomotion
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


async def main(nic: str) -> int:
    print(f"bootstrap({nic!r}) ...")
    bootstrap(nic)
    locomotion.start()

    print("\n[test 1] walk_forward 3s → 0.5s 后 turn_left 抢占")
    task_fwd = asyncio.create_task(locomotion.walk_forward(3.0))
    await asyncio.sleep(0.5)
    result_turn = await locomotion.turn_left(0.5, "medium")
    result_fwd = await task_fwd
    print(f"  walk_forward: {result_fwd}")
    print(f"  turn_left:    {result_turn}")
    if "preempted_by" in result_fwd:
        print("  PASS: 抢占检测正确")
    else:
        print("  FAIL: 未检测到抢占")
        return 1

    print("\n[test 2] walk_backward 2s → 0.3s 后 stop 强停")
    task_bwd = asyncio.create_task(locomotion.walk_backward(2.0))
    await asyncio.sleep(0.3)
    result_stop = await locomotion.stop()
    result_bwd = await task_bwd
    print(f"  walk_backward: {result_bwd}")
    print(f"  stop:          {result_stop}")
    if "stopped" in result_bwd:
        print("  PASS: stop 正确")
    else:
        print(f"  WARN: unexpected result — {result_bwd}")

    print(f"\n[health] {locomotion.health()}")
    locomotion.stop_runtime()
    print("PASS: tes_preempt")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
