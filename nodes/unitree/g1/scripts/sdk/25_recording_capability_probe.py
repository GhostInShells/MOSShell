#!/usr/bin/env python3
"""
25_recording_capability_probe — G1 内置录制能力探测

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

人类工程师 2026-06-28 提到: G1 PC1 蓝牙连手机后, 通过手机 App 可以录制动作.
但 SDK 完全不暴露录制接口 (grep 全检, example 0 命中).

如果能找到任何"通过 SDK / DDS topic / 文件系统"接近录制能力的入口,
"录制+回放"channel 就可以接 SDK 通路; 找不到的话, 必须自造 (rt/arm_sdk 写关节角时间序列).

本脚本做三件事:
  1. 扫描 DDS topic, 找名字含 record / playback / motion / capture / replay 的
  2. 试探未公开的 RPC API ID (1100-1110, 7200-7210 等保守范围)
  3. 提供"人类操作录制" + "脚本同步监控所有 LowState 字段变化"的协作探测模式

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 + 任何模式
  2. **如果有手机**: 蓝牙连接 G1 PC1, 装 Unitree App, 准备好录制功能
  3. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

测试流程:
  阶段 1: 用 cyclonedds CLI 扫 topic 名找候选 (你在另一终端跑命令)
  阶段 2: 协作探测 — 你用手机触发"开始录制", 脚本同时监控
          看哪些 topic 在录制开始/结束时有变化, 文件系统是否产生新文件
  阶段 3: 试探未公开 RPC API ID

风险:
  无运动指令. 唯一副作用: RPC 试探可能触发未定义行为 (低概率).
"""
import sys
import time
import threading
import json
import os
from typing import Optional


# 可疑 topic 关键词
SUSPICIOUS_TOPIC_PATTERNS = [
    "record", "replay", "playback", "motion", "capture", "trajectory",
    "teach", "demo", "save",
]

# 试探的未公开 RPC API ID (loco service)
PROBE_API_IDS_LOCO = [
    # 7000-7099: 已知 Get*
    # 7100-7199: 已知 Set*
    # 7200+ 未知, 保守试探
    7200, 7201, 7202, 7210, 7220, 7300,
]

# 试探的未公开 audio API
PROBE_API_IDS_AUDIO = [
    # 1001 TTS, 1002 ASR, 1003 PlayStream, 1004 PlayStop,
    # 1005 GetVolume, 1006 SetVolume, 1010 LED
    # 试探 1007-1009, 1011-1020
    1007, 1008, 1009, 1011, 1020, 1100,
]

# 候选录制文件路径(出厂 PC1 可能存这里)
CANDIDATE_RECORD_DIRS = [
    "/home/unitree/records",
    "/home/unitree/motions",
    "/var/lib/unitree/records",
    "/tmp/unitree_records",
    "/data/unitree",
]


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def main():
    if len(sys.argv) < 2:
        print("用法: python 25_recording_capability_probe.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print("=" * 70)
    print("25_recording_capability_probe — G1 内置录制能力探测")
    print("=" * 70)
    print()
    input("准备好了按 Enter 开始 >>> ")

    print(f"\n初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    loco = LocoClient()
    loco.SetTimeout(5.0)
    loco.Init()

    audio = AudioClient()
    audio.SetTimeout(5.0)
    audio.Init()

    # ── 阶段 1: topic 扫描 ──
    print("\n" + "=" * 70)
    print("阶段 1: DDS topic 扫描")
    print("=" * 70)
    print()
    print("请在另一终端 run:")
    print("  source /etc/profile.d/cyclonedds.sh")
    print("  cyclonedds ls | grep -iE 'record|replay|playback|capture|teach|demo|save|trajectory'")
    print()
    topic_input = prompt("命中的 topic 名 (多个空格分隔, 没有就回车)")
    suspicious_topics = topic_input.split() if topic_input else []

    if suspicious_topics:
        print(f"  收到 {len(suspicious_topics)} 个候选: {suspicious_topics}")
    else:
        print("  没有命中 — G1 录制大概率不走专用 topic")

    # ── 阶段 2: 协作探测 (你用手机触发) ──
    print("\n" + "=" * 70)
    print("阶段 2: 协作探测 — 人类触发录制 + 脚本监控")
    print("=" * 70)
    print()
    has_phone = prompt("你有手机连上 G1 PC1 蓝牙 + 装了 Unitree App 吗? (y/N)")
    if has_phone.lower().startswith('y'):
        print("\n  好. 接下来你触发录制, 我同步监控 LowState 流和文件系统.")
        prompt_continue("打开 Unitree App, 找到录制功能, 准备好后回车")

        # 监控 LowState 字段变化的关键 — 找有"录制中"标志的字段
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init()

        # 取一个基线 snapshot
        msg_before = sub.Read(timeout=2000)
        if msg_before is None:
            print("  !! LowState 不到, 跳过监控")
        else:
            print(f"  基线 snapshot: tick={msg_before.tick}, mode_machine={msg_before.mode_machine}")
            print(f"    motor_state[0].mode={msg_before.motor_state[0].mode}")

            prompt_continue("现在用手机点'开始录制', 然后回车")

            # 拿录制开始后的 snapshot
            msg_recording = sub.Read(timeout=2000)
            if msg_recording is not None:
                print(f"  录制开始 snapshot: tick={msg_recording.tick}, mode={msg_recording.mode_machine}")
                print(f"    motor_state[0].mode={msg_recording.motor_state[0].mode}")
                # 简单字段 diff
                diffs = []
                if msg_before.mode_machine != msg_recording.mode_machine:
                    diffs.append(f"mode_machine: {msg_before.mode_machine} → {msg_recording.mode_machine}")
                if msg_before.motor_state[0].mode != msg_recording.motor_state[0].mode:
                    diffs.append(f"motor_state[0].mode 变化")
                if diffs:
                    print(f"  字段 diff: {diffs}")
                else:
                    print(f"  LowState 字段无明显变化 — 录制状态可能不通过 LowState 暴露")

            print("\n  现在做一些动作让录制有内容(挥手等)...")
            time.sleep(8)

            prompt_continue("用手机点'停止录制', 然后回车")

            # 看文件系统
            print("\n  检查候选录制目录是否有新文件...")
            print("  (这需要 ssh 到 PC2 后检查 — 你也可以远程到 unitree 帐号下查)")
            for d in CANDIDATE_RECORD_DIRS:
                # 我们在 PC2, 但录制文件可能在 PC1 — 仅能查 PC2 局部
                if os.path.exists(d):
                    files = os.listdir(d)
                    print(f"    {d}: 存在, {len(files)} 个文件")
                else:
                    print(f"    {d}: 不存在")

            print()
            print("  如果你能 ssh 到 PC1 (它可能是 192.168.x.x 内网某个 IP),")
            print("  可以执行: find / -name '*.json' -newer /tmp/check_marker -type f 2>/dev/null")
            print("  来找录制文件. 记下命中的路径反馈.")

            sub.Close()
    else:
        print("  跳过协作探测 (没有手机或不方便).")

    # ── 阶段 3: 试探未公开 RPC API ID ──
    print("\n" + "=" * 70)
    print("阶段 3: 试探未公开 RPC API ID")
    print("=" * 70)
    print()
    print("对 loco service 和 voice service 的几个保守 API ID 做空参数调用,")
    print("看是否有'未注册 API'之外的返回. 任何 code != -某些标准错误码 都值得关注.")

    print("\n  Loco service 试探:")
    for api_id in PROBE_API_IDS_LOCO:
        try:
            code, data = loco._Call(api_id, "{}")
            print(f"    Loco API {api_id}: code={code} data={str(data)[:80]}")
        except Exception as e:
            print(f"    Loco API {api_id}: <exception: {type(e).__name__}: {e}>")
        time.sleep(0.3)

    print("\n  Audio (voice) service 试探:")
    for api_id in PROBE_API_IDS_AUDIO:
        try:
            code, data = audio._Call(api_id, "{}")
            print(f"    Audio API {api_id}: code={code} data={str(data)[:80]}")
        except Exception as e:
            print(f"    Audio API {api_id}: <exception: {type(e).__name__}: {e}>")
        time.sleep(0.3)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print()
    if suspicious_topics:
        print(f"阶段 1: 命中 topic = {suspicious_topics}")
    else:
        print("阶段 1: 无命中")
    print()
    print("阶段 2: 见前面输出(协作探测的 diff + 文件系统)")
    print()
    print("阶段 3: 任何 code 不是 '常见错误'(如 -5 等) 的 API ID 都值得关注")
    print()
    print("反馈给模型: 模型据此决定 recording channel 是接 SDK 还是自造 (rt/arm_sdk).")


if __name__ == "__main__":
    main()
