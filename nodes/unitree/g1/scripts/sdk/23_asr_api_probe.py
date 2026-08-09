#!/usr/bin/env python3
"""
23_asr_api_probe — 盲探 G1 内置 ASR 服务的调用协议

═══════════════════════════════════════════════════════════════════════════════
为什么必须跑这个脚本
═══════════════════════════════════════════════════════════════════════════════

SDK 注册了 ROBOT_API_ID_AUDIO_ASR = 1002, 但 Python 客户端 AudioClient 没有暴露
Asr() 调用方法. 协议层存在, 调用约定未知.

要让 g1 channel 里的 asr sensor 跑起来, 必须知道:
  Q1. _Call(1002, ???) 传什么参数才不返回 error code
  Q2. 结果是同步返回 JSON(含 text/speaker_id/方位) 还是异步发 DDS topic
  Q3. ASR 启动后是一次性 / 流式 / 唤醒词触发
  Q4. 中文/英文支持, 字段名

本脚本对 _Call(1002, ...) 做系统性盲探, 同时扫描可疑 DDS topic.

═══════════════════════════════════════════════════════════════════════════════
执行人指引
═══════════════════════════════════════════════════════════════════════════════

前置:
  1. G1 已开机 (任何模式都行 — ASR 不依赖运动模式)
  2. G1 PC1 mic 已就绪 (出厂自带, 不需要额外配置)
  3. 你身处 G1 前方 1-2m, 准备说几句中文 + 英文
  4. cd .moss_ws/apps/bodies/g1 && source .venv/bin/activate

测试流程:
  阶段 1: 盲探 _Call(1002, {}) — 看最简空参数的返回
  阶段 2: 盲探常见参数组合 — {"action":"start"} / {"timeout":5} 等
  阶段 3: 扫 DDS topic — 看有没有 asr/speech/voice/transcript 相关的 topic
  阶段 4: 启动 ASR 后, 你说中文 + 英文, 看返回是同步 / 异步 / 静默

每阶段输出原始返回(code + raw data 字符串), 你不用动脑, 看到后复制反馈即可.

风险:
  无运动指令. 唯一副作用: 可能触发 G1 PC1 的 ASR 服务, 影响 mic 占用.
"""
import sys
import time
import threading
import json
from typing import Optional


CANDIDATE_PARAMS = [
    # (label, dict_or_str)
    ("空参数",            {}),
    ("action=start",     {"action": "start"}),
    ("action=stop",      {"action": "stop"}),
    ("start=true",       {"start": True}),
    ("enable=1",         {"enable": 1}),
    ("timeout=5",        {"timeout": 5}),
    ("mode=once",        {"mode": "once"}),
    ("mode=stream",      {"mode": "stream"}),
    ("language=zh",      {"language": "zh"}),
]

# 可疑 ASR 相关 topic 名候选(用 cyclonedds CLI 扫)
SUSPICIOUS_TOPIC_PATTERNS = [
    "asr", "speech", "voice", "transcript", "stt", "recogniz", "wakeup", "wake_up",
]


def prompt_continue(msg: str) -> None:
    print(f"\n[操作] {msg}")
    input("    按 Enter 继续 >>> ")


def prompt(msg: str) -> str:
    print(f"\n[操作] {msg}")
    return input("    > ").strip()


def call_asr(audio_client, param_dict: dict) -> tuple[int, str]:
    """对 audio_client 调 _Call(1002, json.dumps(param_dict)). 返回 (code, raw_data_str)."""
    from unitree_sdk2py.g1.audio.g1_audio_api import ROBOT_API_ID_AUDIO_ASR
    param_str = json.dumps(param_dict)
    try:
        code, data = audio_client._Call(ROBOT_API_ID_AUDIO_ASR, param_str)
        return code, str(data)
    except Exception as e:
        return -999, f"<exception: {type(e).__name__}: {e}>"


def main():
    if len(sys.argv) < 2:
        print("用法: python 23_asr_api_probe.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print("=" * 70)
    print("23_asr_api_probe — G1 ASR 服务调用约定盲探")
    print("=" * 70)
    print()
    input("准备好了按 Enter 开始 >>> ")

    print(f"\n初始化 DDS (interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()
    print("OK: AudioClient 就绪\n")

    # ── 阶段 1+2: 参数盲探 ──
    print("=" * 70)
    print("阶段 1+2: 对 _Call(1002, ...) 做参数盲探")
    print("=" * 70)
    print()

    results = []
    for label, params in CANDIDATE_PARAMS:
        print(f"  尝试: {label:<20} 参数={json.dumps(params)}")
        code, data = call_asr(audio, params)
        # 截断超长返回
        data_short = data[:200] + ("..." if len(data) > 200 else "")
        print(f"    -> code={code}  data={data_short}")
        results.append({'label': label, 'params': params, 'code': code, 'data': data})
        time.sleep(0.5)  # 别打太快

    # ── 阶段 3: 扫 DDS topic ──
    print("\n" + "=" * 70)
    print("阶段 3: 扫描可疑 DDS topic (用 cyclonedds CLI)")
    print("=" * 70)
    print()
    print("请在另一个终端 run:")
    print("  source /etc/profile.d/cyclonedds.sh  # 如果未 source")
    print("  cyclonedds ls | grep -iE 'asr|speech|voice|transcript|stt|recog|wake'")
    print()
    print("把命中的 topic 名复制回这里:")
    topic_input = prompt("命中的 topic (多个用空格分隔, 没有就回车跳过)")
    suspicious_topics = topic_input.split() if topic_input else []
    if suspicious_topics:
        print(f"  收到 {len(suspicious_topics)} 个候选 topic: {suspicious_topics}")
    else:
        print("  没有命中, ASR 结果大概率不走独立 topic")

    # ── 阶段 4: 真正说话 + 看反应 ──
    print("\n" + "=" * 70)
    print("阶段 4: 启动 ASR 后说话, 看反应")
    print("=" * 70)
    print()
    print("策略: 找一个在阶段 1+2 中 code == 0 的调用作为启动信号.")
    success_calls = [r for r in results if r['code'] == 0]
    if not success_calls:
        print("  !! 阶段 1+2 中没有任何调用返回 code=0.")
        print("     可能 ASR API 需要特殊鉴权 / 不支持这种调用. 跳过阶段 4.")
        sys.exit(0)

    print(f"  阶段 1+2 中 {len(success_calls)} 个调用成功:")
    for i, r in enumerate(success_calls):
        print(f"    [{i}] {r['label']}  data={r['data'][:80]}")

    idx_str = prompt("选哪一个作为 ASR 启动调用? 输入序号 0-N (默认 0)")
    try:
        idx = int(idx_str) if idx_str else 0
    except ValueError:
        idx = 0
    chosen = success_calls[idx]
    print(f"\n  使用: {chosen['label']}")

    print("\n  -> 启动 ASR ...")
    code, data = call_asr(audio, chosen['params'])
    print(f"     code={code}  data={data}")

    print("\n  接下来 10 秒, 请清楚地说一句中文(比如'你好世界, 我是测试'):")
    prompt_continue("准备好了回车, 然后开始说")

    t_start = time.monotonic()
    while time.monotonic() - t_start < 10:
        time.sleep(0.5)

    print("\n  -> 再次调用 ASR(可能是 'get result' 语义) ...")
    code2, data2 = call_asr(audio, chosen['params'])
    print(f"     code={code2}  data={data2}")

    # 试 stop
    print("\n  -> 尝试 stop ...")
    code_stop, data_stop = call_asr(audio, {"action": "stop"})
    print(f"     code={code_stop}  data={data_stop}")

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print()
    print("阶段 1+2 完整结果:")
    for r in results:
        ok = "✓" if r['code'] == 0 else " "
        print(f"  {ok} {r['label']:<20} code={r['code']:<5} data={r['data'][:80]}")

    if suspicious_topics:
        print(f"\n阶段 3 可疑 topic: {suspicious_topics}")

    print(f"\n阶段 4 说话后的两次返回:")
    print(f"  调用 1 (启动): code={code} data={data}")
    print(f"  调用 2 (说话后): code={code2} data={data2}")
    print(f"  stop: code={code_stop} data={data_stop}")
    print()
    print("反馈给模型实例:")
    print("  - 阶段 4 的 data 内容中是否含文本结果?")
    print("  - 阶段 3 是否找到 ASR 专用 topic? 如有, 单独写订阅探测脚本")
    print("  - 模型据此决定 asr sensor channel 的实现路径(同步 _Call vs DDS 订阅)")


if __name__ == "__main__":
    main()
