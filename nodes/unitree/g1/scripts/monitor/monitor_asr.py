#!/usr/bin/env python3
"""
monitor_asr — 监控 G1 语音识别 (ASR) 输入

订阅 rt/audio_msg 和 rt/audio_msg/filter, 实时打印 ASR 识别结果.
启动时尝试开启麦克风, Ctrl+C 退出时关闭.

用法:
  python monitor_asr.py <networkInterface>
  python monitor_asr.py eth0

输出:
  [时间] ASR: "识别文本"  speaker=N  angle=N°  confidence=N.N  final=True/False
"""

import sys
import time
import threading
import json


def try_start_asr(audio_client) -> bool:
    """盲探 ASR 启动协议. 来自 script 23 的 _Call(1002, ...) 探测."""
    params_to_try = [
        {"action": "start"},
        {"start": True},
        {"enable": 1},
        {},
    ]
    for params in params_to_try:
        try:
            if isinstance(params, dict):
                param_str = json.dumps(params)
            else:
                param_str = str(params)
            code, data = audio_client._Call(1002, param_str)
            if code == 0:
                print(f"  ASR 启动成功 (params={params})")
                return True
            else:
                print(f"  ASR _Call(1002, {params}): code={code} data={data}")
        except Exception as e:
            print(f"  ASR _Call(1002, {params}): exception={e}")
    print("  ! 未能通过 _Call(1002) 启动 ASR. 尝试直接订阅 rt/audio_msg ...")
    return False


def try_stop_asr(audio_client) -> None:
    try:
        audio_client._Call(1002, json.dumps({"action": "stop"}))
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        print("用法: python monitor_asr.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (interface={nic}) ...")
    ChannelFactoryInitialize(0, nic)

    # ASR 结果订阅
    sub_main = ChannelSubscriber("rt/audio_msg", String_)  # 实际是 String_ JSON
    sub_main.Init()
    print("订阅 rt/audio_msg 就绪")

    sub_filter: object = None
    try:
        sub_filter = ChannelSubscriber("rt/audio_msg/filter", String_)
        sub_filter.Init()
        print("订阅 rt/audio_msg/filter 就绪")
    except Exception as e:
        print(f"rt/audio_msg/filter 订阅失败: {e} (继续)")

    # 尝试启动 ASR
    audio = AudioClient()
    audio.SetTimeout(5.0)
    audio.Init()
    print("AudioClient 就绪")

    try_start_asr(audio)

    running = True

    def _poll(sub, label: str):
        while running:
            msg = sub.Read(timeout=500)
            if msg is None:
                continue
            try:
                raw = msg.data  # String_.data: str
                data = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                data = {"raw": str(msg.data)[:200] if msg.data else "(empty)"}

            ts = time.strftime('%H:%M:%S')
            text = data.get('text', data.get('transcript', ''))
            if text and text != data.get('raw', ''):
                speaker = data.get('speaker_id', '?')
                angle = data.get('angle', '?')
                conf = data.get('confidence', '?')
                is_final = data.get('is_final', True)
                print(f"  [{ts}] {label}: \"{text}\"  speaker={speaker}"
                      f"  angle={angle}°  conf={conf}  final={is_final}")
            else:
                # 原始输出, 帮助理解数据格式
                summary = json.dumps(data, ensure_ascii=False)[:200]
                print(f"  [{ts}] {label} raw: {summary}")

    t1 = threading.Thread(target=_poll, args=(sub_main, "ASR"), daemon=True)
    t1.start()
    t2 = None
    if sub_filter is not None:
        t2 = threading.Thread(target=_poll, args=(sub_filter, "ASR(filter)"), daemon=True)
        t2.start()

    print()
    print("麦克风已开启. 对着 G1 说话.")
    print("按 Ctrl+C 退出 (会自动关闭麦克风)")
    print()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出...")
    finally:
        running = False
        t1.join(timeout=2)
        if t2 is not None:
            t2.join(timeout=2)
        sub_main.Close()
        if sub_filter is not None:
            try:
                sub_filter.Close()
            except Exception:
                pass
        try_stop_asr(audio)
        print("ASR 已关闭")


if __name__ == "__main__":
    main()
