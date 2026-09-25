#!/usr/bin/env python3
"""
TTS 多语料探路：验证不同口播类型下 G1 内置 TTS 的边界。

测试维度：
  - 长度: 短句 / 中句 / 长段
  - 内容: 纯中文 / 中英混合 / 数字符号 / 拟人对话
  - 标点: 句号 / 逗号 / 感叹 / 问号 / 省略号 / 引号
  - 异常输入: 空格、换行、纯符号、emoji（看 G1 是否会发"奇怪声音"）

每条之间 sleep 足够长，由人类听完判断后回车继续，避免上一条没播完就被下一条覆盖。

用法: python 13_tts_corpus.py <networkInterface>
"""
import sys
import time

CORPUS = [
    # (标签, 文本)
    ("S1-极短", "你好。"),
    ("S2-短中文", "我是 MOSS，正在和你协作。"),
    ("S3-中英混合", "请打开 GitHub 上的 Pull Request 看一下。"),
    ("S4-数字单位", "当前电池电量百分之七十八，剩余续航约两小时三十分钟。"),
    ("S5-标点密集", "等等，让我想想——是这样吗？不，应该是那样！对，就是它。"),
    ("S6-长段平淡", "在多模态大模型架构里，Ghost 是持久化的智能体，而 Shell 是它进入物理世界的躯壳。MOSS 提供了 Shell 层的全部基础设施。"),
    ("S7-长段情感", "你看，星空是这样静谧地铺展在我们的头顶。每一颗星，都是亿万年前燃烧着的光，跨越浩瀚抵达此刻。"),
    ("S8-命令式", "向前迈一步，然后停下，举起左手。"),
    ("S9-对话体", "用户问：你能听见我吗？我回答：当然，我一直在听。"),
    ("S10-纯英文", "Hello world, this is a microphone test."),
    # 异常输入 — 验证脚本健壮性，可能触发"奇怪声音"
    ("X1-含空格", "这是  一段    含有    多个    空格的    文本。"),
    ("X2-含换行", "第一行。\n第二行。\n第三行。"),
    ("X3-纯标点", "……！？。"),
    ("X4-含 emoji", "今天天气很好 🌞 我们一起去散步吧 🚶"),
    ("X5-特殊符号", "圆周率约等于 3.14159，公式 E = mc^2 是著名的质能方程。"),
]


def main():
    if len(sys.argv) < 2:
        print("用法: python 13_tts_corpus.py <networkInterface>")
        sys.exit(1)
    nic = sys.argv[1]

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"初始化 DDS (domain=0, interface={nic})...")
    ChannelFactoryInitialize(0, nic)

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    # 读取原始音量便于结束时恢复
    code, vol_orig = audio.GetVolume()
    print(f"原始音量: {vol_orig}")

    audio.SetVolume(100)
    print("音量设为 100\n")

    print("=" * 50)
    print(f"准备播报 {len(CORPUS)} 条语料")
    print("每条之间会自动 sleep 8 秒；如听完想跳到下一条按 Ctrl+C 终止当前等待")
    print("=" * 50 + "\n")

    for i, (label, text) in enumerate(CORPUS, 1):
        print(f"[{i}/{len(CORPUS)}] {label}")
        print(f"  文本: {text!r}")
        code = audio.TtsMaker(text, 0)
        print(f"  TtsMaker code={code}")
        try:
            time.sleep(8)
        except KeyboardInterrupt:
            print("  (跳过等待)")
        print()

    # 恢复音量
    if isinstance(vol_orig, dict):
        v = vol_orig.get("volume", 100)
    else:
        v = vol_orig
    print(f"恢复音量到 {v}")
    audio.SetVolume(v)

    print("\n请人类反馈每条语料的实际表现：")
    print("  - 哪些清晰自然？")
    print("  - 哪些出现'奇怪声音'、停顿异常、读音错误？")
    print("  - 异常输入 (X1-X5) 是否影响后续播放？")


if __name__ == "__main__":
    main()
