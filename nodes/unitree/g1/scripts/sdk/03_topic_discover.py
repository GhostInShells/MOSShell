#!/usr/bin/env python3
"""
DDS topic 真值发现 — cyclonedds CLI wrapper。

前任版本是硬编码 print 清单，没有真实发现能力。本版调用 unitree 帐号下的
cyclonedds CLI (出厂栈，安装在 ~/cyclonedds_ws/install/cyclonedds/bin/) 做真扫描。

执行要求:
  - 必须在 unitree 帐号下运行 (或 moss 帐号 source /etc/profile.d/cyclonedds.sh)
  - 需 G1 开机 + DDS 通讯正常 (ufw 已修复 IP 分片)

用法:
  python 03_topic_discover.py        # 默认扫描所有 topic
  python 03_topic_discover.py rt/    # 仅打印以 rt/ 开头的 topic

如果 cyclonedds CLI 不在 PATH，按提示运行 source /etc/profile.d/cyclonedds.sh 再试。
"""
import sys
import shutil
import subprocess


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else None

    cli = shutil.which("cyclonedds")
    if cli is None:
        print("FAIL: cyclonedds CLI 不在 PATH。\n")
        print("解决:")
        print("  1. unitree 帐号: 出厂栈应已包含 ~/cyclonedds_ws/install/cyclonedds/bin/")
        print("     若仍找不到，运行 source ~/.bashrc 或检查 cyclonedds_ws install")
        print("  2. moss 帐号:  source /etc/profile.d/cyclonedds.sh")
        print("  3. 兜底 (Python 端探测):")
        print("     运行 06_battery_sub.py — 会试探多个候选 topic 名")
        sys.exit(1)

    print(f"调用 cyclonedds CLI: {cli}")
    print(f"等待 ~3s 收集 topic discovery...\n")

    try:
        # `cyclonedds ls` 列出所有 topic + 读者写者
        result = subprocess.run(
            [cli, "ls", "--suppress-progress-bar"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        print("FAIL: cyclonedds ls 超时 15s — DDS 通讯可能不通")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"FAIL: 调用失败 {e}")
        sys.exit(1)

    if result.returncode != 0:
        print(f"WARN: cyclonedds ls 返回 {result.returncode}")
        print(f"stderr: {result.stderr}")

    out = result.stdout
    if prefix:
        lines = [ln for ln in out.splitlines() if prefix in ln]
        print(f"=== 过滤前缀 '{prefix}' ===")
        print("\n".join(lines))
    else:
        print(out)

    print("\n=== 下一步 ===")
    print("把上面的真实 topic 清单与 docs/sdk-topics.md 对比，订正前任硬编码清单。")
    print("特别关注:")
    print("  - rt/lowstate vs rt/lf/lowstate (G1 实际发哪个，或两个都发)")
    print("  - rt/sportmodestate (前任 session 报无数据)")
    print("  - rt/odommodestate / rt/lf/odommodestate (前任阻塞)")
    print("  - rt/bmsstate / rt/lf/bmsstate")
    print("  - rt/audio_msg (ASR 输出)")


if __name__ == "__main__":
    main()