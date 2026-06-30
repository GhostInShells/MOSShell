"""
_headphone_buttons_probe — 蓝牙耳机按键事件探针.

场景:
  这个脚本不属于 listener 模块. 它的目的是: 在 PC2 (或开发机) 上, 让你戴着蓝牙
  耳机, 按一遍所有按键 (单击交互键 / 双击 / 长按 / 音量+ / 音量-), 脚本把每个
  按键事件的原始事件签名 (type / code / value / timestamp) 完整打出来.

  你跑完一次, 把输出贴给模型 (我), 我就能据此写 g1.runtime.headphone_buttons —
  跟 sdk/_buttons.py 同范式的回调注册模块. 不同蓝牙耳机的按键事件签名差异极大
  (AirPods 单击 = play_pause, 三键耳机的中间键也可能是 play_pause, 但小米的
  双击可能是 next_track), 不实测无法预先写死.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_buttons_probe

  # Linux 上推荐先以 root 跑一次确认权限正常, 然后切回普通用户:
  sudo python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_buttons_probe

  # 跳过设备选择, 直接探测某个 /dev/input/event*:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_buttons_probe \
      --device /dev/input/event5

前置:
  Linux (主目标 — PC2):
    - pip install evdev          (uv pip install evdev)
    - 当前用户加入 input 组, 或脚本以 root 跑
      sudo usermod -a -G input $USER   # 然后 logout/login
    - 蓝牙耳机已连且在 /proc/bus/input/devices 里可见 (按键事件经 BlueZ HFP/AVRCP
      暴露成虚拟键盘 /dev/input/event*)

  macOS (开发兜底):
    - pip install pynput
    - 系统设置 → 隐私与安全 → 辅助功能, 允许 Terminal/iTerm 访问

预期:
  Linux 路径:
    [evdev] 共 12 个 input 设备:
      [0]  /dev/input/event0   AT Translated Set 2 keyboard
      [1]  /dev/input/event5   AirPods Pro (AVRCP)         ★ bluetooth-like
      [2]  /dev/input/event6   AirPods Pro (HFP)
      ...
    选择设备 (回车 [1]):
    [probe] 监听 /dev/input/event5  (AirPods Pro (AVRCP))
            按你想测的所有键. Ctrl+C 停止.

    [12:34:56.123]  EV_KEY  KEY_PLAYPAUSE      value=1 (pressed)
    [12:34:56.234]  EV_KEY  KEY_PLAYPAUSE      value=0 (released)
    [12:35:01.001]  EV_KEY  KEY_VOLUMEUP       value=1 (pressed)
    ...

    ^C
    [summary]
      Unique key events (paste this to the model):
        KEY_PLAYPAUSE        (EV_KEY code=164)   pressed=3  released=3
        KEY_VOLUMEUP         (EV_KEY code=115)   pressed=2  released=2
        KEY_VOLUMEDOWN       (EV_KEY code=114)   pressed=1  released=1

      Device path : /dev/input/event5
      Device name : AirPods Pro (AVRCP)
      Total events: 12

  macOS 路径:
    [pynput] 监听键盘事件 (媒体键经系统传入). 不能区分多个设备.
             按你想测的所有键. Ctrl+C 停止.

    [12:34:56] press   Key.media_play_pause
    [12:34:56] release Key.media_play_pause
    ...

读完 docstring 还看不懂请回去读 runtime/README.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from typing import Optional


# ── 平台分发 ─────────────────────────────────────────────────────────────

def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_mac() -> bool:
    return sys.platform == "darwin"


# ── Linux: evdev ─────────────────────────────────────────────────────────

_BT_HINT_KEYWORDS = (
    "bluetooth", "bt", "airpods", "wireless", "headset", "buds",
    "beats", "avrcp", "hfp", "a2dp",
)


def _list_evdev_devices():
    """返回 [(path, name, is_bt_like), ...]."""
    import evdev
    devices = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            name = dev.name
            is_bt = any(k in name.lower() for k in _BT_HINT_KEYWORDS)
            devices.append((path, name, is_bt))
            dev.close()
        except Exception:
            continue
    return devices


def _evdev_pick_default(devices) -> int:
    for i, (_, _, is_bt) in enumerate(devices):
        if is_bt:
            return i
    return 0


def _evdev_pick_by_path(devices, path) -> Optional[int]:
    for i, (p, _, _) in enumerate(devices):
        if p == path:
            return i
    return None


def _prompt_index(devices, default: int) -> int:
    while True:
        try:
            raw = input(f"选择设备 (回车 [{default}]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if not raw:
            return default
        try:
            idx = int(raw)
            if 0 <= idx < len(devices):
                return idx
        except ValueError:
            pass
        print(f"无效输入, 请输入 0..{len(devices)-1} 或直接回车")


def _evdev_main(args) -> int:
    try:
        import evdev
        from evdev import categorize, ecodes
    except ImportError:
        print("[error] evdev 未安装. 装一下:", file=sys.stderr)
        print("    uv pip install evdev", file=sys.stderr)
        print("    (或在项目根目录: uv add evdev)", file=sys.stderr)
        return 2

    if args.device:
        dev_path = args.device
        try:
            dev = evdev.InputDevice(dev_path)
        except PermissionError:
            print(
                f"[error] 没有权限读 {dev_path}. 当前用户不在 input 组. 选项:",
                file=sys.stderr,
            )
            print("  sudo python -m ghoshell_moss_contrib.unitree.g1.runtime._headphone_buttons_probe", file=sys.stderr)
            print("  或 sudo usermod -a -G input $USER (然后 logout/login)", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[error] 无法打开 {dev_path}: {e}", file=sys.stderr)
            return 1
    else:
        devices = _list_evdev_devices()
        if not devices:
            print(
                "[error] 没找到任何 input 设备. 可能没权限读 /dev/input/event*.\n"
                "  以 root 跑或加入 input 组.",
                file=sys.stderr,
            )
            return 1
        print(f"\n[evdev] 共 {len(devices)} 个 input 设备:")
        for i, (path, name, is_bt) in enumerate(devices):
            mark = " ★ bluetooth-like" if is_bt else ""
            print(f"  [{i:2d}]  {path}  {name}{mark}")
        print()
        try:
            idx = _prompt_index(devices, _evdev_pick_default(devices))
        except KeyboardInterrupt:
            return 130
        dev_path, dev_name, _ = devices[idx]
        try:
            dev = evdev.InputDevice(dev_path)
        except Exception as e:
            print(f"[error] 无法打开 {dev_path}: {e}", file=sys.stderr)
            return 1

    print(f"\n[probe] 监听 {dev.path}  ({dev.name})")
    print("        按你想测的所有键 (单击 / 双击 / 长按 / 音量). Ctrl+C 停止.\n")

    # 累计 key 事件统计 (用于退出时的 summary)
    key_stats: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"name": "", "pressed": 0, "released": 0, "held": 0}
    )
    total_events = 0
    started_at = time.time()

    try:
        for event in dev.read_loop():
            total_events += 1
            ts = event.timestamp()
            ts_str = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"

            type_name = ecodes.EV.get(event.type, f"EV_{event.type}")
            # KEY 事件: 0=released, 1=pressed, 2=held(autorepeat)
            if event.type == ecodes.EV_KEY:
                code_name = ecodes.KEY.get(event.code, f"KEY_UNKNOWN_{event.code}")
                if isinstance(code_name, list):  # 同 code 多 alias
                    code_name = code_name[0]
                state = {0: "released", 1: "pressed", 2: "held"}.get(event.value, f"v={event.value}")
                print(f"[{ts_str}]  EV_KEY  {code_name:30s}  value={event.value} ({state})")
                stats = key_stats[(event.type, event.code)]
                stats["name"] = code_name
                if event.value == 1:
                    stats["pressed"] += 1
                elif event.value == 0:
                    stats["released"] += 1
                elif event.value == 2:
                    stats["held"] += 1
            elif event.type == ecodes.EV_SYN:
                # syn report, 默默吞掉, 否则刷屏
                continue
            elif event.type == ecodes.EV_MSC:
                # MSC_SCAN — HID scancode, 信息冗余, 静默
                continue
            else:
                # 其它类型 (REL/ABS 等)
                print(f"[{ts_str}]  {type_name}  code={event.code}  value={event.value}")
    except KeyboardInterrupt:
        print()
    except OSError as e:
        # 设备消失 (蓝牙断了)
        print(f"\n[!] 设备读异常 (可能蓝牙断了): {e}")
    finally:
        try:
            dev.close()
        except Exception:
            pass

    elapsed = time.time() - started_at
    print("\n" + "=" * 72)
    print("[summary] 把下面这段贴给模型, 它会据此写 headphone_buttons.py")
    print("=" * 72)
    print(f"  Device path  : {dev.path}")
    print(f"  Device name  : {dev.name}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print(f"  Total events : {total_events}")
    if not key_stats:
        print("  [!] 一个 key 事件都没有. 可能选错了设备 (耳机有时分 HFP/AVRCP 两个), "
              "或耳机按键不走 evdev. 换设备重试.")
    else:
        print("\n  Unique key events:")
        for (etype, ecode), s in sorted(key_stats.items(), key=lambda x: x[0][1]):
            print(
                f"    {s['name']:30s} (EV_KEY code={ecode:4d})  "
                f"pressed={s['pressed']}  released={s['released']}  held={s['held']}"
            )
    print("=" * 72)
    return 0


# ── macOS: pynput ────────────────────────────────────────────────────────

def _mac_main(args) -> int:
    try:
        from pynput import keyboard
    except ImportError:
        print("[error] pynput 未安装. 装一下:", file=sys.stderr)
        print("    uv pip install pynput", file=sys.stderr)
        return 2

    print("\n[pynput] 监听键盘事件 (媒体键经系统传入). 跨多个蓝牙设备聚合, 不可区分.")
    print("         按你想测的所有键 (耳机按键也会触发媒体键). Ctrl+C 停止.\n")
    print("         如果毫无反应: 系统设置 → 隐私与安全 → 辅助功能, 允许当前 terminal.\n")

    key_stats: dict[str, dict] = defaultdict(lambda: {"pressed": 0, "released": 0})
    total = 0
    started_at = time.time()

    def _name(key) -> str:
        if hasattr(key, "name"):
            return f"Key.{key.name}"
        if hasattr(key, "vk"):
            return f"vk={key.vk}"
        return repr(key)

    def on_press(key) -> None:
        nonlocal total
        total += 1
        name = _name(key)
        key_stats[name]["pressed"] += 1
        print(f"[{time.strftime('%H:%M:%S')}] press   {name}")

    def on_release(key) -> None:
        nonlocal total
        total += 1
        name = _name(key)
        key_stats[name]["released"] += 1
        print(f"[{time.strftime('%H:%M:%S')}] release {name}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        while listener.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
    finally:
        listener.stop()

    elapsed = time.time() - started_at
    print("\n" + "=" * 72)
    print("[summary] 把下面这段贴给模型")
    print("=" * 72)
    print(f"  Platform     : macOS (pynput)")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print(f"  Total events : {total}")
    if not key_stats:
        print("  [!] 一个事件都没有. 检查辅助功能授权, 或耳机按键没传到系统层.")
    else:
        print("\n  Unique keys:")
        for name, s in sorted(key_stats.items()):
            print(f"    {name:30s}  pressed={s['pressed']}  released={s['released']}")
    print("=" * 72)
    return 0


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(prog="headphone_buttons_probe", description=__doc__.splitlines()[0])
    ap.add_argument("--device", help="(Linux) /dev/input/event* 路径, 跳过选择")
    args = ap.parse_args()

    if _is_linux():
        return _evdev_main(args)
    if _is_mac():
        return _mac_main(args)
    print(f"[error] 不支持的平台 {sys.platform}.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
