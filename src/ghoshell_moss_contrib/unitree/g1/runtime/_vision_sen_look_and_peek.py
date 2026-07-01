"""Mac 场景快测 — vision runtime look + peek 双路径.

## Usage

```
.venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._vision_sen_look_and_peek
```

## 依赖

```
pip install opencv-python Pillow numpy
```

macOS 首次运行会弹窗请求摄像头权限. 允许 Terminal / iTerm / IDE. 若之前拒绝过,
在 系统偏好设置 → 隐私 → 摄像头 手动打开.

## 场景

起 vision runtime, fps=1.0, max_pinned=3. 后台子线程持续抓最新一帧.
主循环等待用户指令:

- `l [note]` → look 抓一帧关键帧, 打印 pin id
- `p`        → list_pinned, 打印当前关键帧区全部 id / note / t
- `d <id>`   → drop_pinned, 主动释放某 id
- `e`        → enable_context (睁眼)
- `x`        → disable_context (闭眼, look 仍可用)
- `h`        → health 打印
- `.`        → 打印当前 peek_latest (single frame snapshot)
- `q`        → 退出

## 预期

- 首次 `.` 应 print latest frame size (640, 480) + age ~1s 以内
- `l wave` 后 `p` 应看到 id=1, note='wave', t=~当前时间
- 连续 look 4 次, `p` 只剩最后 3 张 (FIFO 挤了 id=1)
- `x` 后 `.` 仍返回帧 (子线程仍跑), `p` 仍有内容, `l` 仍工作
- 退出后无残留进程 (摄像头释放)

## 安全要点

无 — 纯输入设备, 不控制机械. 可任意退出.
"""
import sys
import time

from ghoshell_moss_contrib.unitree.g1.runtime import vision


def _print_pinned() -> None:
    pinned = vision.list_pinned()
    print(f"  Pinned ({len(pinned)}/{vision.health()['max_pinned']}):")
    for p in pinned:
        print(f"    id={p.pin_id}, t={p.t:.2f}, note={p.note!r}, size={p.image.size}")


def _print_latest() -> None:
    latest = vision.peek_latest()
    if latest is None:
        print("  no frame yet")
        return
    image, t = latest
    age = time.monotonic() - t
    print(f"  latest: size={image.size}, t={t:.2f}, age={age:.2f}s, "
          f"context_enabled={vision.is_context_enabled()}")


def _print_health() -> None:
    for k, v in vision.health().items():
        print(f"  {k}: {v}")


def main() -> int:
    print("=== vision sen: look + peek ===")
    print("Starting vision (camera=0, fps=1.0, max_pinned=3)...")
    try:
        vision.start(camera_index=0, fps=1.0, max_pinned=3)
    except RuntimeError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("Started. Wait 1-2s for first frame, then:")
    print("  l [note] / p / d <id> / e / x / h / . / q")

    try:
        while True:
            print("cmd> ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            cmd = line[0]
            arg = line[1:].strip()

            if cmd == "q":
                break
            elif cmd == ".":
                _print_latest()
            elif cmd == "p":
                _print_pinned()
            elif cmd == "l":
                result = vision.look(note=arg)
                if result is None:
                    print("  look: no frame available (camera warming up?)")
                else:
                    print(f"  look: pinned id={result.pin_id}, t={result.t:.2f}, "
                          f"note={result.note!r}")
            elif cmd == "d":
                try:
                    pin_id = int(arg)
                except ValueError:
                    print("  usage: d <int_id>")
                    continue
                ok = vision.drop_pinned(pin_id)
                print(f"  drop id={pin_id}: {'OK' if ok else 'not found'}")
            elif cmd == "e":
                vision.enable_context()
                print(f"  context_enabled = {vision.is_context_enabled()}")
            elif cmd == "x":
                vision.disable_context()
                print(f"  context_enabled = {vision.is_context_enabled()}")
            elif cmd == "h":
                _print_health()
            else:
                print(f"  unknown cmd: {cmd!r}")
    except KeyboardInterrupt:
        print("\n(Ctrl+C)")
    finally:
        print("Stopping vision...")
        vision.stop()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
