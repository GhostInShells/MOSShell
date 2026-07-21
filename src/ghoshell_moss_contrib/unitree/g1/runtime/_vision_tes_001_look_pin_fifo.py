"""单测 — vision look 分配 id + FIFO 挤旧 + drop 语义.

## Usage

```
.venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._vision_tes_001_look_pin_fifo
```

## 依赖

`opencv-python Pillow numpy` + 摄像头权限 (macOS 首次弹窗).

## 断言 (退出码 0 = 通过, 非 0 = 失败)

1. 未 start 时 `look()` 返回 None
2. start 后 5s 内能拿到首帧
3. 连续 look 4 次, 关键帧区容量 3 时最旧被 FIFO 挤出
4. drop_pinned(existing_id) 返回 True 且从队列移除
5. drop_pinned(missing_id) 返回 False
6. stop 后 look 返回 None

## 前置

- 摄像头可用
- 无 vision 实例正在运行 (是模块单例, 全局状态)
"""
import sys
import time

from ghoshell_moss_contrib.unitree.g1.runtime import vision


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    try:
        vision.stop()
    except Exception:
        pass
    return 1


def main() -> int:
    # 1. 未 start 时 look 返回 None
    if vision.is_running():
        return _fail("vision already running at test start (dirty state)")
    if vision.look() is not None:
        return _fail("look() before start returned non-None")
    print("  [1] look before start: None (OK)")

    # 2. start + 等首帧
    print("Starting vision (camera=0, fps=2.0, max_pinned=3)...")
    try:
        vision.start(camera_index=0, fps=2.0, max_pinned=3)
    except RuntimeError as e:
        return _fail(f"start raised: {e}")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if vision.peek_latest() is not None:
            break
        time.sleep(0.1)
    else:
        return _fail("no frame within 5s of start")
    print("  [2] first frame within 5s (OK)")

    # 3. 连续 look 4 次, FIFO 应挤掉最早
    ids = []
    for i in range(4):
        p = vision.look(note=f"test-{i}")
        if p is None:
            return _fail(f"look #{i} returned None mid-test")
        ids.append(p.pin_id)
        time.sleep(0.6)  # fps=2.0 保证间隔内有新帧
    print(f"  [3a] 4 looks, ids = {ids}")

    pinned = vision.list_pinned()
    if len(pinned) != 3:
        return _fail(f"expected 3 pinned, got {len(pinned)}")
    remaining_ids = [p.pin_id for p in pinned]
    expected_remaining = ids[-3:]
    if remaining_ids != expected_remaining:
        return _fail(
            f"FIFO ordering wrong. expected {expected_remaining}, got {remaining_ids}"
        )
    print(f"  [3b] FIFO OK: remaining ids {remaining_ids}")

    # 4. drop 中间一个
    middle = remaining_ids[1]
    if not vision.drop_pinned(middle):
        return _fail(f"drop_pinned({middle}) returned False for existing id")
    pinned_after = vision.list_pinned()
    if len(pinned_after) != 2:
        return _fail(f"expected 2 after drop, got {len(pinned_after)}")
    if middle in [p.pin_id for p in pinned_after]:
        return _fail(f"dropped id {middle} still present")
    print(f"  [4] drop existing id OK: remaining {[p.pin_id for p in pinned_after]}")

    # 5. drop 不存在的
    if vision.drop_pinned(999999):
        return _fail("drop_pinned(999999) returned True unexpectedly")
    print("  [5] drop missing id: False (OK)")

    # 6. stop 后 look 返回 None
    vision.stop()
    if vision.is_running():
        return _fail("still running after stop")
    if vision.look() is not None:
        return _fail("look after stop returned non-None")
    print("  [6] look after stop: None (OK)")

    print("PASS: tes_001_look_pin_fifo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
