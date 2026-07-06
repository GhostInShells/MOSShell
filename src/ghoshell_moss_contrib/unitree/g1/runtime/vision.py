"""G1 Vision Runtime — 单帧最新 + 关键帧区 双路径视觉感知 (L3, runtime 层).

## 依赖 (使用前必装)

```
pip install opencv-python Pillow numpy
```

Mac 首次运行会请求摄像头权限 (系统偏好设置 → 隐私 → 摄像头, 允许当前终端 / IDE).
Jetson 上若 cv2.VideoCapture(0) 直出失败, 参考本模块 "硬件路径 fallback" 节.

## 双路径设计

vision 对 LLM 暴露两条路径, 严格分立:

### 路径 A — 被动持续观察 (context_messages 里的"当前一帧")

- 子线程以低频 (默认 1Hz) 抓帧, 只留最新一帧 (覆盖式 latest snapshot)
- Channel context_messages 每次刷新调 `peek_latest()`, 拿当前最新帧
- 不做滑动窗口, 不做 drain — 图片 token 压力大 + 首 token 延迟敏感, 单帧够用
- LLM 每 tick 看到"眼下这一刻在看什么", 帧的时序连续性靠 LLM 通过多轮 tick 自建
- 可通过 `disable_context()` 显式关掉 (子线程仍跑, look 仍工作) — 相当于闭眼

### 路径 B — 主动抓拍钉帧 (look → 关键帧区)

- LLM 通过 CTML 命令 `<vision:look note="..."/>` 主动抓一帧
- runtime `look()` 分配自增 id, 帧进关键帧区 (FIFO 容量默认 3), 命令返回 id
- 命令建议标记为 `always_observe=True` — 抓完必然触发下一轮观察, 不然 look 完不看等于发呆
- 关键帧区里的图片会在下一轮 context_messages 里出现, 每张带 "id=X, note=..., t=..."
  的 text block 前置说明, LLM 通过 id 引用建立 "历史里的 id 提及 ↔ 当前 context 里的图" 的关联
- 满 FIFO 挤旧, id 一旦被挤就再也看不到 — instruction 明说 "context 里没找到你引用的 id
  = 那一帧已被忘记"

## 为什么图片不进对话历史 (关键设计原理)

图片放历史里, 越往后越 stale — LLM 越来越难关联到"这张图当时讲的什么", 语义粘性快速衰减.
且每轮 tick 都灌一堆历史图片, 首 token 响应速度线性下降.

**id 引用 + 固定区块**是解法: 图片只出现在 context_messages 的固定位置, 数量有上限, 历史里
只留下 LLM 自写的 "id=X, 用户在挥手" 这样的语义锚点. 语义锚点靠 LLM 自己的文字比图片本身
更耐 stale.

context_messages 呈现顺序建议 (channel 层实现):

```
[关键帧区]
  Message(text): "关键帧 id=1, 记录: 'user waving', t=123.4"
  Message(image): <图>
  Message(text): "关键帧 id=2, 记录: 'wave arm', t=125.7"
  Message(image): <图>
  ...
[被动最新一帧]
  Message(text): "当前视野, t=130.2"
  Message(image): <图>
```

Instruction 应当说明:
- 图片以 "id=X, ..." text block 前置引出, 图紧跟其后
- 你在历史里引用了某 id 但 context 里没找到 = 那一帧已被忘记
- 关键帧区容量有限, 想留就 look, 会挤掉最旧
- 使用 <vision:look note="..."/> 抓拍, note 给这张图一个语义标签帮你后续记忆
- <vision:drop_pin id="..."/> 主动释放关键帧位

## 与 vision app (.moss_ws/apps/sensors/vision/main.py) 对照

Mac AVFoundation 版本的同范式 app. 本模块借鉴 capture 结构, 但:

- 该 app 通过 Matrix 跨进程, 本模块 G1 进程内直接子线程, 不依赖 Matrix
- 该 app context_messages 只有单帧 latest_frame, 本模块加了关键帧区双路径
- 该 app `capture()` 命令只更新 `_latest_frame` 不返回图, 本模块 `look()` 返回 id 供
  历史里 LLM 引用

## 硬件路径 fallback (Jetson 上实测确认)

start() 内部按序尝试:
1. `cv2.VideoCapture(camera_index)` 直出 — USB 摄像头通常成立
2. (若失败, 未实装) GStreamer V4L2 / Jetson CSI pipeline

本期 Mac 场景先跑 (1), Jetson 上失败再加 (2)/(3), 不预写.

## 建议的 channel 层实现 (草稿, 集成时定稿)

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel, Message, CommandUtil
from ghoshell_moss_contrib.unitree.g1.runtime import vision

channel = new_channel(name="vision", description="视觉感知 channel")


@channel.build.command(always_observe=True)
async def look(note: str = "") -> str:
    '''抓一张关键帧钉入 context. 返回 id.
    note: 给这张图一个短标签, 帮你在历史里回忆自己 look 过什么.
    '''
    pin = vision.look(note=note)
    if pin is None:
        return "vision: no frame available (camera warming up or disabled)"
    return f"vision: pinned id={pin.pin_id}, t={pin.t:.2f}, note={pin.note!r}"


@channel.build.command()
async def drop_pin(pin_id: int) -> str:
    '''主动释放一个关键帧, 让位给新的 look.'''
    ok = vision.drop_pinned(pin_id)
    return f"vision: dropped id={pin_id}" if ok else f"vision: id={pin_id} not in pinned"


@channel.build.command()
async def enable() -> str:
    '''睁眼: 每轮 context 附上当前视野一帧.'''
    vision.enable_context()
    return "vision: context enabled"


@channel.build.command()
async def disable() -> str:
    '''闭眼: 停止在 context 附当前视野. look 仍可用.'''
    vision.disable_context()
    return "vision: context disabled"


@channel.build.context_messages
async def _context():
    parts = []
    # 关键帧区
    pinned = vision.list_pinned()
    for p in pinned:
        parts.append(Message.new().with_content(
            f"[vision] 关键帧 id={p.pin_id}, note={p.note!r}, t={p.t:.2f}"
        ))
        parts.append(Message.new().with_content(p.image))
    # 被动最新一帧
    if vision.is_context_enabled():
        latest = vision.peek_latest()
        if latest is not None:
            image, t = latest
            parts.append(Message.new().with_content(f"[vision] 当前视野, t={t:.2f}"))
            parts.append(Message.new().with_content(image))
    if not parts:
        parts.append(Message.new().with_content(
            "[vision] no frames yet (camera warming up or disabled)"
        ))
    return parts


@channel.build.instruction
def _instruction():
    return (
        "vision channel — 视觉感知.\\n"
        "\\n"
        "图片以 'id=X, ...' text block 前置引出, 图紧跟其后. "
        "若你在历史中引用了某 id 但当前 context 里没找到, 那一帧已被忘记, 无法回看.\\n"
        "\\n"
        "关键帧区容量有限 (FIFO), 用 look 抓的新帧会挤掉最旧一张. "
        "想给某帧较长记忆, 在 note 里写下语义锚点, 帮你后续对话中回忆.\\n"
    )
```

具体格式和 instruction 措辞集成时可调, 上面是起点.

## 待实测 / 待回填

- Mac 场景脚本首次运行是否顺利拿到摄像头 (macOS 权限)
- Jetson (G1 PC2) 上 cv2.VideoCapture 直出是否成立, 否则补 GStreamer fallback
- fps=1.0 单帧 640x480 的首 token 延迟增量 (跟部署 LLM 挂钩), 决定 default 是否要调
- max_pinned 默认 3 是否够, 实际交互中 LLM 通常会 pin 到几张
- disable_context 后子线程是否需要停 (省 CPU vs 保持预热), 目前默认不停
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("moss.g1.runtime.vision")

# 延迟依赖 import — 允许 docstring / 接口反射在无 cv2 环境下仍可读
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    _DEPS_OK = True
    _DEPS_ERR = ""
except ImportError as e:
    _DEPS_OK = False
    _DEPS_ERR = str(e)
    Image = None  # type: ignore


# ── 数据结构 ──

@dataclass(frozen=True)
class Pinned:
    """关键帧区里的一帧记录."""
    pin_id: int
    image: object   # PIL.Image.Image (惰性类型避免无依赖环境 import 失败)
    t: float        # monotonic seconds
    note: str


# ── 模块级私有状态 ──

_cap = None
_cap_lock = threading.Lock()

_thread: Optional[threading.Thread] = None
_running = threading.Event()

_latest_frame = None            # PIL.Image | None
_latest_t: float = 0.0
_latest_lock = threading.Lock()

_pin_id_counter: int = 0
_pinned: deque = deque(maxlen=3)
_pin_lock = threading.Lock()

_context_enabled: bool = True

_config = {
    "camera_index": 0,
    "fps": 1.0,
    "resolution": (640, 480),
    "max_pinned": 3,
}


# ── 生命周期 ──

def start(
    camera_index: int = 0,
    fps: float = 1.0,
    resolution: tuple[int, int] = (640, 480),
    max_pinned: int = 3,
    context_enabled: bool = True,
) -> None:
    """启动 vision 子线程. 幂等 — 已 running 时直接 return.

    :param camera_index: cv2.VideoCapture 索引 (Mac 通常 0)
    :param fps: 子线程抓帧频率 (Hz). 默认 1.0 — 图片 token 敏感, 无需高频.
    :param resolution: (width, height). 默认 640x480 (medium), 不追高清.
    :param max_pinned: 关键帧区容量 (FIFO). 默认 3.
    :param context_enabled: 启动时是否开启被动帧灌入 context (True=睁眼).
    :raises RuntimeError: 依赖缺失或摄像头打开失败.
    """
    global _thread, _pinned, _context_enabled, _pin_id_counter
    if not _DEPS_OK:
        raise RuntimeError(
            f"vision runtime deps missing: {_DEPS_ERR}. "
            f"Install: pip install opencv-python Pillow numpy"
        )
    if _running.is_set():
        logger.debug("vision.start called but already running, ignored")
        return

    _config["camera_index"] = camera_index
    _config["fps"] = max(0.1, fps)
    _config["resolution"] = resolution
    _config["max_pinned"] = max(1, max_pinned)
    _context_enabled = context_enabled

    with _pin_lock:
        _pinned = deque(maxlen=_config["max_pinned"])
        _pin_id_counter = 0

    if not _open_camera(camera_index, resolution):
        raise RuntimeError(
            f"vision: failed to open camera index={camera_index}. "
            f"On macOS, check camera permission (系统偏好设置 → 隐私 → 摄像头). "
            f"On Jetson, may need GStreamer pipeline fallback (see module docstring)."
        )

    _running.set()
    _thread = threading.Thread(
        target=_capture_loop,
        daemon=True,
        name="moss.g1.vision.capture",
    )
    _thread.start()
    logger.info(
        "vision started: camera=%d, fps=%.2f, resolution=%s, "
        "max_pinned=%d, context_enabled=%s",
        camera_index, fps, resolution, max_pinned, context_enabled,
    )


def stop(timeout: float = 2.0) -> None:
    """停止子线程, 释放摄像头. 幂等."""
    global _thread, _latest_frame
    if not _running.is_set():
        return
    _running.clear()
    if _thread is not None:
        _thread.join(timeout=timeout)
        _thread = None
    _release_camera()
    with _latest_lock:
        _latest_frame = None
    logger.info("vision stopped")


def is_running() -> bool:
    return _running.is_set()


# ── 摄像头 (内部) ──

def _open_camera(camera_index: int, resolution: tuple[int, int]) -> bool:
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
        _cap = cv2.VideoCapture(camera_index)
        if not _cap.isOpened():
            _cap = None
            return False
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        return True


def _release_camera() -> None:
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None


def _grab_frame():
    """抓一帧, 返回 PIL.Image 或 None."""
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            return None
        ret, frame = _cap.read()
        if not ret:
            return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ── 子线程主循环 ──

def _capture_loop() -> None:
    global _latest_frame, _latest_t
    period = 1.0 / _config["fps"]
    logger.info("vision capture loop started (period=%.2fs)", period)
    while _running.is_set():
        t_start = time.monotonic()
        try:
            frame = _grab_frame()
            if frame is not None:
                with _latest_lock:
                    _latest_frame = frame
                    _latest_t = time.monotonic()
        except Exception:
            logger.exception("vision capture loop error, continuing")
            time.sleep(0.1)
            continue
        elapsed = time.monotonic() - t_start
        remain = period - elapsed
        if remain > 0:
            time.sleep(remain)
    logger.info("vision capture loop exited")


# ── 被动路径 — 只读 ──

def peek_latest():
    """返回 (PIL.Image, t_monotonic) 或 None.

    channel context_messages 用. 不消费. 每次返回当前最新帧的引用副本.
    disable_context 状态下**仍返回**帧 — channel 层负责根据 is_context_enabled
    决定是否喂给 LLM. 这样 look 在 disable 下仍能工作.
    """
    with _latest_lock:
        if _latest_frame is None:
            return None
        return (_latest_frame.copy(), _latest_t)


# ── 主动路径 — 抓拍钉帧 ──

def look(note: str = ""):
    """抓当前最新一帧, 分配自增 id, 加入关键帧区 (FIFO 挤旧).

    返回 Pinned(pin_id, image, t, note) 或 None (无可用帧).
    channel look command 用. 建议 always_observe=True (抓了不看等于发呆).

    :param note: 语义标签, 帮 LLM 在历史里回忆自己 look 过什么.
    """
    global _pin_id_counter
    latest = peek_latest()
    if latest is None:
        return None
    image, t = latest
    with _pin_lock:
        _pin_id_counter += 1
        pin_id = _pin_id_counter
        pinned = Pinned(pin_id=pin_id, image=image, t=t, note=note)
        _pinned.append(pinned)
    return pinned


def list_pinned() -> list:
    """快照返回关键帧区当前全部 Pinned (FIFO 顺序). 不影响 deque."""
    with _pin_lock:
        return list(_pinned)


def drop_pinned(pin_id: int) -> bool:
    """主动移除一个关键帧. 返回是否找到."""
    with _pin_lock:
        for pinned in list(_pinned):
            if pinned.pin_id == pin_id:
                _pinned.remove(pinned)
                return True
    return False


# ── 显式开关 ──

def enable_context() -> None:
    """睁眼 — channel 层 context_messages 应喂被动最新一帧."""
    global _context_enabled
    _context_enabled = True


def disable_context() -> None:
    """闭眼 — channel 层 context_messages 应跳过被动帧. look 仍可用."""
    global _context_enabled
    _context_enabled = False


def is_context_enabled() -> bool:
    return _context_enabled


# ── health ──

def health() -> dict:
    with _latest_lock:
        last_t = _latest_t
        has_frame = _latest_frame is not None
    with _pin_lock:
        pin_count = len(_pinned)
        next_id = _pin_id_counter + 1
    now = time.monotonic()
    return {
        "running": _running.is_set(),
        "context_enabled": _context_enabled,
        "fps": _config["fps"],
        "resolution": _config["resolution"],
        "camera_index": _config["camera_index"],
        "max_pinned": _config["max_pinned"],
        "pinned_count": pin_count,
        "pin_id_next": next_id,
        "has_latest_frame": has_frame,
        "last_frame_age_s": (now - last_t) if has_frame else -1.0,
    }
