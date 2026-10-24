---
created: 2026-10-24
depends: []
description: 把 src/ghoshell_moss_contrib/moss_in_reachy_mini 迁成一个 moss node， 并顺带起一路
  MJPEG 推流供 screen iframe 消费。
milestone: null
priority: P2
status: in-progress
status_note: 调研完成，决策记录完毕；进入 node 骨架实现阶段
title: Reachymini Node Integration
updated: '2026-10-24'
---

# Reachymini Node Integration

## Motivation

`moss_in_reachy_mini` 当前仍是面向老 apps 体系的 Channel 库——`provide_channel(matrix)`
入口 + `ReachyMiniChannelCreator`。新的 nodes 体系已经是运行时自迭代的落脚点，Ghost
在运行中通过 Matrix 发现、启动、拉入 reachymini 的躯体能力，应该通过"一个 reachy node"
一步接入，而不是把它作为老形态的孤例保留。

与此同步，希望让"Ghost 看得见自己身体当前在做什么"也走通：reachymini 节点启动时
附带一路低帧率 MJPEG 推流，人类侧由 screen 用 iframe 拉入做为常驻窗口。推流本身
不进入 vision 感知链路——它是**身体占位的面**，不是模型的视觉输入。两件事绑在一个
workstream 做，是因为它们共享同一个 reachymini node 进程 / 同一把 SDK 锁。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### D1. 一个 node = 整台 reachymini

reachymini SDK (`ReachyMini()`) 对硬件（头/天线/电机/相机）是一进程一把锁。因此
body channel、media 推流必须在**同一个 node 进程**内共存，不拆分成"reachymini-body"
和"reachymini-camera"两个独立 node。

**Why**：拆开要求 SDK 支持"仅 body / 仅 media"分进程 open，没有证据表明 SDK 允许，
且主 `main.py` 当前默认 `REACHY_MEDIA_BACKEND='no_media'` 的事实说明项目没有做过
双进程验证，不能先假设。

**How to apply**：node 目录命名 `nodes/reachymini/`（或 contrib 下相应位置），
`main.py` 内 `Matrix.discover().run(main)`，`main(matrix)` 同时做两件事——
`provide_channel(reachy.as_channel())` + 起一个 MJPEG server 子任务。

### D2. 推流独立于 vision 感知链路

reachymini 的画面 **不接入 `nodes/visions/stream`**，不走 FaceTopic / watch 门控 /
尾帧那套。推流只是把 `mini.media.get_frame()` 以 MJPEG 吐在
`127.0.0.1:PORT/stream`，供浏览器 iframe 查看。帧率低（例如 5~10 fps 即可），延迟
不敏感。

**Why**：vision 家族的契约是"producer 持设备、无 channel、带 FaceTopic / 授权闸口"；
reachymini 的 node 已经同时承载 body channel 和 SDK 锁，无法照搬该契约。而人类侧
真实需求是 screen 常驻窗口能看见"Ghost 的身体视角"，这是一次纯粹的**占位可视化**，
不是感知输入——如果以后要让模型"看见"，再加一条 `look` 命令随命令结果返回帧
（走命令、不走持续感知）即可，和推流解耦。

**How to apply**：MJPEG server 走 `aiohttp` 或 stdlib `http.server` 都可以；在
`provide_channel` 里 `asyncio.create_task(serve_mjpeg())`，node 退出时取消。
地址在启动日志里 announce，screen 侧用 iframe `<iframe src="http://127.0.0.1:PORT/stream">`
嵌入。

### D3. 视觉命令（`look`）与推流并存但独立

现有 `Vision` 子 channel（`components/vision.py`，`allow_vision=True` 启用）保持
"命令随结果返回帧"的形态——这是像素进历史、能被模型回忆的路径。推流负责"人类
实时看"，`look` 负责"模型按需看"。两条路径都调 `mini.media.get_frame()`，共用同一
设备句柄。

**Why**：vision 家族 README 的"像素走命令"约定（可记忆的"看"必须随命令返回图像
数据）与推流性质正交。推流是人类侧窗口，不进历史；`look` 走命令、进历史。

**How to apply**：`allow_vision` 默认开关由 node 配置决定（env 或启动参数），
与推流启停独立。

### D4. 走 contrib → nodes 的最小耦合

node 的 `main.py` 复用 `ghoshell_moss_contrib.moss_in_reachy_mini.main.provide_channel`
作为底座——老代码本身 API 形态已经和当前 Channel/Matrix 对齐（用的是 `new_prime_channel`
/ `StatefulChannelRuntimeImpl` / `ChannelCreator`，已验证符号存在）。node 入口只做
"加 MJPEG server 子任务 + 调用 provide_channel"的组合，不动 contrib 内部实现。

**Why**：contrib 包的 Channel 组装没有过时成分，没有重构的必要；成本控制在新建
节点骨架 + 推流桥。

**How to apply**：node 目录 ~3 文件（`NODE.md` / `main.py` / `INSTALL.md`），
`main.py` 几十行。

## Exploration paths

### 调研：reachymini 的流**不是公共流**

- `self.mini.media.get_frame()` 是 `ReachyMini` SDK 的 in-process Python 调用，返回
  numpy 帧，SDK **不对外 publish** 任何 RTSP/MJPEG server。即便底层是 GStreamer，
  pipeline 的 sink 没有暴露成网络地址。
- `nodes/visions/camera`：`OpenCVSource` 用 `cv2.VideoCapture(index)` 开 OS 级设备
  （V4L2 / AVFoundation），**认不到 reachymini**。
- `nodes/visions/stream`：ffmpeg ingest 协议地址（RTMP/RTSP/SRT/MJPEG），**需要地址
  才能消费**，reachymini 侧没有现成地址。

因此"用现有 vision node 接入 reachymini"是条死路——除非 reachymini node 自己把帧
再 publish 成一个地址。本 workstream 选择的方案就是让 reachymini node 内部起一个
MJPEG server，但**不接入 vision 家族**（见 D2）。

### 调研：contrib 代码的 API 兼容性

验证过以下符号仍存在于当前主干：
- `ghoshell_moss.new_prime_channel`（`src/ghoshell_moss/__init__.py:41`）
- `ChannelCreator`（`src/ghoshell_moss/core/blueprint/channel_builder.py:855`）
- `StatefulChannelRuntimeImpl`（`src/ghoshell_moss/core/py_channel.py:575`）

contrib 包里 `main.py` 的 import 全部对齐，不需要动。

## Methods

### 迁移骨架（目标形态）

参考 `.moss/system_test_nodes/hello_world/` 作为 node 模板：

```
nodes/reachymini/        # 或放到合适的位置
├── NODE.md              # name/description/singleton:true/exec: python main.py
├── main.py              # Matrix.discover().run(main) + MJPEG server 子任务
└── INSTALL.md           # reachy_mini / reachy_mini_dances_library / env 说明
```

`main.py` 骨架：

```python
async def main(matrix: Matrix):
    # 1. 调 contrib 的 provide_channel（body + 可选 vision）
    # 2. asyncio.create_task(serve_mjpeg(mini, port=...))
    # 3. announce 推流地址给 screen
    # await provide_channel 返回时退出，推流任务被 cancel
```

### 待确认 / 实现时注意

- **emotions 资源路径**：contrib README 提到 `.moss_ws/configs/reachy_mini_emotions/`，
  当前 workspace 已改名 `.moss/`。需检查 `components/body.py` 内是硬编码还是走
  `ws.configs()`。
- **MJPEG server 寄生进程**：server 必须和 `provide_channel` 共享同一个 `ReachyMini`
  实例（SDK 单例锁），因此必须 inline 在 node 进程里，不能另起子进程。
- **帧率预算**：浏览器 iframe 展示用，5~10 fps 够了。编码走 `cv2.imencode('.jpg', ...)`
  或 `PIL.Image.save(buf, 'JPEG')`。
- **screen 侧消费形式**：iframe `src` + 启动时 announce 的地址——screen 节点的
  webview adopt 机制已有，照 `nodes/visions/push` 的 webview 页面 announce 做法参考。

## Implementation Notes

<!-- 实施时填 -->