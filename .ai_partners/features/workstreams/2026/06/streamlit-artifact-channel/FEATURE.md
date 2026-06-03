---
title: Streamlit Artifact Channel
status: draft
priority: P2
created: 2026-06-04
updated: 2026-06-04
depends: []
milestone:
description: >-
  Ghost 可控的 Web UI artifact 下发 Channel — Streamlit 作为首个 GUI 后端，Item dispatch + Stream chunk + 单 viewer lock，Ghost 自开网页下发 markdown/chart/code 等预设视觉元素。
---

# Streamlit Artifact Channel

> Channel type: 交互能力 | alpha

## Motivation

Ghost 需要一个"可控画布"——不是 MOSS 运维控制台，不是人类交互界面，而是 Ghost 自己打开、自己控制、自己关闭的 artifact 下发通道。模型在推理过程中决定"我要展示这个图表"、"我要流式输出这篇 markdown"、"我要给用户看这段代码"，然后通过 Channel command 下发到网页。

GitHub Copilot Chat、ChatGPT Canvas 这类产物走的是前端自研+后端紧耦合的路线。MOSS 的路子不同：**不做前端**。用现有 Web 框架的预设视觉元素，通过 Channel 的 item 下发协议控制渲染。Streamlit 是第一个 GUI 后端——不是唯一的，后续 pygame/pyqt6/fastapi/reflex 各有所长。

**为什么是 Streamlit 第一个**：开箱即用，零前端代码，`st.markdown()` / `st.line_chart()` / `st.write_stream()` 直接覆盖 80% 的 artifact 展示需求。GhostOS 已有验证模式（`@st.fragment` + `write_stream` + item dispatch）。

## Design Index

- GhostOS 参考实现: `/Users/BrightRed/Develop/github.com/ghost-in-moss/GhostOS/libs/ghostos/ghostos/prototypes/streamlitapp/`
- MOSS TUI 流式渲染参考: `src/ghoshell_moss/host/tui.py` — `LiveStreamSink` (janus 桥)
- Channel 构建: `moss codex blueprint channel_builder`
- App 体系: `moss docs read app-system.md`

## Key Decisions

### 1. Streamlit = artifact 下发，不做交互

页面不设输入控件（chat_input、button、form）。Ghost 控制一切——开页面、写内容、关页面。用户只是"看"。如果未来需要交互，走 Signal 上行（和 voice App 一个模式）。

**Why**: 交互 = 前端控件设计 + 事件处理 + 状态回环，工程量指数增长。artifact 下发是独立价值点，先闭环。

**Rejected**: 交互式 Web UI（chat 界面等）。那是另一个 feature 的事。

### 2. Item dispatch 协议：{method, args, kwargs} 而非 exec 源码

```python
# Channel 侧下发
_items["main"].append({"method": "markdown", "args": ["## Hello"]})
_items["main"].append({"method": "line_chart", "args": [data], "kwargs": {"x": "col1"}})

# Streamlit 侧渲染
for item in _items["main"]:
    getattr(st, item["method"])(*item["args"], **item.get("kwargs", {}))
```

直接复用 Streamlit 的 public API 方法名，模型凭预训练知识知道 `markdown` / `line_chart` / `code` / `image` 等用法。不需要 eval/exec，安全。

**Why 不是源码下发** (`exec` 用户的 streamlit 代码): 安全问题（任意代码执行），且模型已经知道 Streamlit API——method dispatch 等价于源码下发的表达能力，但安全边界清晰。

**Rejected**: `st.echo` / `exec(code)` 模式。安全性问题 + 错误处理复杂。

### 3. 流式 chunks 走 janus.Queue + st.write_stream

MOSS TUI 的 `LiveStreamSink` 已论证：`janus.Queue` 跨 asyncio/sync 边界，`None` sentinel 标记流结束。Streamlit 侧一模一样：

```python
# Channel 侧 (asyncio)
_streams[stream_id] = janus.Queue()
_streams[stream_id].async_q.put(chunk)

# Streamlit 侧 (sync, fragment 内)
def chunk_gen(q):
    while True:
        chunk = q.sync_q.get()  # 阻塞等
        if chunk is None: break
        yield chunk
st.write_stream(chunk_gen(_streams[stream_id]))
```

**Why janus 而不是 Queue.Queue**: Channel 侧运行在 asyncio event loop，`janus.Queue` 提供 `async_q.put()` 不阻塞 event loop。Streamlit 侧运行在普通线程，用 `sync_q.get()`。

### 4. 单 viewer lock，不搞多 session 广播

一个 Ghost 只有一个"脸"。新浏览器打开同样的端口 → 显示"occupied"或"已被占据，需要抢占吗？"

锁机制：Channel 侧持有一个 `_viewer_lock: str | None`，首次连接的 session 获取 lock。session 断开（WebSocket 关闭 / timeout）→ lock 释放。

**Why**: 广播多 viewer 的正确解是 Zenoh Topic（MOSS 已有），不是 Streamlit 的多 session hack。先单 viewer 闭环，多 viewer 走 Topic 升级。

**Rejected**: 多 session 共享状态（Streamlit 的 session_state 天然 per-session，强制共享需要跨 session 通讯，工程复杂度远超价值）。广播协议（内容中嵌入 `[MSG_END]` sentinel，escaping 问题）。

### 5. st.fragment 做异步容器（GhostOS 已验证）

GhostOS 的 `duplex.py` 和 `async_example.py` 验证了 `@st.fragment` + `while True` 可以做持久化的 poll-and-render 循环。fragment 内的循环不阻塞页面其他部分，可以独立消费 shared state 的更新。

```python
@st.fragment
def render_main():
    while True:
        # drain items from shared state
        for item in drain_items("main"):
            getattr(st, item["method"])(*item["args"], **item.get("kwargs", {}))
        time.sleep(0.1)
```

### 6. 端口管理：固定端口，不动态分配

先上固定端口（8502）。后续如果需要多端口，再引入端口池管理。

**Why**: 单 viewer lock 下不需要多端口。多 Ghost 实例各占不同端口是未来需求。

**Rejected**: 动态端口分配 + 返回 URL。单 viewer 场景不需要。

### 7. App 架构：Channel Provider App

遵循标准 App 入口模式：`Matrix.discover().run(main)`。App 内部两线程：
- Matrix 线程 (asyncio): Channel Runtime，处理 command
- Streamlit 线程 (sync): `streamlit run`，fragment 内 poll shared state

两线程通过 `janus.Queue` + `threading.Lock` 保护共享状态。

**Why Channel Provider 而不是纯进程 App**: Ghost 需要通过 CTML 命令控制页面内容，必须注册 Channel。

## Implementation Notes

- Streamlit 在独立线程启动，不能占用 asyncio event loop。使用 `streamlit.web.cli.main_run` 或 subprocess 方式启动
- `st.session_state` 的 script re-run 持久化：item 列表需要存在 `st.session_state` 上，或通过 thread-safe 的全局变量 + `@st.cache_resource` 管理
- janus.Queue 的 shutdown 需要正确处理——Channel 侧 close 时 `queue.shutdown()`，Streamlit 侧 `sync_q.get()` 抛 `janus.SyncQueueShutDown`
- `st.write_stream` 在一个 fragment 内只能有一个活跃的 stream generator。多个并发 stream（不同容器）需要各自独立的 fragment
- 首次打开页面时 st.rerun 可能会触发 bootstrap 逻辑重复执行——用 `st.session_state` flag 守卫

## Phase Plan

**Phase 1 — 最小可用**:
- App 骨架 (`webui` App, group=`_system` 或 `interaction`)
- Lock 机制
- Item dispatch (`markdown`, `write`, `code`, `image`)
- Stream chunk（单一容器）

**Phase 2**:
- 更丰富的 element 类型（chart, dataframe, audio）
- 多容器（main, sidebar, popover）

**Phase 3 — 可选项**:
- Zenoh Topic 广播多 viewer
- 替换为 FastAPI SSE / Reflex / PyQt6 的平行实现
