# 2026-07-23 Desktop GUI 时序流程

## 上下文

在重做数据结构之前，先把核心交互流程画清楚。参与角色：

| 角色 | 进程 | 说明 |
|------|------|------|
| Ghost | Shell 进程 | AI 模型，通过 CTML 调用 desktop channel 命令 |
| CTML Shell | Shell 进程 | 解析 CTML，执行命令 |
| Desktop Channel | Shell 进程 | bash + file_editor 的实际实现 |
| Command Adapter | Shell 进程 | 拦截层——审批闸门，Future 管理，Matrix 事件发布 |
| FutureRouter | Shell 进程 | 进程内 Future 路由 |
| Matrix Session | 网络 | IPC 总线 |
| GUI Node | GUI 进程 | Reflex web server，人类的观察与审批窗口 |
| Human | 浏览器 | 人类用户 |

关键约束：
- Ghost **不知道** GUI 的存在——它只用 `desktop.bash:exec` 等标准命令
- Command Adapter 是 Shell 进程内的透明拦截层
- GUI → Shell 的审批反馈走 Matrix Channel RPC（Shell 暴露 `desktop_gui_approval` channel，GUI 获取 proxy 调用）

---

## Flow 1: GUI Node 启动

```mermaid
sequenceDiagram
    participant Host as MossHost
    participant Matrix as Matrix Session
    participant GUI as GUI Node (Reflex)
    participant Browser as Browser

    Host->>GUI: matrix.run_node("desktop_gui")
    GUI->>GUI: main thread: reflex run
    GUI->>GUI: daemon thread: asyncio.run(main_async())

    rect rgb(240, 248, 255)
        Note over GUI,Matrix: daemon thread — Matrix 集成
        GUI->>Matrix: Matrix.discover()
        Matrix-->>GUI: matrix instance
        GUI->>Matrix: matrix.session.sub_topic("desktop/commands", on_command_event)
        GUI->>Matrix: matrix.session.sub_topic("desktop/command-update", on_command_update)
        GUI->>Matrix: mesh = await matrix.mesh()
        GUI->>Matrix: approval_channel = mesh.get_channel_proxy("desktop_gui_approval")
        GUI->>GUI: store approval_channel for later RPC calls
    end

    rect rgb(255, 248, 240)
        Note over GUI,Browser: main thread — Reflex UI
        GUI->>Browser: WebSocket connect
        GUI->>Browser: render dual-pane layout
        Browser-->>Human: Desktop GUI ready
    end
```

要点：
- daemon 线程跑 Matrix asyncio，主线程跑 Reflex
- GUI 订阅两个 topic：`desktop/commands`（新命令）和 `desktop/command-update`（状态变更）
- GUI 获取 `desktop_gui_approval` channel 的 proxy，用于后续审批 RPC
- 线程间通过 `asyncio.Queue` 或直接操作 Reflex State（线程安全）来传递事件

---

## Flow 2: 命令需要审批（happy path）

Ghost 执行 `desktop.file_editor:str_replace(path="/etc/config", old_str="...", new_str="...")`，
Adapter 判定需要审批。

```mermaid
sequenceDiagram
    participant Ghost
    participant CTML as CTML Shell
    participant Adapter as Command Adapter
    participant Future as FutureRouter
    participant Desktop as Desktop Channel
    participant Matrix as Matrix Session
    participant GUI as GUI Node
    participant Human

    Ghost->>CTML: desktop.file_editor:str_replace(...)
    CTML->>Adapter: intercept(command)

    rect rgb(255, 240, 240)
        Note over Adapter,Future: 审批闸门 — 创建 Future，发布事件
        Adapter->>Adapter: approval_policy.check(command) → requires_approval=True
        Adapter->>Future: create(command_id) → (future_id, Future)
        Adapter->>Matrix: pub_topic("desktop/commands", {id, status:"awaiting_approval", ...})
        Adapter->>Adapter: await Future (non-blocking for CTML stream)
    end

    rect rgb(240, 255, 240)
        Note over Matrix,GUI: GUI 接收并渲染
        Matrix-->>GUI: on_command_event(event)
        GUI->>GUI: state.commands.append(CommandRecord(status="awaiting_approval"))
        GUI-->>Human: sidebar: orange pulsing dot + command summary
        Human->>GUI: click command
        GUI-->>Human: detail panel: diff view + [Approve] [Reject] buttons
    end

    rect rgb(240, 240, 255)
        Note over Human,Adapter: 人类审批 — GUI RPC → Shell
        Human->>GUI: click [Approve]
        GUI->>GUI: approval_channel.approve(id, reason="")
        GUI->>Matrix: RPC call to desktop_gui_approval:approve
        Matrix->>Adapter: approve(command_id, reason)
        Adapter->>Future: resolve(future_id, "approved")
    end

    rect rgb(255, 255, 240)
        Note over Adapter,Desktop: 执行命令
        Adapter->>Desktop: execute original command
        Desktop->>Desktop: actually edit the file
        Desktop-->>Adapter: result: "File edited: 2 replacements"
        Adapter->>Matrix: pub_topic("desktop/command-update", {id, status:"completed", result})
        Adapter-->>CTML: return result
        CTML-->>Ghost: "File edited: 2 replacements"
    end

    rect rgb(240, 255, 240)
        Note over Matrix,Human: GUI 更新
        Matrix-->>GUI: on_command_update(event)
        GUI->>GUI: update CommandRecord(status="completed", result=...)
        GUI-->>Human: green solid dot, result shown in detail
    end
```

要点：
- Adapter 在 `await Future` 时不阻塞 CTML 流——其他 channel 的命令继续执行
- Ghost 的体验：调用了 `desktop.file_editor:str_replace`，等了一会儿，拿到结果。不知道审批发生过
- 审批决策通过 Matrix Channel RPC 从 GUI 传回 Shell

---

## Flow 3: 命令无需审批（auto-pass）

Ghost 执行 `desktop.bash:exec("ls -la")`，Adapter 判定无需审批，直接执行。

```mermaid
sequenceDiagram
    participant Ghost
    participant CTML as CTML Shell
    participant Adapter as Command Adapter
    participant Desktop as Desktop Channel
    participant Matrix as Matrix Session
    participant GUI as GUI Node
    participant Human

    Ghost->>CTML: desktop.bash:exec("ls -la")
    CTML->>Adapter: intercept(command)

    rect rgb(255, 255, 240)
        Note over Adapter: 无需审批 — 直接执行
        Adapter->>Adapter: approval_policy.check(command) → requires_approval=False
        Adapter->>Matrix: pub_topic("desktop/commands", {id, status:"running", ...})
        Adapter->>Desktop: execute immediately
        Desktop->>Desktop: run ls -la
        Desktop-->>Adapter: result: stdout string
    end

    Adapter->>Matrix: pub_topic("desktop/command-update", {id, status:"completed", result})
    Adapter-->>CTML: return result
    CTML-->>Ghost: "total 42\ndrwxr-xr-x ..."

    Matrix-->>GUI: on_command_event → on_command_update
    GUI->>GUI: add+update CommandRecord(status="completed")
    GUI-->>Human: see completed command in sidebar (for observation only)
```

要点：
- 无需审批时，命令直接执行，但仍然发布到 topic 供 GUI 展示
- Human 看到命令在 sidebar 中出现并快速完成（绿色灯）
- 这是"观察"而非"审批"——人类只看到 Ghost 做了什么

---

## Flow 4: 人类拒绝

```mermaid
sequenceDiagram
    participant Ghost
    participant CTML as CTML Shell
    participant Adapter as Command Adapter
    participant Future as FutureRouter
    participant Matrix as Matrix Session
    participant GUI as GUI Node
    participant Human

    Ghost->>CTML: desktop.bash:exec("rm -rf /important")
    CTML->>Adapter: intercept(command)
    Adapter->>Adapter: approval_policy → requires_approval=True
    Adapter->>Future: create(command_id) → Future
    Adapter->>Matrix: pub_topic("desktop/commands", {id, status:"awaiting_approval"})

    Matrix-->>GUI: render command with orange dot
    Human->>GUI: click [Reject], type reason: "don't delete that!"
    GUI->>Matrix: RPC: desktop_gui_approval:reject(id, "don't delete that!")

    Matrix->>Adapter: reject(command_id, reason)
    Adapter->>Future: reject(future_id, reason)

    rect rgb(255, 230, 230)
        Note over Adapter,Ghost: 命令不执行，直接返回拒绝
        Adapter->>Matrix: pub_topic("desktop/command-update", {id, status:"rejected", human_reply:"don't delete that!"})
        Adapter-->>CTML: raise ObserveError("rejected: don't delete that!")
        CTML-->>Ghost: signal: command rejected with reason
    end

    Matrix-->>GUI: update CommandRecord(status="rejected")
    GUI-->>Human: red solid dot + rejection reason shown
```

要点：
- 命令**从未执行**——Adapter 在 Future 被 reject 后直接返回错误
- Ghost 收到的是 `ObserveError`（信号），不是普通返回值——Ghost 可以决定怎么回应
- Ghost 可以解释原因（通过下一轮 CTML），这就是审批即对话

---

## Flow 5: 人类追问（审批即对话）

```mermaid
sequenceDiagram
    participant Ghost
    participant Adapter as Command Adapter
    participant Matrix as Matrix Session
    participant GUI as GUI Node
    participant Human

    Note over Ghost,Human: 前置：命令已处于 awaiting_approval（同 Flow 2 前半段）

    rect rgb(255, 245, 235)
        Note over Human,Ghost: 追问回路
        Human->>GUI: type: "why are you editing this file?"
        GUI->>Matrix: RPC: desktop_gui_approval:dialogue(id, "why are you editing this file?")
        Matrix->>Adapter: dialogue(command_id, message)

        Adapter->>Adapter: wrap as Signal
        Adapter-->>Ghost: signal: human_dialogue {command_id, message}

        Ghost->>Ghost: think about human's question
        Ghost->>CTML: desktop.bash:exec("cat /etc/config") — show human the context
        Note over Ghost,CTML: Ghost 可以执行其他命令来回应人类

        Ghost->>CTML: response signal: "I need to update the port config because..."
        CTML-->>Human: Ghost's explanation appears in GUI (as output/signal)
    end

    rect rgb(240, 255, 240)
        Note over Human,Ghost: 人类做出决定
        Human->>Human: reads explanation, understands
        Human->>GUI: click [Approve]
        GUI->>Matrix: RPC: approve(id, "ok, go ahead")
        Matrix->>Adapter: approve(command_id, "ok, go ahead")
    end

    Note over Adapter,Ghost: 后续同 Flow 2 执行阶段
```

要点：
- 追问不改变命令的 `awaiting_approval` 状态——它仍然是 pending
- Ghost 在等待期间可以继续执行其他 CTML 命令来解释和回应
- 这实现了"审批即对话"——不是 yes/no 的二元闸门，而是人类与 Ghost 在 desktop 知觉空间中的协作对话
- 追问消息本身不是 command，是 signal——它不阻塞任何 channel 的 FIFO

---

## Flow 6: 超时 / Stale

```mermaid
sequenceDiagram
    participant Adapter as Command Adapter
    participant Future as FutureRouter
    participant Matrix as Matrix Session
    participant GUI as GUI Node
    participant Human

    Note over Adapter: 命令 awaiting_approval 超过阈值（如 5 分钟）

    Adapter->>Adapter: timeout detected
    Adapter->>Future: cancel(future_id)
    Adapter->>Matrix: pub_topic("desktop/command-update", {id, status:"stale"})

    Matrix-->>GUI: update CommandRecord(status="stale")
    GUI->>GUI: if show_stale==False: hide from sidebar
    GUI-->>Human: command disappears (unless "Show stale" is toggled)

    Note over Human: 人类可以 toggle "Show stale" 重新看到被取消的命令（只读，不可操作）
```

---

## 对数据结构设计的启示

从这些流程中，命令的生命周期状态机可以确定为：

```
pending → running → (awaiting_approval) → approved → completed
                    ↓                    ↓
                    stale               rejected → (stale after timeout)
                                        ↓
                                       error
```

其中 `awaiting_approval` 是可选的——取决于 approval_policy。

### 关键实体分离

画完图后，可以清楚看到三种实体：

1. **CommandInvocation** — Ghost 发出的命令本身
   - 包含：id, channel_path, command_name, params, status, result
   - 生命周期：贯穿整个流程

2. **ApprovalRequest** — 审批闸门（可选，只有需要审批的命令才有）
   - 包含：command_id, future_id, prompt, status (pending/approved/rejected/stale)
   - 生命周期：从 `awaiting_approval` 到 `approved`/`rejected`/`stale`

3. **DialogueMessage** — 追问消息（可选）
   - 包含：command_id, sender (human/ghost), message, timestamp
   - 关联到审批请求，但不改变审批状态

前两种实体是 1:0..1 的关系，第三种是 1:N 的关系。

### 线程与通信模型

```
Shell 进程                         GUI 进程
┌──────────────────┐              ┌──────────────────────┐
│  Desktop Channel │              │  Reflex (main thread)│
│       ↓          │              │        ↑             │
│  Command Adapter │──Matrix────▶│  DesktopState        │
│       ↓          │  topic       │        ↓             │
│  FutureRouter    │              │  Sidebar + Detail    │
│       ↑          │              │        ↓             │
│  desktop_gui_    │◀──Matrix────│  Human clicks        │
│  approval chan   │  RPC         │  (approve/reject/    │
│                  │              │   dialogue)          │
└──────────────────┘              └──────────────────────┘
```

- Shell → GUI：Matrix topic（命令事件广播）
- GUI → Shell：Matrix Channel RPC（审批反馈）
- GUI 内部：daemon 线程 Matrix asyncio ↔ Reflex State（asyncio.Queue 或直接写入）

当前记录者视角：

画完这六个流程后，数据结构的设计方向变得清晰了。核心洞察：

1. `CommandRecord` 不应该是一个"万能 record"——它需要分成 `CommandInvocation` 和 `ApprovalRequest`
2. 审批是可选的——大部分 `desktop.bash:exec("ls")` 不需要审批，不需要创建 ApprovalRequest
3. 追问是一个独立的消息线程——不是 command，不是 approval decision，是对话
4. 现有的 `FutureRouter` 完美匹配"创建 Future → 等待 → resolve/reject"的模式，不需要重新发明
5. GUI Node 只需要持有 `desktop_gui_approval` channel 的 proxy，通过 RPC 回传审批结果——这比 pub/sub 更清晰
