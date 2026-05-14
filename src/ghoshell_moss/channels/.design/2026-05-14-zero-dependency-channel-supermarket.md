# 零依赖 Channel 超市 — 设计探索

## 背景

MOSS 架构中 Channel 是能力的组织单位。"零依赖 channel 超市" 是一组只用 MOSS core + Python stdlib 实现的 channel 原型。目标：

1. 为上层 channel (contrib/) 提供最小的通用组件
2. 作为 AI 开发 Channel 的入门模板和测试案例
3. 通过"小实现逐步合并"的方式，降低每个原型的复杂度

## 核心设计策略

### Channel Interface vs As Channel

两种封装思路，在本目录下并存：

**Channel Interface** — 先定义接口，再注册。类自身声明命令方法，`bootstrap()` 中通过 Builder 注册。`Builder.command(interface=...)` 更进一步：可以给模型看虚拟函数签名，实际执行另一个实现体。

**As Channel** — 外部已有的事物包装为 Channel。module、CLI、API、设备等外部能力被反射/封装，不要求被包装者感知 Channel 的存在。

| 原型 | 策略 | 层级 |
|------|------|------|
| `module_channel` | As Channel | L0: 零手工反射 |
| `notebook_channel` | As Channel | L1: 文件系统即笔记本 |
| `mac_channel` | Channel Interface | L1: Builder 注册单个命令 |
| `speech_channel` | Channel Interface | L2: 继承 ABC，封装 contract |
| `typer_channel` | As Channel | L1+: instruction + context |

### 构建梯度

| 层级 | API | 场景 |
|------|-----|------|
| L0 | `new_module_channel()` | 纯函数模块，零手工 |
| L1 | `new_channel()` + Builder | 需 instruction/context/生命周期 |
| L2 | 继承 Channel ABC | 复杂运行时，封装 contract |
| L3 | StatefulChannel | 运行时切换状态/能力集 |
| L4 | PrimeChannel | 全能：stateful + mutable + builder |

## 已完成原型

### module_channel

反射任意 Python module 的函数为 Channel command。`respect_all=True` (默认) 时，有 `__all__` 则尊重，否则 fallback 到 `dir()`。支持 `include`/`exclude` 过滤。证明了反射模式的可行性——一行代码，模块变能力。

测试覆盖：dir() fallback、respect_all、include/exclude、私有函数排除、非 callable 跳过、字符串 import、math 边界。

### notebook_channel

文件系统目录即 notebook。每个文件是一页。支持 write/read/append/list_pages/delete。`context_messages` 自动展示目录树。路径安全：拒绝 `../` 穿透和绝对路径。

测试覆盖：读写基线、追加、列表、删除、路径安全、context 树、嵌套目录、覆盖。

## 从 contrib alpha 学到的事

`ghoshell_moss_contrib/channels/` 下有早期 alpha 实现，分为两类：

**有实现代码的**（依赖外部库，待重做）：
- `mermaid_draw` — webbrowser 打开 mermaid 图表
- `web_bookmark` — 收藏夹 + webbrowser
- `screen_capture` — mss 截图，有 `as_channel()` 模式
- `opencv_vision` — cv2 摄像头，有 `as_channel()` 模式
- `mpv_video` — mpv 播放器控制

**TODO 草图**（等 Beta）：
- `docs_reader` — 文档目录浏览/pin/搜索/FQA
- `project_manager` — GhostOS project 复刻：目录结构/CRUD/gitignore/快速编辑
- `terminal` — 受控 shell + GUI IO + bash 脚本 as command

关键观察：
1. `as_channel()` 方法模式已被多个 alpha 实现采用——一个类内部持有状态，暴露 `as_channel()` 返回包装好的 Channel
2. `project_manager`、`docs_reader`、`notebook_channel` 共享核心能力：对目录的结构化认知。共性可以提取
3. `terminal` 需要审计机制——这是一个横切关注点

## 关键设计洞察

### 文件系统：操作 vs 认知

bash/terminal 已经解决了文件操作。Channel 的价值在于提供文件系统状态的**动态认知**——目录树、文件类型分布、最近变更、gitignore 感知。`notebook_channel` 的 context tree 是这个方向的简化版。`project_manager` 的草图是完整版。二者共享同一个核心：context_messages 展示目录状态。

### 审计作为横切面

terminal 等高危操作需要一个通用的审计机制：一个自解释对象（类似 choice/confirm），包含上下文 id + Future。可插拔的审计组件（最小实现：asyncio 超时 + callback，完整实现：GUI approve/deny）。这不是 terminal 专属——任何高危 command 都可以包一层审计。

### Project Channel 作为父 Channel

`project_manager` 的设计方向是父 channel——管理目录认知 + 文件操作 + git + 审计等子 channel。通过 observe error 退化：当不需要实时行为规划时，退化为 tool 模式（阻塞调用，返回结果）。

### 不要复刻 Agent Tool

凡是 agent 行业已经通过 tool calling 解决的问题，不在 Channel 层复刻。Channel 的独特价值在于：有状态运行时、双工通讯、context_messages 动态认知、CTML 时序规划。简单的一次性工具调用用 module_channel 反射就够。

## 探索路径

零依赖超市的演进顺序（非具体文件路径，而是能力层次）：

1. **反射** — module_channel。最小胶水层，模块即能力。证明了 As Channel 模式
2. **临时笔记** — notebook_channel。目录即笔记本。引入 context_messages 认知
3. **目录认知** — 从 notebook_channel 和 project_manager 的共性中提取：对任意目录提供结构化认知（树、变更、gitignore），不依赖外部工具
4. **审计包装** — 横切面。Future + 上下文 id + 可插拔审计器。让高危操作安全化
5. **组合** — project channel 作为父 channel，组装目录认知 + 文件操作 + 审计 + git 等于 channel。通过 observe error 退化

每一步都是上一步的自然延伸，且每个原型都可独立使用。

## 开放问题

- 审计机制的最小实现应该放在 core 还是 channels 下？如果放在 core，它成为 ChannelRuntime 的内置能力；如果放在 channels，它是一个可选的 wrapper channel
- project channel 的 gitignore 感知：是依赖 `.gitignore` 解析（零依赖，自己写），还是用 git 命令（依赖 git 安装）？
- contrib alpha 的 `as_channel()` 模式是否应该提升为正式约定？还是保持 Builder 装饰器作为主要入口？
