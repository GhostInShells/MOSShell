# Subprocess Sandbox Eval Protocol: 持久化 REPL 作为 AI 能力接口

**日期**: 2026-06-09
**来源**: 人类工程师与 Claude Opus 4.7 在 Playwright App 开发中提炼
**关联**: `module-eval-channel` feature, MOSS 原论文 (arXiv:2409.16120)

## 问题

ModuleEvalChannel 的设计目标是：让 AI 直接写 Python 代码控制领域对象（Playwright 浏览器、pandas DataFrame、ROS 节点），而非通过预定义的 Command 函数封装。Channel 是薄壳，Module 源码就是 instruction。

核心挑战：**如何安全、可靠地执行 AI 写的代码，同时让领域对象的副作用正常产生？**

对于 Playwright——一个需要在独立线程运行 sync API 的有状态长生命周期对象——这个问题被放大为：Sandbox 在什么进程/什么线程执行？

## 探索与结论

三种方案被逐一排除：

| 方案 | 问题 | 根因 |
|------|------|------|
| Sandbox.exec() 在 async handler 内 | `playwright._impl._errors.Error: using Playwright Sync API inside the asyncio loop` | Playwright sync API 检测到运行的 event loop 后拒绝初始化 |
| pexpect `python -i` REPL | 多行代码 echo 与 prompt 匹配歧义，输出不可靠 | Python REPL 按行 echo 输入，每行产生 prompt，pexpect 匹配第一行就停止 |
| Janus 桥（queue.Queue + threading.Event） | 仍未解决 asyncio 冲突（初始化在 event loop 线程），且线程模型脆弱 | Playwright 构造函数本身检测 event loop |

**选择**：子进程隔离。父进程（Channel, async）通过 subprocess.Popen 启动子进程（Eval Server, sync），stdin/stdout 管道通讯，JSON-line 协议。

这是一个正确性决策，不是性能决策。正确的来源是：Playwright sync API 的约束无法通过线程模型的调整满足，因为约束作用于构造函数而不仅是方法调用。子进程提供了干净的"无 event loop 环境"。

## 架构

```
Parent Process (async event loop)       Child Process (sync, no event loop)
─────────────────────────────────       ─────────────────────────────────────
Channel (main.py)                       eval_server.py
  EvalServer.__init__()                   import playwright
    Popen(sys.executable,                   sync_playwright().start()
           eval_server.py)                  chromium.launch()
    stdout.readline() ← "ready"             Sandbox(name, builtins=None)
                                            sandbox.set("page", page)
  exec command:                              sandbox.set("json", json)
    server.send(text__)                     Sandbox(parent, SANDBOX_BUILTINS)
      stdin → JSON-line                     print("ready")
      stdout ← JSON-line
                                            while True:
  close:                                        request = stdin.readline()
    stdin → {"code":"__SHUTDOWN__"}              result = sandbox.exec(code)
    process.wait()                               stdout.write(json(result))
                                              browser.close()
```

### 协议

```
Request  →  {"code": "page.goto('...')\nprint(page.title())"}
Response ←  {"returns": null, "std_output": "Example Domain",
             "exception": null, "traceback": null}
```

JSON-line：一行一条完整消息。多行代码中的 `\n` 被 JSON 转义保留，不影响行分割。无 prompt 标记，无 ANSI 清理，无歧义。

### 安全模型

- **子进程度**：Child 使用 `SANDBOX_BUILTINS`（禁用 `__import__`、`open`、`eval`、`exec`、`compile`、`input`、`breakpoint`）
- **命名空间注入**：领域对象（`page`、`browser`）和工具模块（`json`、`urllib`）在 child 进程初始化时注入 Sandbox，后续 exec 中可直接使用
- **进程隔离**：Child crash 不影响 Parent Channel；Parent 检测 pipe 断开后可重启 child
- **清理**：shutdown 信号 → `browser.close()` → `playwright.stop()` → `sandbox.close()`

## 核心决策

### 1. 子进程而非线程

Playwright sync API 的约束是架构级事实。`sync_playwright()` 构造函数内部检测 `asyncio.get_event_loop().is_running()`，无论从哪个线程调用都会命中。Thread + Janus 桥不能解决构造时的 event loop 检测。

子进程是唯一干净的方案。这个决策也意外带来了额外好处：进程隔离让 crash 恢复更简单，pipe 断开是明确的失败信号。

### 2. 模块级 EvalServer 初始化

`EvalServer.__init__()` 在 `Matrix.discover().run(main)` 之前运行——此时没有 asyncio event loop。这不是偶然，是结构要求。如果 EvalServer 在 `main()` 内创建，会重新进入 event loop 线程问题。

代价：`build_channel()` 需要同步完成（包括等待 "ready" 信号和浏览器启动）。这约需 3-5 秒，对 App 启动是可接受的启动成本。

### 3. Sandbox 作为持久化执行引擎，不作为能力框架

Sandbox 只做三件事：builtins 安全、持久化 namespace、执行结果结构化。Channel 层负责 CTML 协议适配和生命周期管理。这种分层使得 Sandbox 可替换——如果未来 Jupyter kernel 或 `python -i` 方案解决了当前问题，可以替换 Sandbox 而不影响 Channel 接口。

### 4. JSON-line 而非 terminal 协议

pexpect 路径的问题本质是：将 Python REPL（有状态、line-based echo）当作 shell（无状态、command-based output）来处理，导致协议层不匹配。JSON-line 消除了所有 terminal 语义——没有 prompt，没有 echo，没有 ANSI。

## 与 2024 年 MOSS 论文的关系

此方案本质上是 MOSS 论文 (arXiv:2409.16120) 核心架构在 2026 年的重新实现：

| 论文概念 | 当前实现 |
|----------|----------|
| 持久化 Python 执行环境 | Sandbox (ModuleType namespace, cumulative exec) |
| Code-driven evolution | AI 通过 `exec` 在持久化 namespace 中增量写代码 |
| IoC 容器 + 抽象接口 | MOSS Channel 体系 + `matrix.provide_channel()` |
| WYSIWYG 环境 | Module 源码即 instruction (Code as Prompt) |

区别：论文原实现为进程内架构，当前版本增加了子进程安全隔离。这解决了 2024 年版的一个隐性问题——AI 代码执行的安全边界在同一个 Python 进程中难以强制执行。

## 验证：Playwright App

`browsers/playwright` 作为第一个验收用例，全链路验证通过：

- CTML `<playwright:exec>` → Channel → EvalServer → Sandbox → Playwright → Web
- 变量跨 exec 调用持久化
- 异常正确处理和回传
- 浏览器可见窗口（headless=False）正常工作
- data URL 和 file:// 协议渲染 mermaid 图表

实现量：`eval_server.py` ~60 行 + `main.py` ~90 行。核心逻辑（Subprocess + Sandbox + JSON-line）不到 120 行。

## 泛化路径

同一 Eval Server 模式可包裹任意 Python 模块，只需替换子进程中的领域对象初始化代码：

| 领域 | 注入对象 | 使用模式 |
|------|---------|---------|
| pandas | DataFrame | `df.groupby(...)` |
| ROS2 | Node, Publisher | `node.publish(...)` |
| OpenCV | VideoCapture | `cap.read()` |
| SQLite | Connection | `conn.execute(...)` |

Channel 层和 Eval Server 通信层完全不感知领域差异。这是 ModuleEvalChannel 作为正式 channel type 的基础。

## 扩展方向

- **context_messages**：约定一个 `observe()` 函数，每个关键帧自动执行，返回结构化状态。人类可以直接看浏览器窗口，AI 通过 context_messages "看"
- **Resource 集成**：Eval Server 产出图片/视频/HTML → 注册到 ResourceRegistry (`scheme://host/path`) → 人类和 AI 共享视觉认知
- **Session Bus dispatch**：当前是同一机器父子进程，未来扩展为同 session 内跨进程 Future 协议——同一套接口，不同的 transport
- **安全模型细化**：从固定 SANDBOX_BUILTINS 白名单演进为按 Module 声明的能力清单

## 体系位置：五层架构的首次全链路实现

Playwright App 是 MOSS 五层架构首次在一个具体应用中同时落地：

```
CTML 并行多轨流式控制
  │  AI 输出 token → 实时解析 → 多 channel 并行执行
  │  Playwright App: <playwright:exec> 与 <speech:say> 可以在同一关键帧并行
  │
Code as Prompt
  │  Module 源码 = instruction，AI 看到的是 page.goto() 的真实签名
  │  不是 wrapper，不是 JSON Schema，是源码本身
  │
Python 沙盒 (有状态 Runtime)
  │  Sandbox: persistent ModuleType namespace + SANDBOX_BUILTINS
  │  变量跨调用累积，exec() 是 REPL 的一个 step
  │
跨进程通讯总线
  │  Parent (Channel) ↔ Child (Eval Server): stdin/stdout JSON-line
  │  Channel ↔ Host: Zenoh (Matrix 自动管理)
  │  locality 对 AI 透明——它只看到 CTML 命令签名
  │
运行时依赖注入 (Matrix)
  │  Matrix.discover() → matrix.provide_channel()
  │  运行时注册，运行时发现，无需重启 Host
```

每一层都是独立可替换的模块。CTML 语法不关心 Channel 实现；Sandbox 不关心通讯协议；Matrix 不关心 Channel 内部架构。五层之间的接口契约——CTML 标签名、Command 函数签名、JSON-line 格式——构成体系的 API 边界。

这是从 2024 年论文到 2026 年实现的完整映射。论文提出了"持久化执行环境 + Code as Prompt + IoC 抽象"三位一体；当前实现在此基础上增加了进程安全隔离和流式并行控制两个维度。

## 关联文件

- `.ai_partners/features/workstreams/2026/06/module-eval-channel/FEATURE.md` — 需求与决策记录
- `.moss_ws/apps/browsers/playwright/main.py` — Channel 端实现
- `.moss_ws/apps/browsers/playwright/eval_server.py` — Eval Server 实现
- `src/ghoshell_moss/core/codex/sandbox.py` — Sandbox 引擎
- `src/ghoshell_moss/channels/module_eval_channel.py` — 原 ModuleEvalChannel 工厂（单进程版本）
