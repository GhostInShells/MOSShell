---
title: Module Eval Channel
status: in-progress
priority: P2
created: 2026-06-03
updated: 2026-06-09
depends: [codex-module-sandbox]
milestone:
description: >-
  Generic channel type that wraps any Python module as an eval container — AI sees module source as instruction, writes code via named exec command with text__ parameter, persistent namespace across calls, module defines the domain (Playwright, pandas, ROS…), channel is a thin reusable shell.
---

# Module Eval Channel

> Channel type: 系统管理 | alpha

## Motivation

现有的 Channel 构建范式（L0-L4）要求每个领域能力都写一遍 Channel：定义 command、写 docstring、管理生命周期。但对于"给 AI 一个 Python 运行时，让它直接写代码控制某个领域对象"这类场景，Channel 总是相同的壳——`exec` 命令（`text__` 参数接收代码） + exec 执行 + 持久化 namespace + `vars()`/`api()` 辅助 + 崩溃恢复。不同的只是 Module 里是什么。

Playwright 浏览器控制是这个模式的第一个实例。如果把 Module（定义 page/browser 对象、启动/清理逻辑、源码作为 instruction）和 Channel（通用 eval 壳）分开，那么写一个"浏览器控制"就是写一个 50 行的 `playwright_module.py`，而不是重写整个 App。

最终目标：`ModuleEvalChannel` 成为一个正式 channel type，位于 `ghoshell_moss.channels` 中，可被任何 App 复用。

## 与 Sandbox 的分层

`codex-module-sandbox` (P1, in-progress) 提供了一个纯 Python 的安全执行引擎——基于 `Compiler` 的 `ModuleType` 容器，builtins 可控，`exec(code)` 捕获 stdout，父子沙盒共享 `__dict__`。但它不懂 CTML，不懂 Channel，不懂 Matrix。

ModuleEvalChannel 和 Sandbox 是**分层关系，不是竞争**：

```
AI (CTML)
  └─ ModuleEvalChannel   ← 我们：CTML 接口 + instruction + context_messages
       └─ Sandbox          ← codex-module-sandbox：安全 exec + stdout 捕获 + 生命周期
            └─ Compiler     ← 已有：ModuleType 容器
                 └─ 领域对象 (page, browser, df, db...)
```

**职责边界**：

| 层 | 负责 | 不负责 |
|---|---|---|
| Sandbox | builtins 安全、exec 执行、stdout 捕获、namespace 管理、init/destroy 生命周期 | CTML 命令、instruction 生成、视觉 context、Matrix 通讯 |
| ModuleEvalChannel | `exec`/`vars`/`api` CTML 命令、Module 源码 → instruction、`context_messages`、Janus 线程桥 | builtins 安全策略、ModuleType 创建、exec 底层实现 |

ModuleEvalChannel 的 `exec` 实现就是 `sandbox.exec(text__)` + 格式化 + 返回。Sandbox 已经解决了 builtins 安全、stdout 捕获、父子 namespace 传导——Channel 层不需要重做这些。

## Key Decisions

### 1. 分离 Module 与 Channel

Module 是领域容器——定义对象、启动逻辑、清理逻辑、源码即 prompt。Channel 是通用壳——`exec` + `vars()` + `api()` + stdout 捕获。Channel 不关心 Module 里是 Playwright 还是 pandas。

**Why**: 模型看到的 instruction 直接是 Module 源码，比任何手写的 docstring 都精确——源码里有什么对象、什么类型、什么方法，一目了然。Code as Prompt 的极限形式。

**Rejected**: 为每个领域单独写 Channel（如 `PlaywrightChannel`、`PandasChannel`）。会导致大量重复的 `exec` + namespace 管理代码。

### 2. exec 命令 + text__ 参数 + 持久化 namespace

使用 `Builder.command(name="exec")` 注册命名命令，参数类型 `text__: str`。按 CTML 规范，`text__` 通过开放-闭合标签传入——模型写在 `<playwright:exec>` 和 `</playwright:exec>` 之间的代码被捕获为完整字符串，在持久化的 `module.__dict__` 中执行。变量跨 `exec` 调用自然累积，像一个 Python REPL。

```python
@channel.build.command(name="exec", always_observe=True)
async def exec_code(text__: str, observe: bool = False) -> str:
    """Execute Python code in module namespace. text__: code to execute."""
    ...
```

模型调用：
```ctml
<playwright:exec>
page.goto("https://example.com")
print(page.title())
</playwright:exec>
```

**Why `text__` 而非 `__content__`**：

`__content__` 的签名是 `async def __content__(chunks__=None)`。它是 ChannelRuntime 的魔法方法——通道内非标记文本的默认处理器。主通道的 `__content__` 是语音输出；其他通道的 `__content__` 语义取决于具体实现。用 `__content__` 承载代码执行会导致：
- 语义模糊："这个通道的默认内容" 和 "执行 Python 代码" 是两个完全不同的概念
- 无法与同通道其他命令共存——`__content__` 占用了"非标记文本"的语义位置
- `chunks__` 是流式迭代器，exec 需要完整代码文本，`text__` 直接交付完整字符串

`text__` 是 CTML 的三个流式参数类型之一（`text__` / `chunks__` / `ctml__`），专为"通过开闭标签传入完整文本"设计。配合命名 command `exec`，模型显式表达"我要执行代码"的意图。

**Why**: 比把每个 API 反射成 Command 高效一个数量级。模型凭预训练知识直接写代码，不需要学习"这个 Channel 把 `page.goto` 翻译成了什么 command 名"。Playwright 有上百个 API——反射全部是无用功。

**Rejected**: 
- `__content__` 方案：语义错误。`__content__` 是"通道默认内容处理器"，不是代码执行入口
- 函数级命令封装（`goto(url)`, `click(sel)`, `fill(sel, val)`...）。会丢掉 Playwright 的链式调用、组合灵活性，且维护成本随 API 数量线性增长

### 3. vars() + api() 作为唯一的辅助命令

两个命令，不扩展。`vars()` 列出/查看 namespace 中的变量。`api()` 反射 Module 中对象的签名和 docstring。模型凭预训练知识写代码 → 不确定时查一下 → 继续。

**Why**: 模型预训练知识覆盖 Playwright 80% 的高频 API。不需要全量反射。给模型一个"查文档"的能力就够了——和人类开发者一样。

**Rejected**: 全量反射所有 API 为 command（维护噩梦）。不做任何辅助（错误恢复成本高，模型写错 API → exec 抛异常 → 重试，循环次数多）。

### 4. 崩溃恢复：只保护 storageState

Playwright 的现场分三层：Browser 进程（崩了全丢）→ BrowserContext（cookie/localStorage 可恢复）→ Page（DOM 不可恢复）。唯一需要持久化的是 BrowserContext.storageState。

策略：每次 `exec` 执行后保存 storageState → 浏览器崩了 → 重启 → 恢复 storageState（登录态还在）→ 清理 namespace 中的 Playwright 对象引用（纯 Python 变量保留）→ 模型从 error 中知道需要重新 nav。

**Why**: Page 层的 DOM/URL/JS 上下文不可序列化，强行快照的复杂度远超收益。模型从 error 信息里重建操作序列的成本远低于维护复杂的快照恢复机制。

### 5. Janus 桥：主线程执行

Playwright sync API 需要在主线程/固定单线程运行。Matrix Channel Runtime 在后台 event loop。需要 `queue.Queue` + `threading.Event` 将代码从 Matrix 线程卸载到 Module 主线程执行，结果同步返回。

**Why**: Channel 的生命周期管理（启动/清理/超时取消）由 Matrix 线程控制，但 exec 的副作用必须在 Module 线程产生。两线程通过 Janus 桥解耦——Channel 侧不感知线程模型差异。

### 6. 先做 channel type，后做 Playwright App

第一阶段产出是 `ModuleEvalChannel` 本身 + 单测。第二阶段用它在 `.moss_ws/apps/` 下创建 Playwright App。

Playwright 是核心验收用例——跨调用浏览器进程存活、Janus 桥线程模型、storageState 恢复，验证的是 sandbox 作为持久化 REPL 的价值。Pandas/SQLite 等无状态场景不比 `python -c` 强太多，不做独立验收。

**Why**: 单测验证 channel type 正确性，Playwright 验证泛用性。浏览器是有状态长时间运行对象的代表——这个 case 通了，其他领域对象（pygame、ROS node、OpenCV pipeline）同理。

## Design Index

- Channel 构建参考: `moss codex blueprint channel_builder`
- CTML 语法（`text__` 参数类型）: `moss ctml read v1_0_0.zh`
- States Channel (L3 参考): `moss codex list ghoshell_moss.core.blueprint.states_channel`
- App 体系: `moss docs read app-system.md`
- channeltypes 注册: `moss codex channeltypes`

## Implementation Notes

- **依赖 Sandbox**：`exec` 内部委托给 `Sandbox.exec()`，处理 `ExecutionResult`（std_output, exception, traceback, returns）
- **反射委托 Reflector**：Sandbox init 时持有 `Reflector(module, source=module_source)`，`get_interface()` 返回 source + import attr 块，与 `moss codex get-interface` 一致
- `vars()` 委托给 `sandbox.get_interface()`——返回 Reflector 输出（module source + `<attr>` 块）
- `api(name)` 委托给 `sandbox.get_interface(name)`——import 对象走 `reflect_imported_attr` 管线，本地对象走 inspect fallback
- `api(name, *methods)` 保留 channel 层 logic——`inspect.signature` + `inspect.getdoc` 对 exec 对象始终可用
- **生命周期修复**：cleanup 关闭 `init_sandbox`（root），级联关闭 child sandbox 后清理 namespace
- `exec` 命令签名：`async def exec_code(text__: str) -> str`，`always_observe=True`
- Sandbox 的 builtins 安全策略由 Sandbox 的 FEATURE.md 定义——Channel 层不重复决策
