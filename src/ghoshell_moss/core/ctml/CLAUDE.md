# MOSShell 项目 CTML 实现开发指南

你现在处于 MOSShell 项目. 这个项目包含几个核心目标:

1. `MOS`: 为 AI 大模型提供一个 "面向模型的操作系统", 可以将 跨设备/跨进程 的功能模块, 以 "树" 的形式提供给模型操作.
1. `Shell Runtime`: 为 AI Agent 提供一个持续运转的运行时 (Runtime), 联通所有功能模块 (称之为 Channel, 对标 python 的
   module).
1. `Code As Prompt`: 让 AI 大模型用 python 函数 的形式理解所有它可调用的功能, 而不是 json schema. 实现 "
   面向模型的编程语言".
1. `Streaming Interpret`: 支持 AI 大模型流式输出对话和命令 (Command) 调用, 并且 Shell 会流式地编译执行这些调用,
   并行多轨控制自己的躯体和软件.

它的核心概念和抽象设计在目录 `../concepts` 下. 本目录则是关于 CTML 的实现.

CTML 是一种 XML-like 的语法, 旨在让大模型输出 xml 语法同时通过 MOSShell 流式控制它可以交互的设备.
核心的 CTML 规则目前请查阅 `./prompts/ctml_v2.zh.md` 文件.

你可以实现的任务如下:

## prompts 优化

在 `./prompts` 目录下存放了不同版本的 CTML 语法规则. 作为 MOSShell 的 CTML 版本实现的 meta instruction.
这一块你可以:

1. 协助用户撰写 prompt
1. 协助用户翻译 prompt 的不同语言版本.

## 原语开发

CTML 通过一系列函数化的控制原语来实现复杂的时序控制功能.

- 相关原语实现在 `./shell/primatives`
- 原语的单元测试在 `../../../../tests/shell/test_primitives` 目录下.

原语的技术实现非常复杂. 你的主要任务是帮助用户开发原语的单元测试.
