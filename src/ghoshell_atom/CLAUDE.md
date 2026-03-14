# 关于 atom

当前目录相对于项目根目录 `src/ghoshell_ghost/atom`.
这是 Ghost In Shells (Ghoshell) 框架中, Ghost 抽象的第一个关键实现. 目前正在早期开发阶段. 

# 基本概念

指导这个目录开发的核心设计思想在 [](../concepts) 目录下. 遵循 `ghoshell_ghost.concepts` 的理念设计. 
其中最核心的是 [](../concepts/ghost.py) 文件里包含的设计理念. 

不过两者均在进行中, 会同步改动. 通过 Atom 完善 Ghost 的设计. 

Atom 是 Ghost 的一种实现, 它希望遵循的理念有: 

1. 端侧运行: Atom 类似 ROS2 一样是在端侧运行的, 所以一切技术实现本地优先
2. 可分发: 基于本地优先, 各种数据存储优先文件而不是数据库 (比如 .md). 这样项目本身是用代码仓库可分发的. 
3. workspace: 用 workspace 的方式管理内部文件. 基本的思路是运行目录下有 `.atom` 的文件夹存储了它的一切. 
4. cli 管理: 核心目标是通过 cli 可以管理环境, 不断丰富 cli 指令来 创建/完善/优化 一个 Atom 实例的运行环境. 
5. 原型到实例: Atom 本身是一个原型, 它需要通过 cli 未来的 `ghoshell atom init` 之类的命令实例化到目录, 然后在运行过程中完善. 
6. 持久化进程: Atom 实例的运行过程是端侧的持久化进程. 
7. 父子多进程模型: Atom 分治的一些能力, 通过多进程模型来运行. 具体的依赖下文讨论. 
8. 能力发现: Atom 实例在 workspace 里积累的能力, 优先基于约定, 通过自动发现 (首先是文件发现) 来实现. 约定优先于配置. 
9. 能力成长: Atom 实例应该可以在 Workspace 里不断增加它的能力. 其中一部分沉淀回到 Atom 原型设计中.  
10. 自迭代: 一个 Atom 实例被初始化后, 应该具备 AI 自迭代的效果. 它可能需要支持多种自迭代范式. 后文讨论. 

# 通讯架构基础

* 文件优先: 凡是能通过文件实现通讯的 (watch_dog, 分工读写), 尽量用文件通讯. 存储结构也优先参考文件. 
* 简化存储: markdown, jsonl, yaml (pretty dump) 是比较好的存储方式. 
* Zenoh 进程间通讯: 考虑用 zenoh 实现进程间的 pub/sub, actor 等方式的通讯. 
* diskcache 做存储: 能够用 diskcache 实现的存储, 都用它来进行.  
* circus 进程管理: 多进程管理优先用 circus 来做

具体的实现则遵循 ghoshell_ghost 设计的通讯范式. 

# 目录结构

## 整体目录

- `.atom/` : 原型的 workspace 目录. 需要保存所有的配置, 能力, 运行时信息, 能力发现约定, 以及 coding agent 可阅读的讯息. 
- `framework/`: Atom 的系统框架. 
- `cli/`: 在 Atom 原型上派生出来的命令行工具. 未来集成到 ghoshell_cli 中. 

## workspace 设计

workspace 是 ghost 原型分发的基本方式. 预计通过 `ghoshell atom init` 这样的命令可以初始化环境. 

## `./framework`

在 `ghoshell_ghost.atom` 的原型实现中, 系统开箱自带的能力和运行框架, 都在 `ghoshell_ghost.atom.framework` 中实现. 

framework 下的每个目录是一个具体的模块. 这个模块默认的文件: 

- `README.md`: 让人类工程师阅读的文件. 
- `CLAUDE.md`: 坚持让 Atom 基于 claude code 开发完善. 信息量比 README 更重要. 
- `__init__.py`: 用来整理 package 的各种可引用包和库. 
- `abcd.py`: 全称 `abstract design`, 是模块的抽象设计. 这里应该遵循 `code as prompt` 原则, 最大化地自解释 (面向 AI 协作者).


# 自迭代范式

Atom 需要实现 AI 主导的自迭代. 会结合多种范式. 主要分为运行时自迭代与 AI developer

## AI Developer

这个范式比较容易理解, 基于 workspace 文件创建/编辑 的方式迭代. 可以通过 claude code 或者其它的 AI 项目来迭代. 
所以关键是开发者 (我) 需要把足以 开发能力/工具 的知识记录到关键目录里, 指导开发范式.  

## 运行时自迭代

运行时自迭代指的是 AI 在实时运行过程中, 仍然可以自主创建/修改自己的能力并且热更新. 这些自迭代范式会分为很多种. 

### 迭代动机

对于 Ghost 而言, 触发自迭代的动机应该是: 

* 教学模式: 人类在特定的教学模式, 要求 AI 开发能力. 
* 能力学习: AI 基于上下文, 得到有用的知识和经验, 增加自己的能力. 
* 反思优化: 通过并行思考链路, 反思行为表现, 触发优化. 
* 强制机制: 对于记忆等自迭代对象, 通过系统的强制约定触发自迭代行为. 记忆更新也是自迭代. 

### 自迭代对象

预计框架要支持的自迭代对象包含:

* 能力类
* 系统类
* 元信息
* 记忆 & 知识类. 

具体的讯息以后逐步补充. 

### 技术途经

* 并行思维 & 任务单元: Ghost 可以运行时触发别的模块执行开发, 比如将 claude code 的非交互模式作为一个 moss 的 command. 
* CTML: ctml 语法本身就可以支持自迭代. 
  * 保存/使用: 保存 ctml 到特定目录, 运行时动态呈现, 提示 AI 用特殊token (比如` <😉/>`) 来代指已经保存的 CTML.  
  * 字符串函数: 通过字符串函数语法, 可以将一个字符串模板反射成一个 command. 
* MOSS Channel: MOSS channel 预计实现多种自迭代能力. 
  * Module 封装:  module channel 反射一个 python module, 可以在运行时定义 command function 保存, 生成 command. 
  * Command 封装: 特殊的父 Channel 可以将子 channel 的能力用纯代码封装成新的 command, 自动生效. 
  * 进程 Channel: 基于 python 实现的独立运行的子进程脚本, 理论上都能封装出 Channel, 通过进程提供给 AI. 
  * Realtime GUI Channel: 支持运行时自定义 layout 然后流式使用. 

相关目标还在开发中. 

### 触发机制

- 模式切换: 可以提供专门服务于自迭代的 GhostMode. 用户要求进入 Mode 后才能使用相关功能. 
- 主交互 AI 自迭代: 在与用户交互的过程中, 直接调用提供的工具 (通过 moss 协议) 实现自迭代.
- 并行思维自迭代: 通过并行思维模块, 在主路运行的同时, 通过旁路执行自迭代逻辑. 
- Tasks: 通过后台任务模块触发迭代. 