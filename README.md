# 项目概述

项目名为 `MOS-Shell` (Model-oriented Operating System Shell), 包含几个核心目标:

1. `MOS`: 为 AI 大模型提供一个 "面向模型的操作系统", 可以将 跨设备/跨进程 的功能模块, 以 "树" 的形式提供给模型操作.
1. `Shell Runtime`: 为 AI Agent 提供一个持续运转的运行时 (Runtime), 联通所有功能模块 (称之为 Channel, 对标 python 的
   module).
1. `Code As Prompt`: 让 AI 大模型用 python 函数 的形式理解所有它可调用的功能, 而不是 json schema. 实现 "
   面向模型的编程语言".
1. `Streaming Interpret`: 支持 AI 大模型流式输出对话和命令 (Command) 调用, 并且 Shell 会流式地编译执行这些调用,
   并行多轨控制自己的躯体和软件.

目标是 AI 大模型作为大脑, 不仅可以思考, 还可以 实时/并行/有序 地操作包括 计算机/具身躯体 来进行交互.

MOS-Shell 是 Ghost In Shells (中文名: 灵枢) 项目创建的新交互范式架构, 是第二代 MOSS 架构 (完善了 ChannelApp 和
Realtime-Actions 思想). 第一代 MOSS 架构 (全代码驱动 + FunctionToken) 详见 [GhostOS](https://github.com/ghostInShells/ghostos)

**更多设计思路请访问飞书文档**: [核心设计思想综述](https://ycnrlabqki3v.feishu.cn/wiki/QCKUwAX7tiUs4GkJTkLcMeWqneh)

## Alpha 版本声明

当前版本为内测版 (Alpha), 这意味着:

1. 项目仍然在第一阶段开发中, 会激进地迭代.
1. 主要是验证核心链路和设计思想, 许多计划中的关键功能还未实现.
1. 暂时没有人力去完善文档
1. 不适合在生产环境使用.

如果想要试用项目, 请直接联系 灵枢开发组 配合.

想要阅读架构的设计思想, 推荐直接看 [concepts 目录](src/ghoshell_moss/core/concepts).

## Examples

在 [examples](examples) 目录下有当前 alpha 版各种用例. 具体的情况请查阅相关目录的 readme 文档. 

体验 examples 的方法: 

> 建议使用 mac, 基线都是在 mac 上测试的. windows 可能兼容存在问题. 
 
## 1. clone 仓库

```bash
git clone https://github.com/GhostInShells/MOSShell MOSShell
cd MOSShell
```

## 2. 创建环境

* 使用 `uv` 创建环境, 运行 `uv venv` . 由于依赖 live2d, 所以默认的 python 版本是 3.12
* 进入 uv 的环境: `source .venv/bin/activate`
* 安装所有依赖: 

```bash
# examples 的依赖大多在 ghoshell-moss[contrib] 中, 没有拆分. 所以需要安装全部依赖. 
uv sync --active --all-extras
```

## 3. 配置环境变量

启动 demo 时需要配置模型和音频 (可选), 目前 alpha 版本的基线全部使用的是火山引擎.
需要把环境变量配置上. 

```bash
# 复制 env 文件为目标文件. 
cp examples/.env.example examples/.env

# 修改相关配置项为真值. 
vim examples/.env
```

配置时需要在火山引擎创建 大模型流式tts 服务. 不好搞定可以先设置 USE_VOICE_SPEECH 为 `no`

## 4. 运行  moss agent

```bash
# 基于当前环境的 python 运行 moss_agent 脚本
.venv/bin/python examples/moss_agent.py

# 打开后建议问它, 你可以做什么. 
```

已知的问题: 
1. 语音输入模块 alpha 版本没有开发完.  
2. 目前使用的 simple agent 是测试专用, 打断的生命周期还有问题.
3. 由于 shell 的几个控制原语未开发完, 一些行为阻塞逻辑会错乱. 
4. interpreter 的生命周期计划 beta 完成, 现在交互的 ReACT 模式并不是最佳实践 (模型会连续回复)

更多测试用例, 请看 examples 目录下的各个文件夹 readme. 


## Beta Roadmap

Alpha 版本是内测版. 预计在 Beta 版本完成:

- [ ] 中英双语说明文档
- [ ] 流式控制基线
  - [ ] CTML 控制原语: clear / stop_all / wait / concurrent / observe. 目前原语未完成, 多轨并行和阻塞存在问题.
  - [ ] Speech 模块 Channel 化.
  - [ ] 完善 CommandResult, 用于支持正规的 Agent 交互范式.
  - [ ] 完善 states/topics 等核心技术模块.
  - [ ] 完善 Interpreter 与 AI Agent 的交互范式基线.
- [ ] 完善 Channel 体系
  - [ ] 定义 Channel App 范式, 创建本地的 Channel Applications Store
  - [ ] 完善 Channel 运行时生命周期治理
  - [ ] 完成对 Claude MCP 和 Skill 的兼容
- [ ] 完善 MOSS 项目的自解释 AI
  - [ ] 实现第一个 Ghost 原型, 代号 Alice
  - [ ] 实现架构级的 Channels, 用于支撑基于 MOSS 运转的 Ghost 体系.
  - [ ] 实现一部分开箱即用的 Channels, 用来提供 AIOS 的运行基线.

## Contributing

- Thank you for being interested in contributing to `MOSShell`!
- We welcome all kinds of contributions. Whether you're fixing bugs, adding features, or improving documentation, we appreciate your help.
- For those who'd like to contribute code, see our [Contribution Guide](https://github.com/GhostInShells/MOSShell/blob/main/CONTRIBUTING.md).
