# 项目概述

项目名为 `MOS-Shell` (Model-oriented Operating System Shell), 包含几个核心目标:
目标是 AI 大模型作为大脑, 不仅可以思考, 还可以 实时/并行/有序 地操作包括 计算机/具身躯体 来进行交互.

MOS-Shell 是 Ghost In Shells (中文名: 灵枢) 项目创建的新交互范式架构, 是第二代 MOSS 架构 (完善了 ChannelApp 和
Realtime-Actions 思想). 第一代 MOSS 架构 (全代码驱动 + FunctionToken) 详见 [GhostOS](https://github.com/ghostInShells/ghostos)

**更多设计思路请访问飞书文档**: [核心设计思想综述](https://ycnrlabqki3v.feishu.cn/wiki/QCKUwAX7tiUs4GkJTkLcMeWqneh)

## Beta 版本

当前还在 beta 版本的开发中, 没有时间精力完善文档与工具. 简单介绍下 main 分支的使用: 

```bash
# 1. 下载仓库 - 略
# 2. 使用 uv 创建环境并且安装全部依赖 (暂时没拆分好依赖)
uv venv 
source .venv/bin/activate
uv sync --ative --all-extras

# 初始化运行环境. 
moss ws init

# 查看更多命令
moss

# 交互命令行
moss-cli

# debug 用的 repl
moss-repl

# 以 MCP 的方式运行, 可以提供给 claude code 使用. 
moss-as-mcp
```

更多的介绍等 beta 版本基本收敛后完善. 预计通过 claude code 提供项目解释. 