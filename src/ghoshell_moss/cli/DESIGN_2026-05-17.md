# CLI 命令体系设计方案

> 2026-05-17, 人类工程师 + Claude Code 讨论定稿

## 核心目标

CLI flow 为 **AI 模型低上下文自驱** 设计。AI 通过 startup prompt + `all-commands` 一次性理解全部命令，不需要人工导航。人类的终端体验由 `moss-cli` + shell 补完 + 未来 agent 辅助解决。

## 命名风格

统一为 **短英文名词**，无缩写、无连字符、无拉丁词根。`ctml` 作为项目专有名词保留。

## 命令树

```
workspace/              # 环境地基
  where                 # 定位当前 workspace
  init                  # 初始化 workspace
  override              # 覆盖更新 workspace stub
  copy-env              # 复制 .env_example → .env
  ctml-read [version]   # 读取 CTML 协议内容
  ctml-versions         # 列出环境中 CTML 版本
  manifests/            # 环境发现工具箱（唯一子组，9 命令需折叠）
    providers           # 探查 IoC providers
    topics              # 探查事件 topics
    configs             # 探查环境配置
    channels            # 探查通信 channels
    primitives          # 探查原语 (Commands)
    contracts           # 探查 IoC 绑定契约
    resources           # 探查 ResourceStorageMeta
    nuclei              # 探查 NucleusFactory

apps/                   # 应用开发入口 (CLI flow)
  list                  # 列出环境中发现的应用
  show <fullname>       # 查看应用详情
  test <fullname>       # 前台启动应用子进程调试

modes/                  # 模式配置入口 (CLI flow)
  list                  # 列出所有 mode
  show <name>           # 查看 mode 详情
  create <name>         # 创建新 mode

codex/                  # 代码反射 + 知识索引
  # 反射工具
  get-interface <path>  # 反射模块/属性接口
  get-source <path>     # 获取源码
  info <module>         # 模块详细信息
  list <package>        # 列出包内模块
  # 全局知识索引
  concepts [name]       # 核心概念全景（反射生成）
  blueprint [name]      # 蓝图概念全景（反射生成）
  contracts [name]      # 基础抽象契约全景（反射生成）
  # 知识库入口
  howto [path]          # howto 树状索引 + desc

features/               # MOSS 自身迭代追踪
  specification         # 显示 features 约定
  list                  # 列出活跃 workstream
  status [name]         # 查看 workstream 状态
  create <name>         # 创建 workstream
  set-status <name> <s> # 设置状态
  init                  # 初始化 features 骨架
```

## 设计决策记录

1. **`manifests` 保留为 `workspace` 下唯一子组。** 原因：9 个命令超过单层可扫描阈值（~7），且 `workspace manifests <noun>` 读起来像自然语句。
2. **`ctml` 从 codex 移至 workspace。** 原因：CTML 是 workspace 内的本地文件，属于环境发现的一部分，不属于知识库。
3. **`concepts` 从独立组并入 codex，压平为单命令。** 原因：`concepts core` → `codex concepts`，无子命令，一次调用出全景图。它们是反射生成的知识索引，和反射工具同源。
4. **`howto` 从独立组并入 codex，简化为单命令。** 原因：只输出树状索引 + desc，不保留 recall agent。howto 是图的入口，不是独立的知识库。
5. **`apps` 和 `modes` 保持顶级组。** 原因：它们是 CLI flow 的入口点，是开发循环的起点环节。从属于 workspace 对架构正确，但对 flow 增加了不必要的层级。
6. **全局 `--mode` 参数是 workspace 相关参数**，语义上确认了 apps/modes 与 workspace 的亲和性，但 flow 层面保持顶级入口。

## 未完成事项

- `docs` 命令组：未来作为树形参考文档体系，含 `list` / `read` / `search`。不收纳 howto 和 faq（它们属于图的入口）。
- `workspace apps create`：当前 apps 下无 `create`，未来 workspace 下的开发脚手架需求。
