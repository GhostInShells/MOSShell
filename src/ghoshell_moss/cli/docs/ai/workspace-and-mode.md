# MOSS Workspace 与 Mode — 自举地基

本文档是[架构拓扑](architecture-topology.md)的延伸。读完拓扑文档后，本文解释 workspace 和 mode 的设计：它们为什么存在、解决什么问题、如何协同构成 MOSS 的自举地基。

**读者**：进入 MOSS 项目的智能模型。读完本文应能建立 workspace/mode 的心智模型，知道从哪里开始探索一个具体的环境。

## 1. 定位

在八层拓扑中，workspace 是系统面的最底层——自举地基。mode 不是独立的一层，而是 workspace 之上的**能力视图**。

```
编排面    Ghost → Host
系统面    Mindflow → Matrix → [Mode 视图]
          ─────────────────────────────
          Workspace (文件系统级 IoC 自举)
模型面    Shell → Channel → Logos
```

一句话：**workspace 用目录约定替代生命周期注册。mode 用叠加声明替代配置切换。** 两者合力，让智能模型不需要理解运行时启动链就能理解环境、发现能力、完成迭代。

## 2. 为什么是文件系统

传统框架的能力注册需要回答三个问题：在哪里注册（装饰器？配置文件？）、什么时候注册（启动前？运行时？）、如何注入（构造函数？service locator？）。每个框架答案不同，智能模型每次都要先调研。

MOSS 的回答：**放到约定路径下，写一个 Python 实例。** Matrix 启动时自动发现、注入。

这个设计的前提是有一个能读代码、理解约定的智能模型——人的记忆容量不适合记住所有约定，但模型可以。workspace 本质是为 AI 读者设计的。

### 2.1 三个核心能力

智能模型作为 MOSS 环境的第一使用者和开发者，需要三种能力：

1. **最小知识理解运行时**。不需要调研生命周期和注册链。配合 `moss manifests` / `moss modes` / `moss apps` 三个命令，在几秒内建立对环境能力的完整认知。
2. **方便修改**。文件级关注点分离——providers.py 只管依赖注入，channels.py 只管一级 Channel，primitives.py 只管 Shell 原语。修改路径可预测：改什么就找什么文件。
3. **开发时/运行时自迭代**。必要的修改隔离（app 独立进程 vs src 共享运行时）+ 工具反查知识（`moss codex get-interface` 理解任何模块的契约）。

### 2.2 五个用户故事

| # | 场景 | 模型如何做 | 关键工具 |
|---|------|-----------|---------|
| 1 | 了解环境中已集成的能力 | 扫描 manifests，建立能力清单 | `moss manifests providers/channels/primitives` |
| 2 | 开发新功能，独立集成 | 理解目录拓扑，在正确位置写代码 | `moss ws where` + 目录约定 |
| 3 | 知道启动什么、怎么启动 | 查看 mode 配置，理解 bringup 规则 | `moss modes list/show` |
| 4 | 选择隔离路径还是复用路径 | app→mode 隔离，src→manifests 复用 | `moss apps list` + manifests |
| 5 | 从最佳实践上手，不全貌到执行 | 先看已有示例，需要时深入 | stub 文件 + `moss codex` |

## 3. Workspace — 目录即注册

### 3.1 发现链

`Environment.find_workspace_path()` 按优先级查找：

1. `MOSS_WORKSPACE` 环境变量（精确指定）
2. 当前目录下的 `.moss_ws/`（本地 workspace）
3. 向上递归查找带 `MOSS.md` 的目录（**项目即 workspace**）
4. `~/.moss_ws/`（用户级默认 workspace）

### 3.2 两种形态

- **项目 workspace**（仓库根目录带 `MOSS.md`）：开发和迭代的主战场。workspace 根目录就是项目根目录。
- **用户 workspace**（`~/.moss_ws/`）：全局环境，所有项目的 MOSS 运行时共享。

`moss ws where` 告诉你当前在哪种 workspace 里。

### 3.3 目录拓扑

```
{workspace}/
  MOSS.md                  # 环境元信息（名称、CTML版本、默认mode）
  .env / .env.example      # 环境变量（API key 等凭据）
  src/                     # 加到 sys.path，可被 import
    MOSS/
      manifests/           # ★ 全局能力声明（所有 mode 共享）
        providers.py       #   IoC 依赖注入声明
        channels.py        #   一级 Channel
        primitives.py      #   Shell 原语（sleep/noop/observe/interrupt）
        configs.py         #   配置模型（TTS、音频播放器等）
        resources.py       #   资源存储（本地图片等）
        topics.py          #   事件协议
        nuclei.py          #   感知核（Mindflow）
      modes/               # ★ 模式声明（叠加在全局之上）
        {mode_name}/
          MODE.md          #   模式配置（apps 白名单、bringup 规则）
          primitives.py    #   模式专属原语
          contracts.py     #   模式专属契约
  apps/                    # ★ App（独立进程）
    {group}/{name}/
      APP.md               #   App 元信息（启动方式、描述）
      main.py              #   App 入口
  configs/                 # 运行时配置（通讯、日志等）
  runtime/                 # 运行时数据（sessions、logs）
  ctml_versions/           # CTML 提示词版本
```

每个文件有单一职责。要改依赖注入就找 providers.py，要加 Channel 就找 channels.py。不需要理解整个拓扑才能动手。

### 3.4 MOSS.md

workspace 的身份证。用 YAML frontmatter + Markdown 内容：

```yaml
---
ctml_version: v1_0_0.zh    # 默认 CTML 提示词版本
default_mode: default       # 启动时默认模式
---
# 以下为 system_prompt，会注入到模型 instruction
```

Python 侧由 `MossMeta` 解析。`Environment.bootstrap()` 时读取。

## 4. Manifests — 声明即注入

### 4.1 核心原则

manifests 目录下的 `.py` 文件**不是配置文件**——它们实例化 Python 对象。

```python
# providers.py — 不是 YAML，是 Python 实例
moss_session_provider = WorkspaceSessionProvider()
config_store_provider = HostEnvConfigStoreProvider()
```

manifests 的存在是为了服务 Matrix。Matrix 是 MOSS 环境的通讯与集成总线，manifest 是它在文件系统中的"配料表"——Matrix 启动时，`scan_package` 扫描 `MOSS.manifests.*`，用 `isinstance` 过滤目标类型，自动注入 IoC 容器。**下次启动自动生效**——不需要碰启动代码。

### 4.2 声明类型

manifests 声明覆盖八个领域——IoC 依赖注入（providers）、一级 Channel（channels）、Shell 原语（primitives）、配置模型（configs）、事件协议（topics）、资源存储（resources）、感知核（nuclei）、CTML 版本。

其中 providers 和 contracts 是一对关键概念：

- **contracts** 是 IoC 容器中已绑定的接口——**"可以拿到什么"**。`moss manifests contracts` 列出当前环境所有可获取的服务实例。每个 contract 既是列表中的条目，也可以直接从 IoC 容器取出使用。
- **providers** 是这些绑定的生产方式——**"由谁生产"**。`moss manifests providers` 列出所有工厂声明。providers 服务于 contracts：添加一个 Provider 实例到 providers.py，就声明了一个 contract 的绑定方式。

**真相入口**：`moss manifests explain`。该命令输出当前环境下所有声明类型的完整表格——每类的职责、发现路径、检测方式、键语义。文档不重复此表，因为 CLI 输出随代码自动保持准确。

每个概念的深入路径：

- **Provider / IoC 容器**：`moss how-tos read get-moss-design/how-ioc-container-work-in-moss.md`
- **Channel**：`moss codex get-interface ghoshell_moss.core.concepts.channel`
- **Command**：`moss codex get-interface ghoshell_moss.core.concepts.command`
- **Topic**：`moss codex get-interface ghoshell_moss.core.concepts.topic`
- **Resource**：`moss how-tos read for-moss-app-developer/how-to-add-a-resource-storage.md`
- **Nucleus / Mindflow**：`moss codex get-interface ghoshell_moss.core.blueprint.mindflow`
- **Config**：`moss codex get-interface ghoshell_moss.contracts.configs`

### 4.3 发现机制

所有 manifest 发现共享同一模式：

1. 拼出约定路径（如 `MOSS.manifests.providers`）
2. `scan_package` 用 `importlib` 递归扫描（`max_depth=2`）
3. `isinstance` 过滤目标类型
4. 按类型特定的键聚合

约定路径和类型检测的具体细节由 `moss manifests explain` 给出。设计上不依赖 `__all__`，而是直接遍历模块成员——声明即存在。

## 5. Mode — 能力视图

### 5.1 叠加而非切换

mode 不是一个独立的环境。它是**叠加在全局 manifests 之上的过滤/扩展层**。

```
MergedManifests([env_manifests, mode_manifests])
```

右边的覆盖左边。这意味着：

- mode 不声明 channels → 用全局的 channels
- mode 声明了额外的 primitives → 全局 primitives + mode primitives
- mode 声明了同名 config → mode 的覆盖全局的

同类别内合并，不跨类别。

### 5.2 MODE.md

```yaml
---
apps:
  - '_system_tests/*'   # 允许加载的 app 模式
  - '*/*'               # 允许所有的 app
bringup_apps: []         # 启动时自动启动的 app
---
# Markdown 内容为 mode 的 instruction，注入到模型提示词
```

`apps` 是白名单（支持 `group/*` 通配符，`_` 前缀的 group 不匹配 `*`）。`bringup_apps` 指定启动时自动拉起的 app。

### 5.3 mode 的三层发现

`list_modes_from_root_package()` 按优先级发现 mode：

1. module 属性中直接找到 `Mode` 实例（`mode = Mode(...)`）
2. 目录下 `MODE.md` 文件中解析
3. 都没有 → 自动生成一个最小 Mode（约定优于配置）

当前 workspace stub 中的 `default` 和 `system_test` 两个 mode 都是通过 `MODE.md` 发现的。

## 6. 启动链路 — 两层设计

### 6.1 Host：发现层（无副作用）

```
Host.__init__():
  Environment.discover() + bootstrap()
    → .env 加载, src/ 加入 sys.path
  PackageManifests.from_environment(env)
    → scan_package MOSS.manifests.*
  list_modes_from_root_package()
    → scan_package MOSS.modes.*
  MergedManifests([env_manifest, mode_manifest])
    → mode 覆盖 env（右边优先）
  MatrixImpl(manifest, ...)
    → 创建 IoC 容器，暂不启动
```

`Host` 构造没有副作用——不启动任何服务，不建立任何连接。CLI 的 `moss manifests`、`moss modes`、`moss apps` 都只需要 Host。

### 6.2 MossRuntime：执行层

```
Host.run() → MossRuntimeImpl.__aenter__():
  Matrix.start()
    → 启动通讯层（当前版本基于 Zenoh）
    → 遍历 manifests.providers() → factory(container) → 注入 IoC
    → 遍历 manifests.bootstrappers() → bootstrap(container)
  AppStore.start()
    → bringup 配置的 apps（独立子进程）
  CTMLShell.start()
    → 启动 shell，refresh channel metas
```

### 6.3 为什么分两层

两层设计让**探索和运行分离**。探索环境能力（manifests/modes/apps）不需要启动任何服务，零副作用。只有真正要运行时才进入执行层。这意味着：

- CLI 工具永远只需要 Host
- 出问题可以隔离排查（是发现层的问题还是执行层的问题）
- 开发和调试时不需要反复启动/关闭运行时

## 7. 两条开发路径

### 7.1 app → mode（隔离）

新能力作为独立 app 开发，通过 mode 配置接入：

1. `apps/{group}/{name}/` 下创建 `APP.md` + `main.py`
2. 在目标 mode 的 `MODE.md` 中将 app 加入 `apps` 列表
3. app 在独立子进程中运行，通过 Matrix 总线与主进程通讯
4. app 可以有自己的 `pyproject.toml`，完全独立的依赖环境

**适用场景**：需要独立依赖、独立运行时、崩溃不拖垮主进程的能力。

### 7.2 src → manifests（复用）

新能力直接在 `MOSS/manifests/` 下声明，复用主进程运行时：

1. 写一个 Channel / Provider / Command 的 Python 类
2. 在对应的 manifest 文件中实例化
3. 下次启动自动生效

**适用场景**：轻量能力、需要直接访问 IoC 容器、开发和调试更便捷。

两条路径可以互转——先用 manifests 快速验证，稳定后拆成独立 app 获得隔离。

## 8. 知识探索路径

进入一个陌生的 MOSS 项目时的推荐探索序列：

```bash
# 第一步：定位与全貌
moss ws where                     # 哪种 workspace？在哪？
moss manifests explain            # manifest 体系自描述——整体理解

# 第二步：快速扫描
moss manifests contracts          # IoC 中已绑定了哪些服务？
moss manifests providers          # 这些服务由哪些工厂生产？
moss manifests channels           # 有哪些一级 Channel？
moss manifests primitives         # Shell 有哪些原语？
moss manifests resources          # 有哪些资源存储？
moss modes list                   # 有哪些模式？当前是哪个？
moss apps list                    # 有哪些 app？运行状态？

# 第三步：深入理解
moss modes show <name>            # 某个 mode 的详细配置
moss codex get-interface <path>   # 任何模块的接口契约

# 第四步：看源码
moss codex get-source <path>      # 需要理解实现时
```

原则：**先工具，后源码**。工具确认"有什么"，源码补充"怎么用"。

## 9. 相邻概念速览

- **Matrix**：消费 workspace 自举结果的总线。`MergedManifests` 就是 env + mode 的合并，Matrix 用它来初始化 IoC 容器和通讯网络。
- **Cell**：Matrix 网络中独立运行的节点。host、app、fractal、script 四种类型，每种都是 workspace 中的独立进程。
- **App**：workspace 内的独立子进程。通过 APP.md 声明启动方式，由 AppStore 管理生命周期。app 可被 mode 的 `apps` 白名单过滤。
- **Ghost**：MOSS 运行时之上的智能体适配层。Ghost 不直接依赖 workspace——它通过 Matrix 获取能力。但 ghost 的元信息（`GhostMeta`）同样从 workspace 约定的路径下发现。
- **Fractal**：workspace 的可递归连接。一个 workspace 的 Matrix 可以连接到另一个 workspace 的 Matrix，形成能力的分形共享。
- **Session**：一次 MOSS 运行时的通讯隔离域。同一 session scope 内的所有 Cell 共享通讯总线。
- **IoC Container**：所有 manifest 的最终归宿。Provider 注册依赖，Channel 获取依赖。`ghoshell_container:IoCContainer`。
- **Contract**：IoC 容器中的接口契约。Provider 声明"我能生产这个接口"，消费者说"我需要这个接口"。容器在中间撮合。

深入这些概念：

```bash
moss how-tos read get-moss-design/how-matrix-discovers-capabilities.md
moss how-tos read get-moss-design/how-ioc-container-work-in-moss.md
moss how-tos read get-moss-design/glossary.md
moss docs read architecture-topology.md
```
