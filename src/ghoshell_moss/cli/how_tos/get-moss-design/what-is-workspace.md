---
title: What is a MOSS workspace
description: MOSS workspace 是什么、为什么存在、它的目录约定和自举机制如何让 AI 协作者不依赖人类指导就能理解环境、发现能力、迭代代码。面向首次进入 workspace 的 AI 化身和人类开发者。
---

# What is a MOSS Workspace

## 背景

你进入了一个 MOSS 项目。你看到 `.moss_ws/` 目录，或者项目根目录下有 `MOSS.md` 文件。

这不是一个普通的配置目录。它是一个**文件系统级的 IoC 自举层**——通过目录约定替代生命周期注册，让 AI 协作者可以在正确的位置写代码，下次启动自动生效。

### 先理解核心概念

在深入 workspace 之前，确认你理解了这些词：

```bash
moss how-tos read get-moss-design/glossary.md
```

最少需要理解的：
- **Channel** — MOSS 的通用能力单元
- **Matrix** — 自迭代组网底座，自动发现并集成能力
- **manifests** — 声明体系，告诉 Matrix 环境里有什么
- **mode** — 能力的过滤/叠加视图

### 这个文档解决什么问题

传统框架需要你理解三件事才能注册一个能力：
1. 在哪里注册（装饰器？`app.add_provider()`？`config.yaml`？）
2. 什么时候注册（启动前？运行时？懒加载？）
3. 怎么注入（构造函数？属性？service locator？）

MOSS 的回答：**放到约定路径下，写一个 Python 实例。** Matrix 启动时自动发现、注入。

这个设计的前提：有一个能读代码、理解约定的 AI。人的记忆容量不适合这套约定，但 AI 可以。

## workspace 的两种形态

```bash
moss ws where          # 看你当前的 workspace 在哪
```

`Environment.find_workspace_path()` 按优先级查找：

1. `MOSS_WORKSPACE` 环境变量
2. 当前目录下的 `.moss_ws/` 目录
3. 向上递归查找带 `MOSS.md` 的目录（项目即 workspace）
4. `~/.moss_ws/`（默认）

**这意味着两种 workspace 形态：**

- **用户 workspace**（`~/.moss_ws/`）：全局环境，所有项目的 MOSS 运行时共享
- **项目 workspace**（带 `MOSS.md` 的仓库根）：项目自带 workspace，开发时使用

你当前在项目 workspace 中。

## 目录结构

```bash
# 看 stub 模板（init_workspace 复制的来源）
moss codex get-source ghoshell_moss.host.stubs.workspace
```

```bash
# 列出实际的 workspace
ls -la $(moss ws where 2>/dev/null | grep "Expect Root" | cut -d'|' -f3 | xargs)
```

```
.moss_ws/
  MOSS.md              # MossMeta：环境名称、CTML版本、默认mode、system_prompt
  CLAUDE.md            # 给 AI 协作者的引导（本文档结束后就应该写它）
  .env / .env.example  # 环境变量
  .gitignore
  __init__.py          # workspace 是 Python package

  src/MOSS/
    manifests/         # ★ 全局能力声明（所有 mode 共享）
      providers.py     #   IoC 依赖声明
      channels.py      #   一级 Channel
      primitives.py    #   Shell 原语
      configs.py       #   配置模型
      resources.py     #   资源存储
      topics.py        #   事件协议
      nuclei.py        #   感知核（Mindflow）

    modes/             # ★ 模式声明（叠加在全局之上）
      {mode_name}/
        MODE.md        #   模式的描述
        __init__.py
        primitives.py  #   模式专属原语（如 system_test 多了 loop/sample/branch）
        contracts.py   #   模式专属契约

  apps/                # ★ App 目录
    {group}/{name}/
      APP.md           #   App 的描述（相当于 CLAUDE.md）
      main.py          #   App 入口

  configs/             # 运行时配置（zenoh、logging、circus）
  runtime/             # 运行时数据（sessions、conversations、logs、locks）
  assets/              # 静态资源
  ctml_versions/       # CTML 提示词版本
```

### 关键文件解释

**MOSS.md** 是 workspace 的元信息：

```bash
# 查看你的环境里 MOSS.md 的内容
moss codex get-interface ghoshell_moss.core.blueprint.environment:MossMeta
```

**MOSS/manifests/** 下的每个 `.py` 文件都不是配置文件——它们实例化 Python 对象：

```bash
# 看一个实际的 manifests
cat src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/primitives.py
cat src/ghoshell_moss/host/stubs/workspace/src/MOSS/manifests/configs.py
```

核心模式：`foo = SomeType()` → Matrix 启动时 `scan_package` 发现并注入。

## 启动链路

从发现到执行，分为两层：

### 第一层：Host（发现层，无副作用）

```bash
moss codex get-source ghoshell_moss.host.impl:Host
```

```
Host.__init__():
  Environment.discover() + bootstrap()
    → .env 加载, src/ 加入 sys.path
  PackageManifests.from_environment(env)
    → scan_package MOSS.manifests.*  找到所有声明
  list_modes_from_root_package()
    → scan_package MOSS.modes.*      找到所有模式
  MergedManifests([env_manifest, mode_manifest])
    → mode 覆盖 env（右边优先）
  MatrixImpl(manifest, ...)
    → 创建 IoC 容器，暂不启动
```

Host 是"知识态"——CLI 的 `moss manifests`、`moss modes`、`moss apps` 都只需要 Host，不需要 Runtime。

### 第二层：MossRuntimeImpl（执行层）

```bash
moss codex get-source ghoshell_moss.host.runtime:MossRuntimeImpl
```

```
Host.run() → MossRuntimeImpl.__aenter__():
  Matrix.start()
    → 启动 zenoh 通讯
    → 遍历 manifests.providers() → 逐个 factory(container) → 注入 IoC
    → 遍历 manifests.bootstrappers() → 逐个 bootstrap(container)
  AppStore.start()
    → bringup 配置的 apps（独立子进程）
  CTMLShell.start()
    → 启动 shell，refresh channel metas
```

### manifests 如何被自动发现

```bash
moss codex get-source ghoshell_moss.host.manifests.providers
moss codex get-source ghoshell_moss.host.manifests.channels
```

所有 manifest 类型的发现逻辑一致：
1. 拼出约定路径（如 `MOSS.manifests.providers`）
2. `scan_package` 用 `importlib` 加载模块
3. 用 `isinstance` 过滤目标类型
4. 返回发现结果

### MergedManifests 合并逻辑

```bash
moss codex get-source ghoshell_moss.host.manifests:MergedManifests
```

极其简单：遍历 manifests 列表，`dict.update()` / `list.extend()`，右边优先。同类别内合并——providers 只覆盖 providers，不会跨类别。

## AI 迭代模式

### 模式一：声明新能力（增加 manifest）

在 `MOSS/manifests/providers.py` 里写一个 Provider 实例：

```python
from ghoshell_container import Provider
from my_module import MyService

class MyProvider(Provider):
    def contract(self): return MyService
    def singleton(self): return True
    def factory(self, container): return MyService(...)

my_provider = MyProvider()
```

**下次启动自动生效。** 不需要碰启动代码。

### 模式二：创建新 App

在 `apps/{group}/{name}/` 下放 `APP.md` + `main.py`：

```bash
# 看已有的 app 示例
ls apps/_system_tests/helloworld/
cat apps/_system_tests/helloworld/APP.md
cat apps/_system_tests/helloworld/main.py
```

App 会在独立子进程中运行，通过 Matrix/zenoh 与主进程通讯。

### 模式三：创建新 Mode

```bash
# 看已有 mode 的结构
moss modes list
moss modes show default
```

Mode 目录下写 `MODE.md` + manifest 文件，该 mode 就拥有了专属的能力视图。

### 在迭代前先了解环境

```bash
moss manifests providers    # 当前注入哪些依赖
moss manifests channels     # 当前有哪些 channel
moss manifests primitives   # 当前有哪些 shell 原语
moss manifests resources    # 当前有哪些资源存储
moss modes list             # 当前有哪些模式
moss apps list              # 当前有哪些 app

# 更多工具
moss --help
```

## workspace 的创建

```bash
# 创建新 workspace（交互式）
moss ws init

# 非交互式（AI / 脚本）
moss ws init --cwd -y      # 当前目录，跳过确认
moss ws init /path -y      # 指定路径，跳过确认

# 复制环境变量模板
moss ws copy-env
moss ws copy-env --force   # 覆盖已有 .env

# 查看当前 workspace
moss ws where
```

## 与其他概念的关系

- **Matrix** 是消费 workspace 自举结果的总线。它的 `MergedManifests` 就是 env + mode 的合并。
- **mode** 是 workspace 的视图。不同 mode 看到不同的能力子集。
- **App** 是 workspace 内的独立进程。Matrix 自动发现并管理其生命周期。
- **Fractal** 让 workspace 可以递归——一个 workspace 的 Matrix 可连接到另一个 workspace 的 Matrix。

深入这些概念：

```bash
moss how-tos read get-moss-design/how-matrix-discovers-capabilities.md
moss how-tos read get-moss-design/how-ioc-container-work-in-moss.md
moss how-tos read get-moss-design/glossary.md
```

## 常见问题

### 问题：init_workspace 创建后，怎么让 workspace 可用？

`init_workspace` 从 stub 复制了模板。创建后需要：
1. `moss ws copy-env` — 创建 .env 文件
2. 编辑 .env 配置 API key 等凭据
3. 在 manifests 里声明你的能力（providers、channels 等）

### 问题：workspace 的 src/ 为什么在 sys.path 上？

`Environment.bootstrap()` 调用了 `sys.path.append(source_path)`。
这意味着 workspace 里的 Python 模块可以直接被 import。
这是运行时自迭代的物理基础——AI 创建的模块运行时可见。

### 问题：怎么判断我在哪种 workspace 里？

```bash
moss ws where
```

看 "Expect Root" — 如果路径是 `.moss_ws/`，你在目录型 workspace 里；如果是项目根目录，你在项目型 workspace 里。

## 文档目标

读者按照本文档操作，应该能够：
1. 用 `moss ws where` 确认当前 workspace 位置
2. 用 `moss manifests providers/channels/primitives` 列出环境中的能力
3. 理解 workspace 的目录约定和 manifests 发现机制
4. 在 manifests 目录下添加声明，并在下次启动后验证生效
5. 用 `moss ws init --cwd -y` 创建一个新的 workspace
6. 通过本文档的链接进一步阅读 glossary、IoC、Matrix 等深层文档
