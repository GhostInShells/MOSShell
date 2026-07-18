# MOSS Project — Declaration Ecosystem Design

MOSS 项目层（Project）的核心机制是**声明生态**：不用运行时注册，用磁盘上可读的静态声明描述"这个 workspace 里有什么、如何组装、按什么规矩隔离"。本文档给出这个生态的**设计思想**——每个概念存在的理由、彼此的边界。

字段级细节不抄。所有具体类型、模块路径、当前环境的实际声明，通过以下三条查询路径获取：

```
moss codex architecture                     # 模块地图, 定位实现路径
moss codex get-interface <module:type>      # 字段/方法级契约反射
moss manifests <category>                   # 当前 workspace 的实际声明
```

改代码遇到本文档过期，就地修正——本文档紧邻 mode/manifests 实现代码（`src/ghoshell_moss/project/`），forcing function 就是这个位置本身。

---

## 1. 为什么用声明替代注册

传统框架：应用启动时执行注册代码（`container.register(X, ProviderY)`），依赖顺序、副作用、隐式覆盖。问题：

- **不可静态检查**：出问题只有跑起来才知道。
- **依赖顺序敏感**：谁先 import 决定谁生效。
- **git diff 不承载语义**：改注册代码看不出 capability 拓扑变化。

MOSS 的替代方案：**磁盘上的声明包**（Python 模块 + `HOST.md` / `MOSS.md`），扫描发现，无副作用组装。收益：

- **自解释**：`moss manifests <category>` 直接列出当前有什么。
- **路径可查**：每条声明的 "Found At" 是真实文件路径。
- **git diff 有意义**：改声明就是改能力拓扑，PR 里一眼看到。

代价：需要遵守目录约定（HOST 包结构、命名规则）。收益远大于代价，因为改声明本身是低频事件。

---

## 2. 两层结构：Matrix × Mode

声明分两层。所有 `moss manifests <category>` 都按这个结构展示。

### Matrix layer — `MOSS.manifests.*`

Workspace 根目录下的 `.moss/src/MOSS/manifests/`，是**跨 mode 共享的 baseline**。所有 mode 都自动继承（通过 Python `import`），是"这个 workspace 的进程级公共设施"。

典型声明：Session（zenoh 通讯总线）、ConfigStore、ResourceRegistry、TopicService、Subprocesses——所有 mode 都要用到的基础契约。

### Mode layer — `HOST.*`（per mode）

`.moss/modes/<name>/src/HOST/`，是**特定 mode 的 capability 包**。通过 Python `from MOSS.manifests.X import *` 继承 Matrix 层，然后追加或覆盖。

典型声明：TTS/Speech/AudioPlayer（default mode 才装的 IO）、mode 专属的 channel 组合、mode 专属的 ghost 配置。

### 组合语义

- Matrix 声明 X → 所有 mode 都能拿到 X。
- Mode 声明同名 X → 覆盖 Matrix 的 X（Python import 语义）。
- Mode 声明新的 Y → 只有本 mode 有 Y。

运行时 `Environment.discover()` 认出当前 mode，`Project.current_mode().manifests()` 返回**合并后**的视图。`moss manifests providers` 输出的 `"N from MOSS.manifests, M effective in mode"` 就是这层信息。

---

## 3. 每种声明类型的定位

八类声明，各管一件事。不要混用。

| 类型 | 目录 | 管什么 | 何时看 |
|---|---|---|---|
| `providers` | `providers/` | IoC container 里的服务（Provider → contract） | 用 `container.get(Contract)` 前想知道有没有 |
| `configs` | `configs/` | 强类型配置模型（Pydantic），从 ConfigStore fetch | 需要一份 typed 配置时 |
| `topics` | `topics/` | 事件 topic schema（pub/sub 契约） | 定义跨 cell 事件时 |
| `signals` | `signals/` | Mindflow signal schema（感知输入契约） | 从 channel 向 ghost 发信号时 |
| `resources` | `resources/` | 命名资源存储（VFS scheme + host 路由） | 需要跨 mode 定位一份磁盘资源时 |
| `parameters` | `parameters/` | 单值参数（可 mode 覆盖 Matrix baseline） | 环境变量式的少量运行时参数 |
| `channels` | `channels.py`（**只 mode 有**） | mode 的 `__main__` CTML channel | 定义 mode 暴露给模型的能力集时 |
| `nuclei` | `nuclei/`（**只 mode 有**） | Mindflow nucleus 工厂（信号→冲动的仲裁器） | 定制 mode 的思维模式时 |

**规矩**：每类声明在一个子目录（`__init__.py` 内定义模块级变量），Matrix scanner 反射变量类型识别。字段级细节走 `moss codex get-interface`——例如：

```
moss codex get-interface ghoshell_moss.core.blueprint.project:Manifest
moss codex get-interface ghoshell_moss.core.blueprint.matrix_manifest:MatrixManifest
```

---

## 4. Modes / Ghosts / Networks — 三种运行时坐标

声明是**内容**，Modes/Ghosts/Networks 是**装内容的容器**，是运行时的坐标轴。四维坐标 `(mode × ghost × network × cell)` 决定 scope 隔离（见 `matrix.runtime_scopes()`）。

### Modes — capability isolation

Mode 不是"环境切换"，是"能力包"。选 `default` mode 得到全套 TTS/Speech/Audio；选 `system_test` mode 得到最小能力集。同一个 workspace 可以有 N 个 mode，运行时选一个。

创建：`moss modes create <name>` 从 stub 拷贝目录结构，然后编辑 `HOST.md` + `src/HOST/` 内的声明包。

### Ghosts — agent identity

Ghost 是 mode 之内的**智能体身份**——记忆、prompt、行为模式。同一个 mode 可以承载多个 ghost，运行时选一个。Ghost 声明在 `.moss/ghosts/<name>/` 或作为 Python class 暴露；不用 manifests 目录扩展，因为 ghost 的构造已经用 Python 类承载了自解释性。

### Networks — communication scope

Zenoh 通讯层的作用域配置。同一个 mode 可以运行在不同 network 下（local / dev / prod）。声明在 `.moss/networks/<name>.yml`。

运行时选择通过 `moss --mode X --ghost Y --network Z --scope S <command>` 全局 flag，或 `MOSS.md` 里的默认值。

---

## 5. 查询路径（不抄细节，指路）

想知道**当前 workspace 到底有什么声明**：

```
moss manifests providers                 # 当前所有 provider (Matrix + 当前 mode)
moss manifests configs
moss manifests topics
moss manifests signals
moss manifests resources
moss manifests parameters
moss manifests channel                   # 当前 mode 的 main channel 树
moss manifests nuclei                    # 当前 mode 的 mindflow nuclei
moss manifests explain                   # manifest 系统的自描述
```

想知道**某个类型的字段/方法契约**：

```
moss codex get-interface <module:type>
```

想知道**实现代码在哪**：

```
moss codex architecture                  # 手工策展的模块地图
moss codex where <module>                # 单模块的定义路径
```

想知道**mode/ghost 的物理位置**：

```
moss modes show <name>                   # 展示 Home、HOST.md、Node Paths
moss ghosts show <name>                  # 展示 Prototype、Import Path、Source File
```

---

## 6. 常见误区

- **在运行时用 `container.register()`**：违背声明生态。要新增 provider 就写进 `providers/__init__.py`。
- **把 mode 当"环境变量集"**：mode 是能力包，环境变量放 `.env`。
- **在 Matrix 层声明只有一个 mode 需要的服务**：污染 baseline，其他 mode 也被迫承受。放到那个 mode 的 HOST。
- **改字段不改 DESIGN.md 里的字段列表**：本文档故意不列字段，就是为了避免这类漂移。字段查询走 `moss codex get-interface`。
- **同一份声明既在 Matrix 又在 Mode**：Mode 层的会覆盖 Matrix 层的（Python import 语义），是合法但危险的用法——通常表示分层没想清楚。
