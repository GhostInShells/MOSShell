---
title: Register Capabilities via Manifests
description: 在 MOSS 环境中注册新能力：providers、channels、configs、nuclei、resources、topics 的声明方法。面向需要在 workspace 中添加能力的开发者（AI 和人类）。
---

# How to Register Capabilities via Manifests

## 背景

MOSS 不需要 `app.register()` 或 `config.yaml`。能力注册的方式是：**在约定路径下写一个 Python 实例，下次启动自动生效。**

Matrix 启动时扫描 `MOSS.manifests/` 下的文件，通过 `isinstance`（或 topics 的 `issubclass`）过滤目标类型，自动注入 IoC 容器和通讯总线。

先了解当前环境已有的声明：

```bash
moss manifests explain            # 完整自解释——所有声明类型的表格
moss manifests providers          # 当前 IoC 绑定
moss manifests channels           # 当前主 channel
moss manifests contracts          # IoC 中已绑定的接口
```

## 7 种 manifest 类型速览

| 类型 | 文件 | 声明方式 | 检测 | 键 |
|------|------|---------|------|---|
| Provider | `providers.py` | `my_provider = MyProvider()` | `isinstance(obj, Provider)` | `contract()` import path |
| Channel | `channels.py` | `main = new_shell_main_channel()` | `isinstance + name == "__main__"` | `"__main__"` |
| Config | `configs.py` | 类引用: `TTSManagerConfig` 或实例覆盖: `tts = TTSManagerConfig(...)` | `issubclass(obj, ConfigType)` 或 `isinstance(obj, ConfigType)` | `conf_name()` |
| Nucleus | `nuclei.py` | `factory = ExampleNucleusMeta()` | `isinstance(obj, NucleusMeta)` | `name()` |
| Resource | `resources.py` | `meta = LocalImageResourceMeta()` | `isinstance(obj, ResourceStorageMeta)` | `{scheme}:{host}` |
| Topic | `topics.py` | 定义或 import `TopicModel` 子类 | `issubclass(obj, TopicModel)` | `topic_name` |
| CTML | `ctml_versions/` | 放 `.md` 文件 | 文件扫描 | filename |

## 注册 Provider（最常用）

Provider 声明"这个接口由这个工厂生产"。放在 `MOSS/manifests/providers.py`：

```python
from ghoshell_container import Provider
from my_module import MyService

class MyServiceProvider(Provider):
    def contract(self):
        return MyService          # 接口的 Python import path
    def singleton(self):
        return True               # True=单例, False=每次工厂
    def factory(self, container):
        return MyService(...)

my_service_provider = MyServiceProvider()
```

**约定**：变量名自解释，一个文件里放多个 Provider 实例。

**Mode 继承**：默认 `from MOSS.manifests.providers import *`，然后追加 mode 专属的。

深入：`moss howtos read get-moss-design/how-ioc-container-work-in-moss.md`

## 注册 Channel（能力树入口）

Channel 是 CTML shell 的能力树根。放在 `MOSS/manifests/channels.py`，必须定义 `name == "__main__"` 的 Channel 实例：

```python
from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.ctml.shell.ctml_main import inject_system_primitives

main = new_shell_main_channel(description="My main channel")

# 注册系统原语（sleep/noop/observe/interrupt 等）
inject_system_primitives(main)

# 挂载子 channel
main.import_channels(AppStoreChannel(name='apps'))
main.with_module(SpeechChannelModule())
```

> **原语不再有独立的 primitives.py。** Shell 原语（sleep/noop/observe/interrupt）通过 `inject_system_primitives()` 在 channels.py 中直接注册到 main channel。

**Mode 继承**：mode 的 `__main__` channel 是运行时唯一生效的。可以从零构建（`new_default_shell_main_channel()`），也可以 `from MOSS.manifests.channels import main` 复用全局的再增量改造。

深入：`moss codex blueprint channel_builder`

## 注册 Config（两种注册语义）

扫描逻辑自动区分类的类型和实例变量：

```python
from ghoshell_moss.host.providers.audio_player_provider import AudioPlayerConfig

# 1. Type[ConfigType] 类引用 → 文件持久化（系统默认）
#    放类本身，不实例化。Bootstrap 时 get_or_create() 读 YAML/写默认。
AudioPlayerConfig = AudioPlayerConfig  # re-export

# 2. ConfigType 实例 → 内存覆盖（运行时覆盖）
#    实例化并带值。Bootstrap 时仅 set_config(override=False)，不写磁盘。
tts_override = TTSManagerConfig(default_speaker="大壹")
```

两种语义：

1. **类引用（Type[ConfigType]）**：`ConfigStore.get_or_create(config)` — 优先从 `workspace/configs/{conf_name}.yml` 读文件，不存在才用类的默认构造写文件。文件是 truth source。
2. **实例变量（ConfigType 实例）**：`ConfigStore.set_config(config, override=False)` — 只更新内存缓存，不写磁盘。mode 用此语义覆盖全局默认值，不会污染 workspace/configs/。

全局 manifest 放类，mode manifest 通过 `from MOSS.manifests.configs import *` 继承类，然后实例化需要覆盖的配置。

每个 ConfigType 子类必须实现 `conf_name()` 返回唯一标识（即 YAML 文件名）。

深入：`moss codex get-interface ghoshell_moss.contracts.configs`

## 注册 Nucleus（感知核工厂）

NucleusMeta 是生产感知核（Nucleus）的工厂。放在 `MOSS/manifests/nuclei.py`：

```python
from ghoshell_moss.core.blueprint.mindflow import NucleusMeta, Nucleus, InputSignalMeta
from ghoshell_container import IoCContainer


class MyNucleusMeta(NucleusMeta):
    def name(self) -> str:
        return "my_nucleus"

    def description(self) -> str:
        return "描述这个感知核做什么"

    def signals(self) -> list[type[SignalMeta]]:
        return [InputSignalMeta]

    def factory(self, container: IoCContainer) -> Nucleus:
        return MyNucleus(...)


my_nucleus_factory = MyNucleusMeta()
```

**Mode 继承**：默认 `from MOSS.manifests.nuclei import *`，同 `name()` 的可以覆盖。

深入：`moss codex get-interface ghoshell_moss.core.blueprint.mindflow`

## 注册 Resource Storage

放在 `MOSS/manifests/resources.py`：

```python
from ghoshell_moss.core.resources.local_image import LocalImageResourceMeta

local_image_storage_meta = LocalImageResourceMeta()
```

ResourceStorageMeta 实例声明一个可寻址的资源数据集，以 `{scheme}:{host}` 为全局资源句柄。`factory()` 生产 `ResourceStorage`。

深入：`moss howtos read host-dev/add-a-resource-storage.md`

## 注册 Topic（协议声明）

Topic 是纯协议声明，无副作用。放在 `MOSS/manifests/topics.py`：

```python
from ghoshell_moss.core.concepts.topic import TopicModel

class MyTopic(TopicModel):
    """描述这个 topic 的用途"""
    payload: str

    @classmethod
    def topic_type(cls) -> str:
        return "my_topic"
```

类定义本身即是注册——出现在模块命名空间就会被 `scan_package` 通过 `issubclass(obj, TopicModel)` 发现。

> 注意：`SignalMeta` 子类（如 `InputSignal`）不属于 Topic。Signal 是 Mindflow 的输入信号，在 `nuclei.py` 中通过 `NucleusMeta.signals()` 引用，不应在 topics.py 中声明。

深入：`moss codex get-interface ghoshell_moss.core.concepts.topic:TopicModel`

## 验证注册

```bash
# 检查你的声明是否被发现
moss manifests providers | grep -i "my_service"
moss manifests channels
moss manifests configs
moss manifests nuclei

# 完整视图（mode 的能力声明）
moss manifests explain
```

## 文档目标

读者按照本文档操作，应该能够：
1. 在正确的 manifest 文件中添加声明，知道用什么类型和键语义
2. 区分 Provider（工厂）、NucleusMeta（工厂）、ConfigType（文件持久化）、TopicModel（协议声明）的不同注册方式
3. 理解 mode 的显式继承模式（from MOSS.manifests.xxx import * + 扩展）
4. 通过 `moss manifests explain` 验证声明是否被正确发现
