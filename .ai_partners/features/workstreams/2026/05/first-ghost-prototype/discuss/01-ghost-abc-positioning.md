# 01 — Ghost ABC 定位与三层抽象

日期：2026-05-14

## 背景

`ghost.py` 定义了 `GhostFactory` 和 `Ghost` 两个 ABC。在行业 Agent 框架快速发展的背景下，需要重新审视它们的定位——从"完整实现规范"调整为"自解释 Adapter"。

## 讨论要点

### 1. Ghost ABC 的 Adapter 定位

- Ghost 不是要实现所有方法的规范，而是一个**自解释的 Adapter**
- 大部分方法是 **hook**（有默认返回值），不是 abstractmethod
- hook 存在本身 = code as prompt：让模型知道"这个能力在框架中可被扩展"
- 必须实现的 abstractmethod 有限：`articulate()`、`__aenter__/__aexit__`、`system_prompt()`

### 2. 三层抽象

```
GhostPrototype  = type[GhostFactory]     # class，定义一族 ghost 的"原型型号"
GhostBootstrapper = GhostFactory(...)    # instance，文件即配置，自解释可注册单元
GhostRuntime     = Ghost                 # instance，由 factory(container) 产出
```

- **GhostPrototype**：继承 `GhostFactory` 的类，定义型号名和版本。可以直接 `isinstance` 检查。
- **GhostBootstrapper**：`GhostFactory` 的实例，放在约定目录下，通过 manifests 机制被系统自动发现。携带元信息（name、nuclei、contracts）让系统在实例化前就能理解其协议。
- **GhostRuntime**：`Ghost` 实例，由 `GhostFactory.factory(container)` 产出。运行时状态，持有 meta 引用回到 Bootstrapper。

### 3. 文件即注册

注册目录：`[workspace]/src/MOSS/ghosts/`

- 每个 module/package（不递归）包含一个 `GhostFactory` 实例
- 文件系统即命名空间：一个文件 = 一个 ghost 注册，互不冲突
- 发现逻辑：`isinstance(obj, GhostFactory)` 校验，已有成熟模式可复用
- 发现和注册机制放在后续步骤讨论

### 4. 最小化实现

- 第一个 Ghost 原型的目标：产出 **参照基线**
- 最小实现 = 注册到 manifests 里的一个 `GhostFactory` 实例
- 其他 Ghost 原型都以此为模式来扩展

## 决策结论

1. Ghost ABC 定位为 **自解释 Adapter**，hook 为主，abstractmethod 为辅
2. 三层抽象：Prototype(class) → Bootstrapper(instance) → Runtime(instance)
3. 文件即配置：一个文件 = 一个 GhostFactory 实例 = 一个 Ghost 注册
4. 本轮产出最小化实现，作为后续所有 Ghost 原型的参照基线
5. 需要在 `ghost.py` 中以注释/伪代码方式体现上述关系

## 后续影响

- 可能需要微调 Ghost ABC 的 method 签名（从 abstractmethod 降级为 hook）
- 需要创建 `workspace/src/MOSS/ghosts/` 下的注册示例
- 发现机制已有成熟模式，实现放在后续步骤
