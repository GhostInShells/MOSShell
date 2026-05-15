# 04 — Ghost Runtime 集成方式

日期：2026-05-14

## 背景

Ghost Runtime 需要和 MossRuntime 协同工作——共享生命周期、注册 Mindflow、编排 Shell 启动顺序。需要确定集成架构。

## 讨论要点

### 1. SystemPrompter 暴露

**问题**：当前 `SystemPrompter` 通过 `Matrix._prepare_system_prompter()` 内部组装，Ghost 开发者只能拿到 `moss_instruction()` 压平后的字符串，看不到可组装的各层。

**方案**：继承 `BaseSystemPrompter`，新增 `MossSystemPrompter`（放在 `contracts/system_prompter.py`）：

```python
class MossSystemPrompter(BaseSystemPrompter):
    """每个组装单元用命名方法显式声明，code as prompt。"""

    def ctml_instruction(self) -> str: ...
    def environment_instruction(self) -> str: ...
    def mode_instruction(self) -> str: ...
    def static_instruction(self) -> str: ...

    # instruction() 继承自 BaseSystemPrompter，默认全量拼合
```

Ghost 开发者拿到 `MossSystemPrompter` 实例，IDE 自动补全即可看到四层。函数签名本身就是自解释，不需要魔法字符串。

Ghost 的 `system_prompt()` 实现：

```python
def system_prompt(self) -> str:
    prompter = self._runtime.system_prompter
    return self._soul + "\n\n" + prompter.instruction()
```

### 2. GhostRuntime 集成架构

三种方案：

| # | 方案 | 判定 |
|---|------|------|
| 1 | **包裹** MossRuntime，同接口 + Ghost 生命周期编排 | 选中 |
| 2 | 平级实现，独立生命周期 | 多此一举 |
| 3 | 继承 MossRuntimeImpl | 耦合实现细节 |

**方案 1 详细设计**：

`GhostRuntime` 实现 `MossRuntime` ABC，内部持有一个 `MossRuntimeImpl` 做业务透传。只负责**生命周期编排**。

```
GhostRuntimeImpl.__aenter__:
    AsyncExitStack:
        1. Matrix 启动           (→ MossRuntimeImpl)
        2. Ghost 实例化          (GhostMeta.factory(container))
        3. async with ghost      (Ghost 自身生命周期)
        4. Mindflow 准备         (ghost.mindflow() or default, 注册 nuclei)
        5. Session signal 回调   (session.on_signal → mindflow.add_signal)
        6. Shell 启动            (→ MossRuntimeImpl)
        7. Mindflow loops 启动   (三循环)
```

### 3. 生命周期约束

- Ghost 实例化必须在 Matrix 启动后（依赖 IoC container）
- Ghost 启动不依赖 Shell 启动
- Mindflow 不由 Ghost 自己启动，由 GhostRuntime 编排
- 所有 MossRuntime 业务方法（moss_exec/moss_observe/...）透传给内部 MossRuntimeImpl

### 4. 三循环原型参考

`tests/ghoshell_moss/core/mindflow/test_base_mindflow.py` 中的 `MindflowSuite` 提供了完整的三循环原型：

```
_main_loop:       mindflow.loop() → distribute articulate/action to janus.Queue
_articulate_loop: read queue → async with articulate → articulate_func(articulate)
_action_loop:     read queue → async with action → action_func(action)
```

跨线程通讯通过 `janus.Queue` + `threading.Barrier` 同步。v0 可能不需要多线程，但 MindflowSuite 证明了多线程是可行的。

## 决策结论

1. 新增 `MossSystemPrompter(BaseSystemPrompter)`，四个命名方法暴露可组装层
2. GhostRuntime 采用**包裹模式**——实现 MossRuntime ABC，内部透传给 MossRuntimeImpl
3. 生命周期由 GhostRuntime 统一编排：Matrix → Ghost → Mindflow → Shell → Mindflow loops
4. 三循环原型参考 `MindflowSuite` 的实现模式
5. v0 单线程运行，多线程作为后续优化

## 后续影响

- 需创建 `GhostRuntime` 类（可在 `ghosts/atom/_runtime.py` 或独立文件）
- 需创建 `MossSystemPrompter` 在 `contracts/system_prompter.py`
- MossRuntimeImpl 的 Matrix 启动逻辑需要可被 GhostRuntime 复用
