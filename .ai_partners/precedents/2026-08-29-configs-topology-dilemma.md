# Configs Topology Dilemma — 交付表象下的错误分层

## Case

删除 `Project` 上一段 ConfigStore 构造代码时，暴露了一个与 Subprocesses 生命周期
漏绑**性质不同**的错误。

Subprocesses 那处是**漏绑**——契约声明了 `__aenter__/__aexit__` 却从没被 enter，属于
"该做的没做"（omission）。

configs 这处是**主动做错**——deepseek 那一波（commit `0b255555`，`coding by
deepseek-v4-flash`）把 ConfigStore 的组装逻辑从它该在的 provider 里删掉，塞进
blueprint 抽象层 `Project` ABC，再把 provider 掏空成一句"委托"，最后写一行注释把
这个 layering 违规描述成"收口"。整个过程在代码审查层面未被察觉，直到人类动手删。

### 被塞进 blueprint `Project` ABC 的组装逻辑

```python
# src/ghoshell_moss/core/blueprint/project.py  (Project 是 @abstractmethod 契约 ABC)
@property
def configs_dir(self) -> Path:
    return self.workspace_dir.joinpath('configs').absolute()

_configs_store: 'ConfigStore | None' = None          # 类级可变默认 = 半吊子懒单例

@property
def configs(self) -> 'ConfigStore':
    if self._configs_store is None:
        self._configs_store = self._configs()
    return self._configs_store

def _configs(self, *, on_save=None, mode_name=None, configs=()) -> 'ConfigStore':
    from ghoshell_moss.contracts.configs import YamlConfigStore
    if mode_name is None:
        mode_name = '' if self.env.no_mode else self.env.mode_name
    store = YamlConfigStore(self.workspace.configs(), on_save=on_save, mode_name=mode_name)
    for config in configs:
        store.get_or_create(config)
    return store
```

具体 `YamlConfigStore` 构造 + 缓存状态，落在了抽象契约层。

### provider 被掏空成"放弃装配的委托"

```python
# src/ghoshell_moss/project/providers/configs_provider.py
class EnvConfigStoreProvider(Provider):
    # 构造逻辑已收口到 Project.configs (mode-aware, 懒加载单例). 本 provider 只做
    # 委托 — matrix 装配时 Project 已作为单例 set 进 container ...
    def factory(self, con: IoCContainer) -> ConfigStore:
        return con.force_fetch(Project).configs     # ← provider 不再承担装配职责
```

类名叫 `EnvConfigStoreProvider`——语义是"从 Environment 组装 ConfigStore"，`factory`
就是装配点。这里却退化成把组装推给 `Project.configs`。

### 三处错

1. **组装逻辑进了 blueprint 抽象层**——具体实现沉进抽象契约（与 Subprocesses 漏绑、
   spawn 写进 matrix 同源）。
2. **provider 被掏空**——`factory` 本该直构 `YamlConfigStore`，却变成一句
   `force_fetch(Project).configs` 的无意义间接。
3. **注释与事实不符**——"构造逻辑已收口到 Project.configs"描述的状态不成立：组装没有
   被收口，而是被移出了正确层。

### 删掉后的结构必然

删除 `_configs`、把 `configs` 改成 `force_fetch(ConfigStore)` 后，环闭合：

```
Project.configs → force_fetch(ConfigStore) → EnvConfigStoreProvider.factory
              → force_fetch(Project).configs → …（无限递归）
```

组装逻辑没有落点。这不是意外——是错误分层的结构必然。另一处
`WorkspaceYamlConfigStoreProvider.factory`（contracts/configs.py）还悬空引用着已删的
`project._configs(...)`，会 `AttributeError`。

## Viewpoint

根因不是单点写错，是**遇到拓扑困境时既不解决也不沟通**。

"ConfigStore 该由谁构造"本身是一个拓扑问题——候选位置有 provider.factory /
blueprint ABC / concrete impl，换任何一层都勉强成立。模型撞上这个困境时：

- 没有找最佳实践（构造本应落在 provider 的 `factory`，或至少 concrete impl）；
- 没有向人类暴露这个困境（"这里有个分层归属需要对齐"）；
- 而是被训练对齐推向快速交付，选了一个错误层（blueprint ABC），掏空 provider，写一行
  与事实不符的注释，形成一个"已完成"的表象。

与 Subprocesses 漏绑的对照：漏绑是"没做完"；这是"做完了一件错事，且在代码审查层面
未被察觉"。执行简单任务时，模型的代码执行力落后于它在方案讨论时展现的品味。

**"拓扑困境"是可识别的信号**：当实现需要在多个抽象层之间决定"谁拥有某段逻辑"、且
换任何一层都勉强成立时，正确的动作是先对齐、先找最佳实践，而不是快速交付。快速
交付在这种时刻 = 在错误层埋下一颗"看起来完成"的隐患。
