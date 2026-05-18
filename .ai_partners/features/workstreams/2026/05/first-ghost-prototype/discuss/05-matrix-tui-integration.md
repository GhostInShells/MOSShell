# 05 — Matrix / TUI 集成

日期：2026-05-14

## 背景

GhostMeta 注册到 `workspace/src/MOSS/ghosts/` 后，需要被系统发现并集成到运行时。TUI 需要从"CTML 测试 REPL"变成"和 Ghost 对话"。

## 讨论要点

### 1. 发现机制归属

Manifests 现有的发现模式：

```python
PackageManifests.channels()    → scan_package("MOSS.manifests.channels")    → isinstance(obj, Channel)
PackageManifests.providers()   → scan_package("MOSS.manifests.providers")   → isinstance(obj, Provider)
PackageManifests.nuclei()      → scan_package("MOSS.manifests.nuclei")      → isinstance(obj, NucleusMeta)
```

Ghost 不放入 Manifests。理由：

- Manifests 管理的是 MOSS 的**能力声明**（channel、provider、primitive、nucleus 等）
- Ghost 是**完整的智能体注册**，不是能力片段
- 类比：`MossHost` 管理 `modes`（`all_modes()`），ghost 是同级别的顶层概念

### 2. Host 层发现

```python
# MossHost ABC 新增
def ghosts(self) -> dict[str, GhostMeta]: ...
```

实现方式：扫描 `MOSS.ghosts` package，`isinstance(obj, GhostMeta)` 过滤。技术上复用 `scan_package` + `iter_members` + `isinstance` 模式。

### 3. TUI 架构（现有）

```
MossHostTUI[Runtime]           # 泛型 TUI 框架
  ├── _get_runtime(host)       # 子类决定 runtime 类型
  ├── create_states()          # 子类定义面板
  ├── _main_loop()             # enter runtime + start states + input_loop
  └── _input_loop()            # 读输入 → current_state.handle_input()
```

`MossRuntimeTUI(MossHostTUI[MossRuntime])`：
- `_get_runtime()` → `host.run()` 返回 MossRuntime
- `create_states()` → `MOSSRuntimeREPLState` + `FractalServeState`

### 4. Ghost TUI 改动

核心区别：输入不是 CTML，而是对话。

```
GhostChatState (替代 MOSSRuntimeREPLState):
  _on_text_input(text):
    1. session.add_input_signal(text)     # 输入 → Signal → Mindflow
    2. mindflow.loop() → Attention        # Ghost 被唤醒
    3. ghost.articulate(articulator)      # 生成 Logos
    4. moss_exec(logos)                   # 执行 CTML
    5. session.put("stream/logos", ...)  # 流式推送输出
    6. session.on_output() 监听展示       # buffer + 流式渲染
```

`GhostTUI(MossHostTUI[MossRuntime])`：
- `_get_runtime()` → `host.run_with_ghost("atom")`
- GhostRuntime 实现 MossRuntime ABC，TUI 框架完全透明

大部分 TUI 基础设施（ConsoleOutput、render loop、key bindings、state 切换）直接复用。

## 决策结论

1. Ghost 发现由 `MossHost` 负责，不放 `Manifests`
2. 发现技术模式复用 `scan_package("MOSS.ghosts")` + `isinstance(obj, GhostMeta)`
3. `MossHost.ghosts() -> dict[str, GhostMeta]` 新增
4. `MossHost.run_with_ghost(name)` 作为启动入口，返回 GhostRuntime（包裹模式）
5. TUI 框架复用，新增 `GhostChatState` 替代 `MOSSRuntimeREPLState`
6. GhostChatState 输入走 Signal → Mindflow 链路，而非直接 CTML 执行
