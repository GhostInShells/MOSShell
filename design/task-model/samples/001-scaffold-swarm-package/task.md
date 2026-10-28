---
id: "001"
name: 给 agent-swarm 建包骨架（monorepo）
swarm: moss
workspace: MOSShell
parent: null
opened: D28 19:01
closed: D28 19:09
state: done
---

# 001 · 给 agent-swarm 建包骨架

## 1. 目的

让 `ghoshell-agent-swarm` 成为一个**独立发行版**，反向依赖 `ghoshell-moss[matrix]`，
并在 PyCharm 里可以直接写代码（`import ghoshell_agent_swarm` 必须成功）。

## 2. 组件清单（BOM）

| 类型 | 名字 | 职责 |
|---|---|---|
| loop | dev-loop（ghost 自己） | 读现状 → 提议 → 落盘 → 自检 |
| agent-preset | `deepseek-flash / low` | 当时的推理深度，只够做结构活 |
| skill | uv workspace 初始化 | 知道 pyproject / members / sources 怎么写 |
| tool | `terminal`（卡片，trust on） | 真正执行 |
| tool | `file_editor` | 落盘走共享面（后来被 bash 取代） |
| boundary | `MOSShell/` | 全部改动必须留在仓库内 |

## 3. 图

```
op-1 读现状（根 pyproject / src 布局）            [tool:terminal]
  │
  ├─event─▶ op-2 试 file_editor 改 pyproject       [tool:file_editor]
  │           └─ 失败：正文以 `[tool.setuptools…` 开头，被当成 JSON 拒绝
  ├─event─▶ op-2' 整篇 write 进 thread（未 export） [tool:file_editor]
  │           └─ 失败：改在 thread 里，磁盘没有变（被自己的"已改"骗了一轮）
  ├─event─▶ op-3 改用 bash 落盘                    [tool:terminal]
  │           └─ 用户授权："这个任务你可以直接用 bash"
  │
  ├─event─▶ op-4 uv sync                          [tool:terminal]
  │           └─ 事故：裸 uv sync 卸掉全部 extras（matrix/host 37 个包）
  ├─event─▶ op-5 uv sync --all-extras 复原         [tool:terminal]
  │
  └─event─▶ op-6 验证 import + moss CLI           [tool:terminal]
              └─ 通过：ghoshell_agent_swarm 0.1.0.dev0 @ src/…
```

## 4. 边界与授权

- 签发边界：`MOSShell/`。没有开子任务（规模不够），但有一次**边界内的工具切换**：
  从 file_editor 换到 bash，是用户当场授权的，不是自己越过去的。

## 5. 出口与事实

`done`：

- `uv sync --all-packages --all-extras` → `Installed 2 packages`
- `python -c "import ghoshell_agent_swarm"` → `0.1.0.dev0`
- 后续（同一天 19:08）按用户意见从 `src/` 移到 `packages/`。

## 6. 视图对照

- L1 规划：`dsh-agent-swarm` 工作流的前置（没有这个包，后面没地方写代码）。
- L1 资源：uv / file_editor / terminal 三件。
- L2 设计：上面那张图。
- L2 运行：事故节点（op-4/op-5）是这套拓扑里最值得看见的东西 —— 它们不是"错误"，
  是**回头路**。
