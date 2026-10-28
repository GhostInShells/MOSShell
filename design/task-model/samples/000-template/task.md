---
id: "000"
name: <一句话，动词开头>
swarm: <集群名>
workspace: <权限边界：一个 workspace 路径>
parent: null | <父 task id>
opened: D99 HH:MM
closed: D99 HH:MM
state: done | canceled | exception
---

# <id> · <name>

## 1. 目的（intent）

一句话说清"做完之后世界有什么不同"。不接受"完成任务"这种说法。

## 2. 组件清单（BOM）

任务不是 agent 干的，是一组**组件**被装配起来干的。核心单位是组件，不是 agent。

| 类型 | 名字 | 职责 |
|---|---|---|
| loop | `task-loop` | 驱动 op 序列的循环（谁在推进） |
| agent-preset | `<model> / <depth>` | 推理内核的预设（不看名字看能力） |
| skill | `<skill>` | 可复用的动作知识 |
| tool | `<tool>` | 真正碰世界的手（bash / editor / channel） |
| boundary | `<workspace path>` | 权限边界；子任务不跨它 |

## 3. 图（design graph）

**节点 = op**（一次可重入的小循环，栈不跨 op）；**边 = event**（事实，不是消息）。

```
op-1 ──event──▶ op-2 ──event──▶ op-3
   │
   └──event──▶ op-2'   (返工分支)
```

每个 op 标注它用到的组件：`op-1 [tool:bash] [skill:uv-workspace]`

## 4. 边界与授权

- 本任务在哪个 workspace 内签发：`<path>`
- 子任务只能由父签发、且**不跨 workspace**；跨层要新开一层边界。
- 父对子授权的是**边界**，不是"做事的方法"。

## 5. 出口与事实

终态只有三个：`done` / `canceled` / `exception`。每个都要给**证据**（可点、可复核）：

- done → 提交号 / 文件路径 / 命令退出码
- canceled → 谁取消、为什么
- exception → 哪一步、什么错、留下了什么

## 6. 视图对照（这套拓扑怎么看见它）

- L1 规划：属于哪一层迭代（feature / workstream）
- L1 资源：用了哪些组件（BOM 的另一种画法）
- L1 任务：在哪个 workspace 上飞、飞了多久
- L2 设计：第 3 节的图，点节点看代码
- L2 运行：第 5 节的事实链
