---
id: "002"
name: 在 #121 上补今天的新证据
swarm: moss
workspace: MOSShell
parent: null
opened: D28 18:52
closed: D28 18:53
state: done
---

# 002 · 在 #121 上补证据

## 1. 目的

用户让"记一条增强 issue（moment 与 shell refresh metas 同步）"。查完发现 #121 已经收着这个题目，
所以正确动作不是新开票，而是**在旧票上补新证据**。

## 2. 组件清单

| 类型 | 名字 | 职责 |
|---|---|---|
| loop | dev-loop | 查重 → 读票 → 写 → 发 |
| agent-preset | `deepseek-flash / low` | — |
| skill | issue 约定（`moss features issues`） | 知道"模型不能自己开票，要有当轮授权" |
| tool | `terminal`（gh CLI） | 列 label / 列票 / 读票 / 发评论 |
| tool | `file_editor` | 正文先落 `.playground/`（gitignored） |
| boundary | `MOSShell/` | 只读仓库 + 写 .playground |

## 3. 图

```
op-1 gh label list            ─ event ─▶ op-2 gh issue list
op-2 发现 #121 已存在          ─ event ─▶ op-3 gh issue view 121（读全文）
op-3 确认同一题目               ─ event ─▶ op-4 写评论正文（55 行）
op-4 export 到 .playground/…   ─ event ─▶ op-5 gh issue comment --body-file
                                              └─ 成功：issuecomment-6467100385
```

失败分支（第一次尝试）：把正文塞进 `--body "$(cat <<EOF …)"`，在 `/bin/sh` 下直接崩
（`bad substitution`）—— 记下来，因为它说明"我默认了 bash 语义"。

## 4. 边界与授权

- 授权是**per ticket**：用户当轮说了"记一个增强 issue"，我才动 GitHub。
- 没有开子任务，也没有跨仓库。

## 5. 出口与事实

`done`：<https://github.com/GhostInShells/MOSShell/issues/121#issuecomment-6467100385>

## 6. 视图对照

- L1 规划：`issue-governance` 这条 workstream 的一个实例。
- L2 运行：一条单向链，五个 op，没有回头路 —— 正好是"顺利任务"的形态。
