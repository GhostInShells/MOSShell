---
title: Memento Confluences
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-11-08
updated: 2026-11-08
depends: []
milestone: dsh 0.2.0-rc.2
description: >-
  Branch 之间的汇流（通知 / MR / PR）：出向地址记在 note 上、入向记录在提交时吸收进 commit；
  运行时的投递与轨迹的记录是两条独立通道。它是 memento-agent 的前置。
---

# Memento Confluences

> Use `moss features set-status memento-confluences <status> -m "note"` to update state.

状态: **draft** — 结构已收敛，实现为零。

> 归因沿用项目约定：`他` = thirdgerb 说的；`我推` = moss 的推断，未经裁决；`待定` = 已知未决。
> 来源：D08 17:2x–18:1x 的一轮讨论（`dsh-node` 的 agent 建模引出）。本轮之前的两条岔路
> （把汇流放进 `CommitRef.metadata`、把汇流当成 `commits.jsonl` 里的一行）都已被否掉，理由见
> 「走过的岔路」。

## Motivation

`MementoAgent` 的一次委派 = 一条 memento branch（见 `dsh-node`）。委派之间需要交换数据：

- 子通知父（"我落位了"）；
- 子提请父（MR：把改动交给父）；
- 父取子的东西（PR：发起方反转）。

但"交换"在这套系统里是**两件不同的事**，混在一起就会两头都不成立：

| 通道 | 触发 | 到达 | 内容 |
|---|---|---|---|
| 运行时 | 立刻 | session 的下一轮 | 全文 + 上下文 |
| 轨迹 | note 落盘时 | commit view / 重建时 | 只有 title + 地址 |

**为什么必须两条**（他）：`AgentSession` 本身不会反复刷新 branch commit；不接受消息通知的话，
它可能完全感知不到有这件事发生。反过来，只发消息不落轨迹，重建时就永远缺一块。

所以这个 workstream 的目标只有一个：**让跨 branch 的交换在轨迹里可重建，同时不把运行时绑死在轨迹上。**

## 形状

```
              一次交换发生
             ┌──────┴───────┐
        运行时通道        轨迹通道
     signal / 广播 message   note 落盘时才生产
             │                │
       进 session          进 commit view
       全文 + 上下文        只有 title + 地址
```

## Key Decisions

**KD1 — 两个通道，不是一个**（他）。
运行时投递与轨迹记录各自独立：前者必须立刻，后者跟着 note 走。两者可以晚、可以重放、可以失败，
互不代偿。落点：轨迹侧是文件 append，运行时侧是消息派发。

**KD2 — 汇流由 note 生产，不由 commit 生产**（他）。
更准确地说：**内容就是 note 的 message**。commit 只是那条消息的锚。
推论（我推）：**没有 note 的 commit 本来就没有话要送** —— 所以 `cites` 放在 note 上不是脆弱性，
是自洽。memento 的 note 教义是"可以丢、可以重建"，与之相符。

**KD3 — 出向地址记在 `Note.cites`，不在 `CommitRef` 上**（他自我纠正）。
地址是从内容里读出来的，而 commit 是裸锚点，不该承载"内容派生"的东西。
`CommitRef` 保持它原本的教义：一个锚，一条边都不带。

**KD4 — `cites` 写一次，`message` 可以重写**（我推）。
note 的其他字段 last-wins（摘要器改写措辞是正常行为），但 `cites` 一旦落盘就不再变 ——
否则一次重述就可能把投递目标改掉。判定"首次"的口径待定（见「未决」）。

**KD5 — 入向记录在提交时吸收进 `CommitRef.confluences`**（他）。
branch 目录里有一份 `confluences.jsonl`（收到的暂存）；每次 commit 时先认领、再读、写进当前
commit ref 的 `confluences` 字段，于是**commit 落盘瞬间已经把它收到的全部汇流记在里面了**。

**KD6 — 只复制 title，正文二次关联**（他）。
`Confluence` 只带地址 + 一句话。正文、metadata、node 空间都靠 coord 回源读 —— 这是**显式动作**，
不是渲染的一部分。于是"渲染一个 branch 的 view 只读它自己的文件"成立：
N 条 branch 各 M 条汇流，渲染一次也不会把整个 memento 拉进内存。

**KD7 — 存储形状是扁平列表**（我推）。
`list[Cite]` / `list[Confluence]`；按 branch 去重在写入时做。内存里要不要建 `dict[str, ...]`
索引是消费方的事，不进磁盘格式（JSONL 里存 dict 会丢顺序、难演进）。

**KD8 — 原子性与去重**（我）。
吸收：`os.replace` 认领（同目录 rename 原子）→ 读点文件 → 按 id 去重（对本条与上一条 commit）
→ append commit 行 + fsync → 删点文件。两个崩溃窗口都由"重读 + 去重"兜住。
投递：单行 O_APPEND + fsync；**单行必须保持小**（远小于 PIPE_BUF=4096），否则单行原子性不成立。

**KD9 — memento 不解析正文**（我推）。
`@` 是消费方协议：载体或 hook 脚本从 note 里读出地址，把 `cites` 作为数据交给 memento。
memento 不必知道 `@` 这个字符存在。

**KD10 — 没有 reply 这个 kind**（他）。
回复就是"再送一次"，但它需要知道回投的地址 —— 那属于 group 协议 / `@` 的范围，本轮不做。
所以本轮只有单向的 `notify` / `propose` / `deliver`。

**KD11 — 事件出口 `on_committed`**（他）。
一条 commit 的 note 落盘后触发一次；memento 只发事件、不解释它。控制器（hive / agent runtime）
从这里把 commit 卸载进队列，再找出对应的 session 发消息。这是两条通道在 memento 侧的唯一交点。

**KD12 — note 重写不重复投递**（我推）。
`note()` 是 last-wins，摘要器会重写。投递必须在**首次** note 时发生一次；接收侧同时按
`(branch_ord, commit_ord, kind)` 去重作为第二道保险。

## 数据结构

```python
class ConfluenceKind(str, Enum):
    notify = "notify"      # 只是知会
    propose = "propose"    # MR：我把东西交给你
    deliver = "deliver"    # PR：我来取你的东西


class Cite(BaseModel):
    """出向，写在 Note 上：这条 note 的 message 要投递给谁。"""
    branch: str            # 接收方 branch 名
    kind: ConfluenceKind


class Confluence(BaseModel):
    """入向，吸收进 CommitRef.confluences：谁给我送过什么（只带一句话）。"""
    id: str
    branch_name: str       # 来源 branch 名
    branch_ord: int        # 来源 branch 的 index
    commit_ord: int        # 来源 commit 的 seq
    kind: ConfluenceKind
    title: str             # 唯一被复制的内容
    opened: AwareDatetime

    @property
    def coord(self) -> str:
        return f"{self.branch_ord}-{self.commit_ord}"
```

契约增量：

| 位置 | 变化 |
|---|---|
| `Note` | `+ cites: list[Cite] = []`（写一次） |
| `CommitRef` | `+ confluences: list[Confluence] = []`（吸收而来） |
| `Branch` | `+ confluences()/aconfluences()` 读暂存、`+ confluence()/aconfluence()` 写暂存 |
| `Branch` | `+ on_committed(cb)` 事件出口 |
| `note()` | 首次写入时对每个 cite 做一次投递 |

全部向前兼容：旧数据解析为空列表。

## 走过的岔路

**① 把汇流放进 `CommitRef.metadata`。** 被否（他）：跨 branch 交换是**原始建模**的数据结构，
不是消费者自定义的不透明扩展。放进 metadata 等于放弃建模。

**② 把汇流当成 `commits.jsonl` 里的一行。** 被否。三处会当场撒谎（证据在 `_fs_memento.py`）：
`commit()` 用 `len(commits)+1` 算 seq，外来行会占号；`fork()` 用 `commits[-1]` 当分叉锚；
`BranchView.commit_id` 用"最近一条"。也就是说末端的语义会被污染。真要这么做得把"末端"变成显式
事实（`BranchRef.tip`）—— 那是另一个改动，本轮不做。

**③ 只记地址不带内容，渲染时交叉读。** 被否（他）：汇流一多，渲染一次就等于把整个 memento
拉进内存。所以入向行必须自带 title。

**④ staging 一直挂着不吸收。** 被否：提交那一刻吸收，才让"这条 commit 收到过什么"成为
锚点上的事实，而不是一个要回溯计算的投影。

## 未决

1. **`cites` 的磁盘形状**：`list[Cite]` + 写入去重，还是 `dict[branch, Cite]`？（他未定）
2. **"首次写入"的判定口径**：note 行里 `cites` 为空就是"还没写过"，还是有别的标记？
3. **吸收时机**：commit 之后、note 之前到达的汇流，会落到**下一条** commit —— 可接受？
4. **失败口径**：目标 branch 目录被外部删除、fsync 失败 —— 报错即止还是重试后留痕？
   （我推：都不许静默。）

## Not in scope

- `@` 寻址与 `op(target)` 的关系 —— `@` 需要先有可寻址的 group 协议，本轮不做（他）。
- `reply` 协议（见 KD10）。
- 跨 branch 的"状态"（接受 / 拒绝）—— 用一次新的回复表达，不给 Confluence 加 state。
- `BranchRef.tip`（显式末端）—— 与本工作无关，但它是岔路 ② 的根，将来可能单独做。

## Materials

- `src/ghoshell_moss/memento/abcd.py` — 契约（Note 的 last-wins、CommitRef 的 metadata 教义）
- `src/ghoshell_moss/memento/_fs_memento.py` — 实现（seq 计算、O_APPEND、FileLocker、`_write_text` 的 tmp+rename）
- `src/ghoshell_moss/agents/blueprint/memento_agent.py` — 第一个消费方
- `design/task-model/` — D28 的 BOM + op 图（task 是视图，不是类型）
- 相关 workstream：`dsh-node`（memento-agent 的第一版通过它落地）

## 关联

- **前置关系**：本工作是 `dsh-node` 里 memento-agent 的前置。memento-agent 不单独开 workstream，
  第一版通过 `dsh-node` 落地（他）。
- 与 `memento-hooks`（2026/10, completed）的区别：那个的机制是"往目录里丢一个可执行文件，
  它就运行"，是**文件触发的副作用**；汇流需要的是**结构化数据 + 编程事件**，两码事。
