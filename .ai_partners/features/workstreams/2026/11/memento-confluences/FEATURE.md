---
created: 2026-11-08
depends: []
description: Branch 之间的引用：出向写在 Note.recipients，入向在 commit 时吸收进 CommitRef.confluences；为折叠后的 history 服务。它是 memento-agent 的前置。
milestone: dsh 0.2.0-rc.2
priority: P1
status: in-progress
status_note: 'D09: Address/Confluence + inbox absorb implemented; 42 memento tests
  pass'
title: Memento Confluences
updated: '2026-11-09'
---

# Memento Confluences

> Use `moss features set-status memento-confluences <status> -m "note"` to update state.

状态: **in-progress** — `Cite` / `Note.outbound` 已落地（D09）；汇流的交换机制按 D09 决定暂缓。

> 归因沿用项目约定：`他` = thirdgerb 说的；`我推` = moss 的推断，未经裁决；`待定` = 已知未决。
> 来源：D08 17:2x–18:1x 的一轮讨论（`dsh-node` 的 agent 建模引出）。本轮之前的两条岔路
> （把汇流放进 `CommitRef.metadata`、把汇流当成 `commits.jsonl` 里的一行）都已被否掉，理由见
> 「走过的岔路」。

## D09 修订（本节覆盖下文 D08 的相应判断）

D09 深夜按「先定验收效果，再倒退实现」重做了一遍。**结论：两半都做，而且是为 history 服务的。**

### 验收效果（这一节是设计的锚）

一条发过消息的 commit，重新在自己的 branch 里看：

```
1-12  propose(dev): 交给你合并
      → child-a
```

被压进 history 的那条 commit，看的是收侧：

```
history
  1-9   report: 第一版跑通
        <from name="child-a" at="5-3">落位了，锚点取到了</from>
        <child name="child-b" at="2"/>
```

**history 是压缩之后仍要自给自足的一层。** 预算从我自己的正文里省，不能从"别人对我说过的话"里省 ——
否则重建后的第一件事就是去读一堆 commit，这个结构就白存在了。渲染形态（标签名、inline 与否）
归逻辑层，随时可改；契约只保证字段在。

### 数据结构（已实现）

```python
class Address(BaseModel):        # 谁
    branch_id: str               # 身份：解析从这儿开始
    branch_name: str             # 显示/业务用；会腐烂，不参与解析
    branch_index: int            # 坐标前半段

class Confluence(BaseModel):     # 谁 + 哪一步 + 说了什么
    branch: Address
    commit_seq: int
    title: str                   # 唯一被复制的内容
```

| 位置 | 字段 | 语义 |
|---|---|---|
| `Note.recipients` | `list[Address]` | 出：这条消息发给谁。只有地址，没有内容（内容就是这条 note） |
| `CommitRef.confluences` | `list[Confluence]` | 入：这条 commit 之前收到过什么 |

**为什么两半不对称**：出向指向一条还没写下第一步的 branch，没有 seq 可指；入向指向已经落盘的
一条 commit，seq 就是它的地址。一个类型套两个方向，必然有一个字段恒为 0 —— D09 前半场就是这么错的。

### 机制（已实现）

1. 写 note 时若带 recipients，**首次落盘**即向每个接收方的 `confluences.jsonl` 写一行。
   重写 note 不重投（recipients 不是 last-wins：`None` 沿用、`[]` 显式清空）。
2. 接收方下次 `commit` 时**认领** inbox（`os.replace` → `.confluences.jsonl`，同目录 rename 原子），
   读出来、按来源坐标去重、写进该 commit 的 `confluences`，落盘后删点文件。
3. 两个崩溃窗口都由"重读 + 去重"兜住：认领后没提交 → 下次重读；提交后没删 → 下次重读并去重。
4. 投递失败抛错（对端不可解析）；note 已经 durable，重试免费，因为接收侧按坐标去重。
   **轨迹里没有投递回执** —— 只有声明。

### 子 branch：同一条路，反向的一侧（D09 追加）

早期 v3 契约（`143633b5^:src/ghoshell_moss/memento/abc.py`）里有 `CheckoutRecord` +
`checkouts.jsonl`，还有 `BranchMeta.fork_ref`。lean 版把父侧那半边删掉了，只剩子侧的指针。

本轮补回来，**不加新词**：

- memento 只有一种操作：**从一个父创建一条 branch**（`fork`）。是"委派"还是"同线分叉"是业务，
  memento 不认识 —— 所以没有 kind，没有 spawn/child 的区分。
- `CommitRef.children: list[Child]`；`Child{branch: Address, created}`。
- 缓冲文件 `children.jsonl`：`fork` 时由**父自己**写一行，父的下一次 commit 回收。
- 子侧沿用 `BranchMeta.fork_from`（已存在），不需要新字段。
- 不用 `checkout` 这个词 —— 它该留给"memento 当前主分支"这个概念。

**和 confluences 的镜像关系：**

| | `confluences.jsonl` | `children.jsonl` |
|---|---|---|
| 谁写 | 别的 branch 投递 | 父自己 |
| 为什么需要缓冲 | 跨 branch，我不知道 | 动作发生在两次 commit 之间，压缩时才进轨迹 |
| 边 | 入 | 出 |
| 回收 | 我的下一次 commit | 我的下一次 commit |

同一套 claim（`os.replace` 原子认领）+ 去重 + 崩溃窗口处理。

### 锁口径（订正）

- 契约只承诺"一个写者"，不提 file lock。
- 只有 read-modify-write 需要围栏：`commit`（seq 推导）/ `fork`。`note` 与 `confluence` 是纯 append，
  不需要 —— `anote` 因此不再取写权限。

### 命名

- 出侧 `recipients`（零解释成本）；入侧 `confluences`（项目原生词，解释成本在渲染标签，不在数据键）。
- 不用 `from` / `to`：Python 关键字。

### 验证

- `tests/ghoshell_moss/memento` 42 passed。
- 端到端：投递 → inbox → 吸收 → 重写不重投 → 多接收方 → 未知目标抛错 → 两个崩溃窗口。

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