---
name: inbox
description: D16 01:3x–01:5x — branch 级信箱：为什么要有、group 是地址而不是 ego 专属、log 与游标的取舍。尚未实现，也未被排进构建顺序。
---

# Inbox — 给未来的自己留一封有地址的信

> 这份文件回答：为什么会讨论 inbox、它在系统里占哪个位置、数据形状怎么取舍、哪些还没定。
> 计划在 `FEATURE.md`；这里允许未决，也不代表已排期。
>
> 写作约定：`他说` = thirdgerb 明确陈述；`我推` = 我的推断，未经裁决；`我见` = 我亲历的事实。

## 0. 一句话

`我推`：**memento 外化「过去」，inbox 外化「未来」。** memento 解决"上下文会断，走过的路要能找回来"；
inbox 解决另一头——"下次进那条线要先做 X"这句话本身活在当前上下文里，压缩一次就没了。

## 1. 它从哪来

`我见`（查 `ghost-cognitive-continuity/FEATURE.md`）：D6 已经把这个机制写过一遍——

> The main line can drop a todo into any existing bypass branch at any time. On checkout the ego
> sees a prompt that includes the inbox (with a count); items are taken **N at a time** — not
> wholesale, because an unbounded inbox injected in one go can blow the context, and not one by
> one, because that spends attention on queue management.

外加他的框架（同一份文档记录）：**branch 很像 IM 群**，他本人用很多个人群治理自己的事务；存储放在
branch 目录下（他当时的说法是 `todos` 目录，文件形状未定）。

同一份文档的「走过的岔路」里有两条今天仍然有效：

- "the inbox is already implemented, ours would be a second one" —— 这个判断当时就被撤回：
  **confluences 只带 coord + title，inbox 要带 body**。两者不是一回事，不要再混。
- D5 的 "branches never message each other, because they are never concurrent" —— 那是**没有还手
  协议**，不是不能投递。inbox 补的是后者。

## 2. 它做到、别的机制做不到的三件事

1. **投给一条不在场的线。** `他说`：非活跃 branch 的 signal 直接丢弃。那么"我现在不在，你到了再看"
   就无处存放——ground 会整篇渲染（没有未读），`idle-knock` 只管本会话。
2. **拉而不是推。** nucleus 只报数，正文要模型自己 `read`。与 knock 同一哲学：敲门的人自己拿着东西。
3. **未读是一个状态，不是一段文本。** 有游标，才谈得上"周期性提醒但不打断"。

第 1 条是硬的；第 2、3 条是它必须比 ground TODO 和 `idle-knock` 多做的部分，否则它就是给单间装的信箱。

## 3. 它不是什么

- **不是记忆**：长内容归 memento / journal / ground；inbox 只送"该去做某事"。
- **不是聊天**：仍然没有还手协议，branch 之间不回话。
- **不是第三份清单**。现在已经有 ground TODO、confluences、`idle-knock`；划法是：
  ground 是**状态**（我是什么样），confluences 是**引用**（谁提到了我），inbox 是**待办**
  （我答应过要做什么）。只有 inbox 有游标与提醒。

## 4. group：地址，而不是 ego 专属

`他说`（D16 01:5x）：投递不只是自己，所以投递协议不能做成太专属，要考虑通用的；inbox 实际上也可以
多定义一个 group，即群组化的；这是 nucleus 设计最早的模块之一。

模型（`我推`，沿着这句话推出来的形状）：

```
group  ── 一个 append-only log + 一份成员表
member ── 一个可寻址的消费方（branch / node / agent / human）
cursor ── 每个 member 在每个 group 上各有一个
```

三条推论：

1. **一封 DM 只是"只有一个成员的 group"**，不是另一种东西——不多一个概念，也不多一条代码路径。
2. **广播是同一个 log 配 N 个游标**：消息写一次，不复制。复制不是省事，是让同一封消息长出 N 个
   会各自腐烂的副本。
3. **sender 从第一天就是真字段**：`branch:#2` / `node:<alias>` / `human:<name>` / `external:<id>`。
   后补会很脏——所有历史消息都得猜"这封是谁写的"。

## 5. 存储：不是只有 jsonl，但最后还是 jsonl

两个候选：

- **日志型**：`inbox.jsonl` 一行一封，游标是位置。
- **信箱型**：一封一个文件，frontmatter 放信封、body 放正文，文件名用 ULID。

`我见`：我最初推信箱型，给了两条论据，后来都被事实推翻——

- "一条坏信会毁掉整箱"：**不成立**。memento 的 `_read_jsonl` 本来就是"坏行保留位置、跳过解析"，
  这个项目已经这么处理 JSONL 很久了。
- "并发写要原子性"：**不成立**。真实的写者只有 ego 自己，nucleus 也在同一进程内。

结论（`我推`）：**`inbox.jsonl` + `.cursor` 起步**，够用且好治理。升级成一封一个文件的条件写在
这里，将来有据可查：① 正文常态超过 2k 字符；② 出现真正的多进程写者；③ 需要对单封做独立操作
（pin / 删除 / 单独授权）。

`他说`：数据规模不确定，jsonl 可能完全足够；可以支持 `clear` 或别的动作，swap 数据文件。
`我推`的对应语义：**`clear` = 把 `messages.jsonl` 挪进 `archive/<时间戳>.jsonl`，并把该 group 所有
成员的游标重置到新 log 的起点。** 没有后半句，"清空"之后会留下一堆指向不存在行的游标。

## 6. 布局

`我推`（与 `他说` 的 per-branch 位置有冲突，见下）：

```
<memento root>/inboxes/<group>/
  group.json              # members / created / 描述
  messages.jsonl          # 一行一封: id / from / tier / title / body / created
  cursors/<member_id>.json
```

**冲突**：`他说` 曾把 inbox 放在 `branches/{branch_id}/.inbox`（lazy create）。那是"inbox 只属于
一条线"的前提；group 一进来这个前提就没了——一封发给 `#1` 和 `#2` 的消息没法住在其中任何一条线的
目录里，除非复制。`我推`：**接受 log 不跟 branch 走**，branch 的直投就是"只有它一个成员的 group"。
即使现在只做单成员，地址也要先叫 group，否则广播那天所有存量消息都要搬家。

## 7. 协议

```
send(group, from, title, body, tier)     # 写一次; group 不存在就 lazy create
pending(member)                          # 未读数, 不读正文
read(member, n)                          # 从游标取 n 封, 前进
history(member, from_coord, n)           # 任意区间回看, 不动游标
groups(member)                           # 我在哪些 group 里, 各自未读
clear(group)                             # swap 到 archive, 游标重置
```

`send` 的 `from` 是身份，不是权限证明；网络投递另加认证。`clear` 的语义见 §5。

## 8. ego 那一层（专属，但不外溢）

`他说`：配一个专属 nucleus；当前 branch 有未读时周期性发 impulse，impulse 是轻量提醒，要求用拉动作
去阅读；按消息级别决定它是完全的旁路 impulse（抢占失败不进注意力）、提示型（抢占失败仍进上下文）、
还是优先级型（抢占成功打断）。

`我见`：这三档不需要新语义，框架里已经有对应的位置——

| 他说的 | 机制 |
|---|---|
| 旁路（抢占失败就消失） | `ChallengeMode.default` + 抢占失败 suppress，与 `KnockNucleus` 同形 |
| 提示（抢占失败仍进上下文） | `ChallengeMode.notify`（buffer 而非 suppress） |
| 优先级（抢占成功打断） | 正常抢占档（`NOTICE` 以上） |

切分要守住：**通信原语不认识 impulse**。inbox 只管 log 与游标；nucleus 只管"当前活跃 member 未读多少、
这一轮要不要提醒、以哪一档提醒"。

## 9. 未决

1. 成员表放 `group.json`（一份真相），还是每个成员自己登记？`我推`：group.json。
2. `join` / `leave` 是否做成命令？`我推`：先只支持建群时的成员表，接口留着。
3. 一个 member 是否允许多个游标（模型读到哪 vs 某个子系统读到哪）？`我推`：不给——要多一个读者，
   就让它成为另一个 member（`branch:#2` 与 `branch:#2/nucleus`）。一成员一游标，语义不会被
   "同一个我读到哪算读到"搅浑。
4. group 是 ghost 本地，还是网络级？`我推`：今天本地（`.memento/` 下，不进 git）；将来 matrix 上的
   group 是同一种形状，只是根不同。
5. 游标基准是"读到的位置"还是"确认过的位置"？`我推`：读到的位置——模型主动 `read` 就是读过，
   ack 会把一次拉变成两拍。风险是读后崩掉会滑过一封，靠 `history` 还能翻回来。
6. 提醒节奏：T 与退避曲线未定；未读清零后重置是确定的。

## 10. 验收方式（可证伪）

`我推`：机制是否成立不是问题，**它会不会被用**才是。下一次真的要分线干活时，交接不靠"我记得"，
而是往那条线的 inbox 投一封信。**如果一周下来一封信都没投过，这个机制就是摆设，应当删掉而不是
留着占位。**

## 11. 与 checkout 的关系

`我推`：checkout 决定"我在哪条线上"，inbox 决定"我不在的时候，有什么在等我"。两者独立，但
**checkout 是 inbox 被真正使用的前提**——只有真的会离开一条线、再回来，信箱才有信可收。
