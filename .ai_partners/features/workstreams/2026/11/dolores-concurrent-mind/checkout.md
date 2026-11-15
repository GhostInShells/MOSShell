---
name: checkout
description: D16 02:0x–02:3x — 换线机制：turn 边界换身份、把新线的记忆面作为下一轮的输入，不迁移上下文。待实现。
---

# Checkout — 换的是身份，不是上下文

> 这份文件回答：checkout 到底是什么动作、它在哪个时刻生效、两侧各留下什么、失败怎么处置。
> 计划在 `FEATURE.md`；这里记录定稿与理由。
>
> 写作约定：`他说` = thirdgerb 明确陈述；`我推` = 我的推断，未经裁决；`我见` = 我亲历的事实。

## 0. 一句话

`我推`：**checkout 不是把上下文搬过去，是换一个身份，再把新身份的 memento 面插进当前会话。**
session 不变、模型不变，变的是"我现在属于哪条线"。

## 1. 和 D13 那版的关系

`我见`：`FEATURE.md` D13 行写着 "checkout is abandoned — moving context costs too much"，`clone.md` §23
同样。那一版要迁移上下文，代价是 O(上下文)。

`他说`（D16 02:0x）：**现在这版是另一种机制**——动作发生后，session 不动，只换分支身份并渲染新面。

`我推`：两版的差别只有一句——**搬内容 vs 换视角**。这版每次切换的代价是 O(一次渲染)，所以它不会被
自己的成本压死；但也因此必须守住一条纪律：**任何时候都不许把旧线的内容搬进新线**，否则就退回 D13 了。

## 2. 生效边界：turn

`他说`：肯定是下一轮才生效，用 turn 做边界；checkout 数据治理完之后发 signal 切换。

`我见`：`commit` 本来就是"排队、turn/end 落地"（`commit_anchor`），checkout 用同一个边界，等于白拿
一套现成的时序。

`我推`：**命令的返回值不能承载新线的任何内容**。命令结果落在"还没切换的那一轮"，把 B 的面塞进去
等于让边界两边不一致。所以命令只回执"已排队"。

## 3. turn/end 上的动作顺序

`他说`：memento 渲染面先拿到，然后添加插入 commit，无错误然后再渲染更新后的 memento。

定稿的执行链（`我推` 补齐了出发侧的落点与失败处置）：

1. **渲染 B 的 memento**（冷读一次）——先验证它出得来；出不来，整笔不切。
2. **写 B 的 arrive 锚**（复用 depart 的 session ref，见 §4）。
3. **重渲染 B** ——此时到达标记已经在面里。
4. **写 A 的 depart 锚** ——这一步落地即切换完成。
5. **换运行时身份**。
6. **发 signal**（§6），body 是第 3 步的渲染结果。

失败处置：1–3 失败 → 停在 A，报错，什么都不变。3 之后、4 之前失败 → B 上留一个假的到达标记
（append-only 删不掉），但我仍停在 A；这是两害相权，反向（先写 depart）会留下"A 走了、B 没到"的
悬空，更糟。**depart 是这次切换的提交点。**

## 4. 两侧的记录：depart / arrive

`他说`：必然生成明确的 note——checkout 动作不是 commit，必须有 note；标题形如 `checkout(branch): title`。
`他说`：两个动词用更专业的，`depart` / `arrive` 可以（再思亦无更优）。

`我推` 的形状：

- 出发：`depart(#2 idea): <title>`，写在 A 上，`to` 指向目的线。
- 到达：`arrive(#1 main @1-735): <title>`，写在 B 上，`from` 指向来源线与来源坐标。
- **两个锚共用同一个 title**：离开的理由就是到达的理由，模型只写一次；出发锚的 body 是 `note`，
  到达锚的 body 指向来源坐标。
- **op 默认空字符串**，空 = 无语义约束（今天的普通 commit）。`note` 不是 op 名。
- **op 负责组合默认标题**；结构化参数（`to` / `from` / `coord`）进 commit 本身，**不从标题字符串里解析**。
- `fork` / `archive` 本身不留 commit，不需要 op。

### op 的定义只有一处

`他说`（D16 02:4x）：op 约定类型之后，文档可以在 command 上带；但**不要抄注释**——迭代里会自动腐烂的
抄写最讨厌，最好用 doc fn 生产，或者换别的方式。

`我推` 的落法：一张 `OpSpec` 注册表（name / title 模板 / 需要哪些参数 / 一句话语义 / body 规则），
三处产物全部由它生成：

1. commit 时的标题与校验；
2. channel instruction 里的 op 表；
3. `checkout` / `commit` 命令 docstring 里那段 op 说明——**用 spec 的 `doc()` 拼**，不手写第二份。

### 命令签名

`他说`（D16 02:4x）：checkout 里没有 title 不行；既然定义了 op，就应该是 `checkout(name, title, note)`，
否则 note 又是纯粹独立规则。**note 是长文本，必须一直用 `text__` 定义参数**（`commit` 同理）——
否则在 `""` 里手写转义太痛苦。`text__` 现在还要 CDATA 包一层，这个麻烦一直没时间解决，先用 `text__`。

`我推`：

```python
async def checkout(name: str, title: str = "", text__: str = "") -> str
```

- `name` / `title` 走属性；`note` 走标签体（`text__`）。
- `title` 必填：它就是两侧 note 的标题，缺了就不是"明确的 note"。
- `note` 可空：空则两侧只有标题。

### 零宽锚与 turn 空间

`我见`：commit 区间是 `(start_turn, end_turn]`，下界取当前 branch tip 的 `end_turn`
（`_ego.py:_session_start_turn`）。所以换线之后，B 的下一笔会拿 **B 上一次居住时的 tip** 当下界——
中间那些 turn 全是在 A 上过的。

`他说`（D16 02:5x，定案）：**arrive 的 session ref 和 checkout 触发的 depart 完全一模一样**——
同一个 `session_id`、同一个 `(prev, T]`。不是零宽，也不加任何开关。

`我推` 的后果：

- 同一段区间在两条线上各记一笔：A 记"我从这里走"，B 记"我到这里来"，**认领的是同一段 turn**，
  所以 B 的下一笔自然从 `(T, …]` 起，不重不漏。
- `EgoMementoManager.commit()` 一个字不用改——它本来就收显式的 `start_turn` / `end_turn`，
  arrive 只是把 depart 那对数字原样传进去。没有"空区间"这个特例，也就没有开关。
- `latest_ref()` 在 B 上拿到的是 `end_turn = T` 的 ref，B 的下一笔不会退回 `0` 当下界。
- 实现上两份 ref 的 `session_id` / `start_turn` / `end_turn` 逐字相同；`created` 是各自落盘的
  时刻，因此不同。它是记录的时刻，不是区间的一部分 —— 若真要连它一起复制，等看到真机的两条
  记录再定。

我一度主张零宽 `(T, T]` 再开 `allow_empty`——那是把"一个时刻"的语义压成一个布尔，属于熵。
同一份 ref 显然是更小的形状：没有新概念，没有新分支，没有新开关。

## 5. 身份

`他说`：`.dolores.yml` 是运行时配置项，不可能在这里改动；branch meta 不要加状态字段——它持的是治理面
状态；运行中状态是它的子集，切换行为都在 ego 上，等做 checkout 时维护进去。

`我推`：

- `config.branch_name` = **启动醒在哪**，人的配置，ego 只读。
- 运行时身份 = **我现在在哪**，ego 自己持有；checkout 改的是它。
- 今天两者是同一个字段，必须拆开，否则 checkout 要么写配置（被否掉），要么重启后状态错乱。
- 只能切**活跃**分支，按 name 切；归档分支、不存在的分支一律拒绝。

## 6. signal

`他说`：signal 是内核机制，走 ego 自带的那条。

`我见`：`DoloresEgoNucleus` + `new_dolores_ego_signal` —— 带 body 的信号是 **delivery**，
BACKGROUND + `ChallengeMode.notify`，不抢注意力，抢占失败也 buffer，永不丢。

`我推`：用 `role='mind'`，落 `Moment.percepts['mind']`，不会被误读成人的话。

代价：一次 10–20k 的表面作为一条 signal body，且送出去那一刻就已旧。将来若要省，可退成"指针 + 摘要"
（branch / coord / 未读 / 最后一条 op），让模型自己 `view` 拉全量——那是省钱版，不是这版。

## 7. 约束与错误

`他说`：同一轮里连切两次，命令里 raise 严重的 error，并把上一次的操作清空掉。

`我推` 的完整拒绝表：

| 情形 | 处置 |
|---|---|
| 同一轮第二次 checkout | 清空已排队的切换，抛严重错误 |
| 目标是当前分支 | 拒绝（no-op 不等于切换） |
| 目标不存在 / 已归档 | 拒绝 |

## 8. 未决

1. 反复进出同一条线时，多份 `<memento>` 会堆在上下文里；要不要在每个面打 `epoch` 之类的标记。
2. 重启醒在哪 —— 下一个命题，这里不展开（按现在的逻辑会总是回 `main`）。
3. 到达面的 token 成本是否要按线设预算。

## 9. 验收（可测的行为）

- 一次 checkout 之后，旁路队列为空：两个系统锚都自带 note，sidecar 应当整条让开。
- B 的 tip 是一笔 `arrive`，它的 session ref 与 A 的 `depart` 逐字相同（同一 `session_id`、同一区间）；来源线与坐标写在 op 的 target 里；A 的 tip 是 `depart`。
- B 的下一笔 commit 的下界 = 切换时的 turn（`(T, …]`），不吞 A 的 turn。
- 切到归档分支 / 不存在的分支 / 当前分支：拒绝且无副作用。
- 同轮第二次 checkout：第一次的排队被清空，且命令报错。
- `checkout` 的 `note` 以 `text__` 声明：长文本可以从标签体进来，不必在属性里转义。
- op 的说明只有一处来源：instruction 里的 op 表与命令 docstring 都由 `OpSpec` 生成，改规则只改注册表。
