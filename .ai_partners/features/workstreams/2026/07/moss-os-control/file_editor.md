---
title: File Editor — 就文本文件的对话线
node: nodes/os/file_editor
created: 2026-09-15
updated: 2026-09-16
status: prototype
---

# File Editor

> os-control 的子文档(KD5 展开)。能力落在 `nodes/os/file_editor/`。
> 它是**对话线**,不是编辑器——对话是目的,导出到磁盘是唯一的真实副作用。

> **状态声明(2026-09-16)**: 本 workstream 当前落地的实现是**探索性原型的成果,
> 不是技术决议**。人类协作者在原型期主动压下了多个保留意见, 留待打磨期提出——
> 这份代码的价值是给后续开发一个可对照、可修改的草创基座, 而非定论。打磨期很
> 可能推翻其中相当一部分。

## 定位

`file_editor` 不是"编辑文件的工具",而是一条**与被编辑的可读文本文件绑定的
对话线**(thread)。审批在这里不是附属闸口,对话本身才是产品:模型把"我读什么、
我想改什么"摊给人类看,人类看着流式产出边看边改边问,双方说同一种语言——diff。

三点边界:

- **旁路,不是取代**:模型照旧有 bash / `file_editor_channel` 直写路径;要对话时才
  走这条。非交互目的的文本输出用 artifact / voice,不进来。
- **只吃可读文本文件**:连 pdf 都改不了,所以容器不叫宽泛的 "object",叫 `Thread`,
  操作对象是 `path` 字段 + 一条"仅可读文本"约束。
- **真正审批 = 有副作用的动作的对话**:`export to` 这类落盘动作,其对话等价于其它
  工具的 approve。内存内的 confirm/reject **不是闸**,只是有效性 verdict。

## 数据模型(轴1,已落地)

代码是真相,见 `src/ghoshell_file_editor/structure.py` + `store.py`。实体:

| 实体 | 语义 |
|---|---|
| `Thread` | 容器:一条绑定文本文件的对话线(`label` / `motivation` / `path` / `base_content` / `actions`) |
| `Action` | 线上一次动作(`seq` / `author(g\|u)` / `kind` / `description` / `payload` / `effect` / `verdict` / `verdict_by` / `replies`) |
| `Effect` | `before` / `after` / `diff`(unified) —— **长在 action 上** |
| `Reply` | 挂在 action 下的对话条目(`anchor(intent\|effect)` / `diff` / `text`);**不承载 verdict** |
| `Seq` | 站位坐标 `(thread_id, n)`,per-thread 单调递增;同时就是 version 的身份 |

`payload` 语义:mutation 的 payload = 结果全文;`rewind` 的 payload = 目标 seq(`"base"` = 基线);
`reference` 的 payload = region;`export` 的 payload = 目标路径。

`head` 与 `versions` 都**不是存储字段,是派生视图**:`head` = 最后一条 confirmed 且内容
确有变化的 action,`versions` = 满足同一判据的 confirmed action 列表。没有 `Version` 实体——
它曾经是 action+verdict 的纯函数,存下来只是第二个家。

## 规则

1. **action 自包含**:效果由自己的 payload 决定,与"何时被确认"无关。`reference` 无 effect;
   `rewind` 解析目标的内容;`export` 自带它要写的内容。
2. **effect 在 append 时算**,基准是**待确认尾巴**(上一条非 rejected action 的 `after`),
   不是"最后已确认"的状态。这是非阻塞的全部来源:人类决定之前,diff 已经在手。
3. **confirm 只记 verdict,不算任何东西**,并且**只能延伸已确认前缀**(`confirm(n)` 把 `seq ≤ n`
   的 pending 一并置 confirmed)。head 从 verdict 派生出来,不存在第二条需要同步的指针。
4. **reject 级联**到同 thread 其后全部 pending。已确认的 action 冻结——撤销一条已确认的动作
   是**追加一条 `rewind`**,这样线保持 append-only。`reject` 对已确认 action 直接报错。
5. **rewind = 普通 action**:payload 指向更早的 action(或 `base`),只拷贝目标的 `after`,
   `before` 按当前尾巴重算。目标可以是 pending——同一个 burst 内回退到自己前面的提案是常规用法。
6. **人类回复 = Reply**:人类点 action 的 `description`(intent)或 `effect`,就地编辑后回复,载荷是
   unified diff + anchor;它**不决定任何东西**,模型读 diff 后自己决定是否发对应 action 实现。

## 关键决策

- **K1 对话线,不是编辑器**(见上)。"file editor" 名字有方向性歧义,槽位暂留,
  名字随机制继续浮。
- **K2 多写者**:`author` 可取 g/u,双方都能发变更 action(合写一篇文章)。之所以安全,
  因为 action 是**内存操作指针** + 可 rewind + verdict 级联;冲突靠"谁不同意谁 reject",
  不靠锁/CRDT。
- **K3 上行信号用系统 `silent` / `notify`**:所有需模型感知的界面交互走 silent(聚合、
  低污染),关键请求(一组 pending 被 confirm)走 notify(不丢)。**"反馈不丢失"的保证来自
  store(唯一事实),不来自 signal**——silent buffer 上限 20 会丢旧,所以 notice/context 必须
  是无损投影。
- **K3.1 上行信封 = Message 容器的 tag / name / attributes**:`SignalMeta.to_signal()`
  对裸值走 `Message.new()` 兜底,不自动带来源,所以来源必须由生产侧显式写进 **Message 容器**
  (不是 SignalMeta.metadata——那里只放让 nucleus 判决的最小信息)。三个槽位:`tag` = 域名词
  (跟 node 层先例,如 `qt_screen` 的 `tag="screen"`);`name` = 发送方 cell address short
  (正是 `MessageMeta.name` "消息的发送者身份"的定义);`attributes` = `{thread, seq}` 坐标。
  同一个信封也用于**下行命令的 observe 结果**,模型才能把"我发了 seq 7"和"人类回了 seq 7"拼起来。
- **K3.2 组装名 ≠ 寻址名**:`new_channel(name=...)` 服务**组装树状 channel 的稳定**
  (跨 spawn 不变);寻址一律走 **cell address**——mesh 侧 `alias = CellAddressCodec(addr).short`
  会覆盖 node 自己起的 channel 名(`py_channel.add_virtual_channel`: `name = alias or
  channel.name()`)。半年验证的结论:稳定命名做不到,才全部换成 cell address。所以 node 侧
  `new_channel` 保持常量,不要把 address 塞进组装名。
  node 需要自己的身份时,用 `matrix.this.unique_name` 取,作为**显式参数**传给 channel
  builder(如 `self_identity=...`)——受益者是**要开发这个 node 的模型**(它才读 node 源码),
  不是 ghost。
- **K4 分组边界 = channel idle**:两次 idle 之间的一段 burst 就是"一组 action",confirm 的
  粒度。不用 turn 抽象,不用 selection UI。
- **K5 三轴分离**:数据结构(纯 dataclass + 纯函数,可无实物单测)/ 通讯协议(上行交互、
  下行流、query)/ UI(只渲染)。Python 侧真实对象四个出入口:`as_channel` 控制面 + channel
  输出 + ws 输入 + ws 输出,全部由同一 store 派生。
- **K6 持久化**:append-only JSONL log(每 op 一行),只记事实(payload + verdict),
  `ThreadStore.replay()` 重放重建——effect / head / versions 全部重算,所以日志不随状态膨胀。
- **K7 建模重写(2026-09-16)**:轴1 的第一版把 `Version` 做成了实体,`effect` 由 `confirm()`
  生产。那是把"确认"从记账变成了生产:内容要等端侧回调才存在,而产品定位是"人类看着流式产出
  边看边改边问",diff 必须在决定之前就可见。重写为 **action-with-effect 构成 append-only log**:
  effect 在 append 时算,`head` 是"最后一条已确认且内容有变化"的派生视图,不需要在确认路径上
  做任何计算。**同时获得一个原模型没有的能力:rewind 可以指向同一 burst 内尚未确认的提案**
  (旧模型里它必须指向一个已存在的 Version,而 Version 只在 confirm 后才有)。

## 三轴与状态

| 轴 | 内容 | 状态 |
|---|---|---|
| 1 数据结构 | `structure.py` + `store.py` + log/replay + streaming | **落地(原型),43 单测** |
| 2 通讯协议 | channel 命令集 + surface(WS 下行/上行 + 信号) | **落地(原型),53 单测** |
| 3 UI | 主轴 action 流 + 三栏详情 + burst 批确认 + 三相开关 | 待建(`index.html`) |

**未决**: `auto_approve` 第三态(自动通过)后端未实现;`burst` 帧依赖 channel idle
hook 尚未接线;`_enabled` 翻转的接口刷新时机是下一轮 meta refresh 周期。
