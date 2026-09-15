---
title: File Editor — 就文本文件的对话线
node: nodes/os/file_editor
created: 2026-09-15
status: axis1-landed
---

# File Editor

> os-control 的子文档(KD5 展开)。能力落在 `nodes/os/file_editor/`。
> 它是**对话线**,不是编辑器——对话是目的,导出到磁盘是唯一的真实副作用。

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
| `Thread` | 容器:一条绑定文本文件的对话线(`label` / `motivation` / `path` / `versions` / `order` / `head`) |
| `Action` | 线上一次动作(`seq` / `author(g\|u)` / `kind` / `description` / `payload` / `effect` / `verdict` / `replies`) |
| `Version` | 文件链上的一个状态节点(`parent` / `action_seq` / `content` / `effect`),v0 = 载入基线 |
| `Effect` | `before` / `after` / `diff`(unified) |
| `Reply` | 挂在 action 下的对话条目(`anchor(intent\|effect)` / `diff` / `text` / `verdict`) |
| `Seq` | 站位坐标 `(thread_id, n)`,per-thread 单调递增 |

`payload` 语义:mutation 的 payload = 结果全文;`rewind` 的 payload = 目标 version
id;`reference`/`export` 的 payload = region / 路径。

## 规则

1. **FIFO 生效**:同一 thread 上的 mutation 逐个生效;`reference` 无副作用、不进效果序。
2. **confirm → apply**:mutation 被 confirm 才推进 `head` 产出 Version;非 mutation 只改 verdict。
3. **reject 级联**:拒某条 → 同 thread 其后全部 rejected。这是 store 的纯函数,模型**不用**逐个 close。
4. **rewind = 普通 action**:`result_content` 把 payload 解析成老 version 的 content,链保持线性,不分叉。
5. **人类回复 = Reply**:人类点 action 的 `description`(intent)或 `effect`,就地编辑后回复,载荷是
   unified diff + anchor;它**不改变对象**,模型读 diff 后自己决定是否发对应 action 实现。

## 关键决策

- **K1 对话线,不是编辑器**(见上)。"file editor" 名字有方向性歧义,槽位暂留,
  名字随机制继续浮。
- **K2 多写者**:`author` 可取 g/u,双方都能发变更 action(合写一篇文章)。之所以安全,
  因为 action 是**内存操作指针** + 可 rewind + verdict 级联;冲突靠"谁不同意谁 reject",
  不靠锁/CRDT。
- **K3 上行信号用系统 `silent` / `notify`**:所有需模型感知的界面交互走 silent(聚合、
  低污染),关键请求(一组 pending 被 confirm)走 notify(不丢)。**"反馈不丢失"的保证来自
  store(唯一事实),不来自 signal**——silent buffer 上限 20 会丢旧,所以 notice/context 必须
  是无损投影。信号体自解释:`channel + thread_id + action_seq`,seq 是共享坐标系。
- **K4 分组边界 = channel idle**:两次 idle 之间的一段 burst 就是"一组 action",confirm 的
  粒度。不用 turn 抽象,不用 selection UI。
- **K5 三轴分离**:数据结构(纯 dataclass + 纯函数,可无实物单测)/ 通讯协议(上行交互、
  下行流、query)/ UI(只渲染)。Python 侧真实对象四个出入口:`as_channel` 控制面 + channel
  输出 + ws 输入 + ws 输出,全部由同一 store 派生。
- **K6 持久化**:append-only JSONL log(每 op 一行),`ThreadStore.replay()` 重放重建;
  至少最后一个 version 不纯丢。rewind 也因此免费。

## 三轴与状态

| 轴 | 内容 | 状态 |
|---|---|---|
| 1 数据结构 | `structure.py` + `store.py` + log/replay | **落地,14 单测通过** |
| 2 通讯协议 | uplink 交互动作 / downlink head→deltas→tail 流 / query | 待建 |
| 3 UI | 左主轴 action 流 + 右 thread(object)视图,Reply 就地编辑 | 待建 |

下一步:轴2 协议(交互动作清单逐条分析:reference / mutation / reply / confirm /
reject / rewind / export)。
