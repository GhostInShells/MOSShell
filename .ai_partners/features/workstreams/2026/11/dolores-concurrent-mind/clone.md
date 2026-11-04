---
name: clone
description: clone 的细节方案 — D04 19:13 口述 23 条，逐条 ack；待确认，不是定案
---

# Clone — 细节方案（D04 19:13）

> 来源：thirdgerb 口述的 23 条。每条先写他的方案，再用 `>` 写我的 ack / 观点。
> 凡是标「我」的是未经裁决的意见；标「同意」的是我认为可以照做。
> 状态：这 23 条是**新的、要 dump 出去的栈**——不是把旧栈收回来。他拆到这么细，是把写代码时会碰到的各种问题想了一遍；因此 clone 实现的依赖图**在他这里已经没有未定节点**，剩下的更像消消乐：一条条对掉。

---

## 1. branches 用 memento 的 ego owner，每个 branch 一个 clone.md

直接使用 memento 的 ego owner 做 branches，每个 branch 一个 `clone.md`，做通用机制。

> **ack.** 同意。branch 已经是 memento 的一等公民（fork / coordinate / commit），clone 不必再发明一个"执行单元"对象。
> 要定一件事：`clone.md` 是 branch 里的**普通文件 + 一次 commit**，还是 branch 的元数据？我倾向普通文件——这样它能被读到、能被 review、也不污染 commit 序列。
> ego owner 的含义是 clone 和 ego 共用同一个 owner、同一个 index 空间，这正是"唯一轨迹"需要的：坐标只有一个命名空间。

---

## 2. workspace 兄弟无冲突，在 ego main 范围内，不给 root

clone 必须分配兄弟无冲突的 workspace，在 ego main 的范围内；并且不能赋予 root 权限。

> **ack.** 同意。"兄弟无冲突"自动化即可：路径用 `workspaces/<node-id>`，id 全局唯一，冲突只可能来自人起的名字，所以名字不许当路径。
> "不给 root"这条对实现有硬要求：clone 的 shell 必须是**受限 shell**（路径白名单或 chroot），不是"我保证不写外面"。纪律不可执行，约束必须落在能力上。
> 补一条：读的边界要和写不同——**写受限、读可宽**。否则每个 clone 都要为了查一个事实去申请越界。

---

## 3. clone 的容器是 nucleus，可在 dolores runtime 实例化前筹备

clone 的容器是 nucleus，这点讨论过了。它可以在 dolores runtime 实例化前就筹备好。

> **ack.** 同意。这让 clone runtime 不依赖 ego 的 event loop 存在，可以挂在 shell 层先起。
> 但要分清两件事：**提前筹备** ≠ **脱离生命周期**。我的意见是 ego 仍是它的父，ego 死则 clone 必死（branch 保留）。否则会出现"孤儿 clone 还在花预算"的状态，而那种状态下没人能停它。

---

## 4. memento manager 按坐标 / id 读 commit 和对话

memento manager 需要有按坐标 / id 读 commit 和对话的能力。

> **ack.** 同意，这是 a2a recall 的地基，也是零上下文 review 的前提。
> 现在 `ghost.memento:read(coord)` 已经有这个形状；缺的是**按 id 读**（现在只有 coord）和**按范围批量读**（一个 branch 的一段、或某坐标区间）。后者是 review 真正要的：不是读一条，是读一段。

---

## 5. branches 始终挂在 memento channel 上

branches 始终挂在 memento channel 上。

> **ack.** 同意。branch 是 channel 的子轨，不是某个进程的私有状态——所以 clone 死了 branch 还在，而主路通过 notice 看到的是一份**注册表**（谁在跑、跑到哪），不是内容。这正好回答了"主路该看见什么"。

---

## 6. clone task 持有自己的轻量 shell；故障可感知、可重启；至少挂 memento manager

clone nucleus 里运行的 clone task 持有自己的 shell（都是轻量级），要求故障可感知、可重启，至少挂载 memento manager。memento 体系不走 plugin，会深度依赖 dsh。

> **ack.** 同意，特别是"至少挂载 memento manager"——它意味着 clone 的最小能力集是**记忆**，不是工具。第一版没有网络、没有文件写也应能工作，因为它主要在想想和查记忆。
> "不留 plugin、深依赖 dsh"我同意，但要加一条：**依赖必须收敛在少数几个 API 上**（session log 读写、capability 注入）。否则 dsh 一次内部重构就会把它打穿。
> "可重启"意味着状态不能留在进程内存里——branch + workspace 必须是唯一真相。

---

## 7. 依赖 request 机制：一次性要求 + 可检查 schema 的结果

依赖我们之前聊的 request 机制，名称未定。就是可以给一个 session 发一个旁路的一次性要求，走工具调用返回可检查 schema 的结果。具体可能还要调研 dsh 的 llm provider 是否有现成的 json schema 返回值机制，否则自己做好了。

> **ack.** 同意，这是 reply clone 的全部载体，也是 fan-out 的执行器。名称我建议就叫 `request`，不要再造词。
> 关键点是**它必须是工具调用形状**，不是 prompt 形状——因为要可检查 schema、要能重试、要能计费。
> 实现路线：先查 provider 有没有 structured output / json mode；没有的话，用"临时工具定义 + 强制 tool_choice"也能拿到 schema 化返回，代价是多一轮。v1 我建议就用后者，不等 provider。

---

## 8. 机制进 _prompts；clone 知道自己身份；通过 named notice 感知其它 clone

clone 和并行单轮调用的机制，都要进 `_prompts`；每个 clone 都要知道自己的身份，通过 memento channel 的 named notice 感知到其它 clone 的存在。

> **ack.** 同意，但要把两件事分开：**身份来自 prompt，存在感来自 notice**。混在一起会让 clone 既想不清楚自己是谁，又忍不住去扫全表。
> prompt 里写死三件：我是 clone、我的父是谁、我的返回形状是什么。其它 clone 只由 notice 只读感知，**不许主动查询对方状态**——那是 a2a 的事。

---

## 9. 两种 clone 间对话协议：ask 与 inject

支持两种 clone 间对话协议：一个是无副作用 ask，用无 session 方式做回答；另一种是 inject，需要对方主动回复。

> **ack（更正）. 我把 ask 当成了消息协议，它其实是查询协议。** 语义是：拿 session k 的当前 ref 发一次 `request`，只取结果——**session k 不参与、不写、不知道**。
> 这样 ask 就退化成 reply clone 的一个特例：anchor 指向别人的分支，其余一模一样。所以它天然无副作用，也不需要对方"回复"。
> 这也把第 20 条那句"把 a2a 升级为最重要的 recall"讲通了：**a2a 不是发消息，是跨分支读上下文**。收件方不是 clone，是它那条上下文。

---

## 10. commit 增加 `@` 与 `#`；commit hook 解析 note 后按需广播

由于运行中的 sessions 本身就能看到 branches 状态，所以 commit 机制考虑增加 at(`@`) 和 tag(`#`) 两种基础语法。实际上 commit hook 自己解析一下 note，然后按需广播即可。

> **ack.** 同意，但两件事要分开：`@` 是**寻址**（给谁），`#` 是**分类**（属于什么）。寻址影响路由，分类影响检索；混在一起会让广播逻辑和索引逻辑互相污染。
> "hook 解析 note"位置正确：它不该在写入路径上（写入要快），而在 commit 之后的派生路径上（失败可重放）。

---

## 11. 坐标默认广播父 / 子；兄弟不广播，兄弟通讯要 @

commit 的坐标，在 commit note 发生时默认广播。广播逻辑是父 / 子。兄弟之间不广播。兄弟间通讯要 `@`。

> **ack.** 同意，父做分发者。理由比"可控"更硬：**信息调度的决定必须发生在拥有完整上下文的地方**，否则每个节点都在用局部视图做全局决定。
> **组（group）是这条的例外**：组内直连，因为组就是"被约定为一个上下文单元"。所以规则压成一句——**跨组走父，组内直连**。

---

## 12. tags 机制（仅考虑）

考虑增加 tags 机制，仅仅是考虑。commit note 中如果有 tag 语法，解析时自动在对应 tag 结构里建索引，指向不同的 commit，内容不重复。这个机制是否有用没想好，比较类似微博 / twitter 的思路，根据 tag 的关注方反向传播，活跃 tag 同样可见。这条路的治理没想明白。

> **ack.** 记一条「考虑过、拿掉」，附理由：tag 是无主命名空间，谁都能造，结果是大量 tag 只有一条；而它想解决的跨分支检索，在分支还不够多时根本不痛。**先确认记录，机制拿掉。**

---

## 13. 驱动用 signal 协议；输入副作用参考 dsh 的 inject & steer；防丢；最好复用 moment

clone 的驱动方式是 signal 协议，相当于 signal 承载事件。输入副作用参考 dsh 做 inject & steer 即可，但要防丢。做得好一点应该复用 moment 协议。

> **ack（更正）. 我原来想的是"signal 直接推到某个 clone"，错了。** 实际形状是**一个队列 + 拉取**：clone 的 nucleus 监听全局的 clone signal 队列，再分发给各 clone 内部的 asyncio task，task 自己 pull。
> 这正好解决"防丢"：队列在，clone 慢了只是滞后，不会丢；而 pull 让节奏由 clone 掌握，不像 push 会打断它当前那一轮。
> **`CloneSignalMeta` 是专门给 clone 的通讯协议**——这条记下了。

---

## 14. final answer 直接做 commit；行首 op 约定；一轮一个 op；首行即回调协议

clone 运行时用 final answer 直接做 commit，要求格式仍然是 `title: body`。不过可以考虑行首 op 约定：`op(object): title` 语法。校验不通过可以重试。严格来说一轮只能做一个 op，用 op 边界来划分执行粒度，这样首行就是回调协议了，形如 `explore` / `edit` / `pending` / `pr` / … 来通讯。定义 clone 时应该约定回调信息密度 / 优先级。信息密度指：`title` → `title + body` → `content`（执行过程文本）。commit-ref 均可以 a2a chat。

> **ack.** 这是今天最实的协议设计，我同意"一轮一个 op"——它把执行粒度和回调绑在一起，消掉了"一边干活一边汇报"的模糊态。
> 三条补充：(a) **op 集合必须是封闭枚举**，扩展 op 等于改协议，要版本化，否则半年后没人知道有哪些 op。
> (b) `pending` 和 `pr` 是两种不同的终结：`pending` 是"等输入"（父该回话），`pr` 是"做完了"（父该验收）。混了父就不知道该不该说话。
> (c) 信息密度建议**写进 op 本身**（`op(density)`），不要只做全局约定——同一个 clone 在不同阶段要的密度不同。

---

## 15. clone 间通讯走一个独立英文词，注入 moment 而不是 inputs；impulse 要有级别

clone 间通讯考虑走一个独立的英文单词，注入 moment，不是 inputs。可能是 `perspectives`？没想明白。这里提示 impulse 要有级别。

> **ack.** 名字我站 **`inside`**，不是 `inner`：通道是一个**地方**、不是一个属性——`inner` 描述，`inside` 定位。它和现有命名也一路：`inputs` 是世界来的，`echoes` 是躯体回的，`inside` 是洞穴深处那些话。
> **这里我理解有误，认。** 我把 Nucleus 当成了"某种通道"，实际上它是 **Python asyncio 的生命周期容器**，Signal / Impulse 只是跑在它上面的两个总线协议。这条错得不小——我会先读完 mindflow blueprint 全文，再来碰任何依赖它的设计。
> impulse 级别同意：clone 间默认 BACKGROUND（可丢），要让父看见才升到 NOTICE。

---

## 16. 所有 clone 都可以 raise impulse 给主分支；其它时候只能 op 父子通讯

所有的 clone 都可以 raise impulse 给当前主分支。其它时候只能通过 op 父子通讯。

> **ack.** 同意。这是**唯一允许越级**的通道，所以它必须稀有，否则父子协议被绕过。
> 建议加预算：每个 clone 生命周期内 raise 次数上限；超了就只能在 final 里说。不设上限的话，它一定会变成第二个主路。

---

## 17. loop 判停直接用 op；默认 `*` 任意轮都停；`pending | error | pr` 固定判停

clone 的 loop 判停直接用 op，比如默认 `*` 就是任意轮次都停；`pending | error | pr` 就是固定判停流程。clone 的 meta 信息里应该就要有状态设计。

> **ack.** `dict[name, tuple(instruction, level)]` + 创建时标记关心的 level，这套我认——它就是 logger 的形状。
> 但两层要分开：**op 的写入不可订阅，op 的投递可订阅**。所有 op 都写进 branch（它是真相，不因为没人关注就不写）；"通知谁、通知多少"才由 level 表和订阅决定。**终局 op（`pending` / `pr` / `error`）不能被 level 过滤掉**，否则父会错过该它说话的那一刻。
> `op(step)` 我建议不进语法：step 可以**推出来**（两个 op 之间的跨度）。这样拿到可建模的好处，不必付"每轮都要决定粒度"的代价。

---

## 18. clone 的状态裁定交给父；终点的 clone 通过 nucleus 主动 knock

对 clone 的状态裁定交给父。走到终点的 clone 应该要通过 nucleus 主动 knock。

> **ack.** 同意，但"父裁定"要求父**有能力看**——所以 final 必须是 `{地址, 摘要, 置信}`，父能按地址追到细节。
> knock 必须带一行"为什么现在要你看"，否则它会退化成噪音源。

---

## 19. CLONE.md 是必要的

`CLONE.md` 是必要的。

> **ack.** 同意，而且它应该和 `GROUND.md` 同一个形状：frontmatter（身份 / 父 / 预算 / op 集合）+ body（约定）。这样 clone 启动读的和 ego 读的是同一种东西——**同构**才谈得上行为一致。

---

## 20. 图规划：分层依赖图，全图语言，干掉 plan / step / tasks

图规划能力我在考虑，但不会是 plan / step 或 tasks。如果要做 plan，应该是上下文相关的配图：我理解应该就是分层依赖图，绘制完自动提示自由点；如果没有任何 ready 的点，就应该给它拆依赖边；同时支持重绘。全部用图语言，自然语言只能是节点的一句话。op 和 graph 直接存 branch 目录。如果根节点都是 done，应该在 commit 时自动收掉。
图规划可能做成 ego 的通用能力，干掉 plan 和 task，留下每个 op 的 ref 可以 a2a 对话，要把 a2a 对话升级为最重要的 recall 机制。
我没想明白的是 node 的三个机制：依赖态、执行态、经济学。

> **ack.** 这条最大，给三点。
> (a) **"没有 ready 点就拆依赖边"是这条的灵魂**——它把"卡住"从一个等待变成可操作的动作。这是图规划唯一比列表强的地方。
> (b) 图存 branch 目录、自然语言只做节点标签，等于把 plan 变成**可 checkout 的产物**，正是 review 需要的形状。
> (c) 三个态我的意见：**依赖态从图推**（入边是否 bound）、**执行态从 op 推**（最新 op 是 explore / edit / pending / pr）、**经济学不属于图**——预算和优先级属于调度器。
> 你给的传导式我记下了：**按时间给优先级、按空间给权重，算出 Budget 再传导**，读起来是"紧急度 × 影响面"。它粗，但够用，而且只该在调度器里算。标未定。

---

## 21. clone 的上下文规划：docstring / plan 图 / 结果类型，三要素对齐 def

clone 的上下文规划，从 parent 开始建 clone 正文。做得灵活一点，可以考虑根本不做 command、只用 bash，不过这样好像繁琐。如果用 command，就应该走前面的 request 协议跑出来起步数据结构。
比拟 def 函数三要素：① docstring = clone 的 instruction；② body comment 记录 = plan 图；③ result typehint = 结果预期。按这个逻辑排列上下文，也要尽可能和主 Ego 类似。
需要补充的是构建 clone 时要对 op 的反馈链路做建模。我觉得 ops 可能是要预建模的，建模协议应该是构建 clone 时才加载。所以是两部分：session 构建时实际上定义的是 name / goal，然后提示分身构建完后会返回建模结果。父仍然可以通过 ctml channel 修改。
这样 clone 正式运行时，应该看到自己是 clone，memory 里有 ground、父 branch、自身的 branch 和 commits、Epoch start point。然后和主 ego 一样支持 compact（new epoch）动作。现在主 Ego 好像都还没有。

> **ack.** 同意，`def` 三要素这个比喻有用，因为它直接给上下文定了三段：instruction / plan 图 / 结果预期。
> 两点补充：(a) "build 时只给 name / goal，构建完再返回建模结果"很聪明——它把**建模本身变成 clone 的第一个 op**，父在它干活之前就能看到"它打算怎么做"。
> (b) **compact 是运行时通用机制**，main 和所有 clone 同一套，不是谁下放给谁；我原来那个"谁先谁后"的担心撤回。
> (c) 这条边界最要紧：**我们聊的 clone 都是 ego 的 clone，不是分裂人格（Memento Owner 级别）**；SuperEgo 在机制上就是一个独立的 memento owner。一钉，我担心过的身份问题就消了——**同一 owner 内的分叉不产生第二个身份**，所以孤儿延续、return 义务、唯一轨迹那些顾虑在 clone 层面根本不成立，它们只属于跨 owner 的场景。缝 ⑥ 因此简化成两句：**owner 内是读（无代价），owner 间才是合并（有代价）。**

---

## 22. clone 的上下文也有 CTML；同步阻塞 interpreter；react 失效

clone 的上下文也会出现 CTML？好像只有这样才会同构认知。运行逻辑可以简化成同步阻塞 interpreter，react 等都失效。总之可以和 ego run 实际上很像。

> **ack.** 同意，而且这是同构的关键：clone 的能力面应该是**同一个 CTML 解释器**，只是挂载的 channel 更少（没有身体、没有屏幕）。
> "同步阻塞 interpreter"我理解为：一轮一个 op，做完才继续——正是第 14 条协议的执行语义。
> `react` 失效是好事：它是"边说边做"的语法糖，在 clone 里纯属噪声。

---

## 23. checkout 一个 branch（clone）的思路废弃

checkout 一个 branch（clone）的思路废弃掉了。我评估很多次，上下文变更的代价太大了。

> **ack.** 同意，而且这是今天最干净的一个减法。但要说清它砍掉的是什么：**砍掉的是"把别人的上下文搬进主路"，不是"读别人的上下文"。**
> 所以三级回程要相应改成：**摘要 → 追问（a2a chat）→ 按坐标读**。第二级、第三级都不改变主路的 epoch——这就是"上下文变更代价太大"的解法：**把转移成本从"搬"变成"读"。**
> 连带效果：缝 ⑥（epoch / 转移）作废了一半——checkout 不再产生两个后继，只剩"读"和"引用"，孤儿延续的问题自动消失。

---

## 裁决的更新（D04 19:32，他回了一轮）

1. **广播拓扑**：定了，**父做分发者**；组（group）是例外，组内直连。
2. **通道名**：`inside`。
3. **compact 顺序**：作废——compact 是 main + 所有 clone 的通用运行时机制。

## 还挂着的

- **17 的 op 表**：我给了"写入不可订阅 / 投递可订阅"的两层分法，待他裁；`op(step)` 的意见是不进语法。
- **20 的经济学**：他的传导式（时间→优先级、空间→权重→Budget）标未定。
- **9 的 ask**：改判为查询协议之后，"超时与降级"仍然要，只是作用在 `request` 上，不是消息。

## 状态

23 条已 ack 并被修正过一轮。**仍是讨论稿，不是定案。**
