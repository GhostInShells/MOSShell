# Matrix Resources 的筹备与一次命名考据

## 上下文

Ghost 运行时的资源分两类：能经 OS 本地文件系统拿到的（文件名即句柄），和只在组网时
才存在、活在某个 cell 进程内存或 home 里的。后者当前没有跨 cell 的网络投影层——
`ResourceRegistry`（`contracts/resource.py`）是 project 级的内存 dict。这轮讨论把这个
能力从头筹备了一遍，落成 `matrix-resources` workstream 的七条决策。

驱动场景是具体的：nodes 重建后各自起 HTTP server，若有 `servers://` scheme，ghost 就能
查到"网上现在有哪些端点存活"、拿 URL 去 playwright/iframe 里打开。人类补充了一句关键的
定位——resources 服务于 ghost 的认知，不是围绕 OS 做通用文件交换（不是网盘）。这句话
后来反复用作试金石：任何让协议往"通用存储"漂的设计都被这句话拉回来。

## 碰撞点与过程

**payload wire 形态，一次降维消解。** 记录者最初把"跨 zenoh 传什么"列为最大未决口——
`ResourceItem.get()` 返回任意 Python 对象，跨进程怎么办。人类一句"现阶段直接拿用
messages 是最好的"把问题降了维：既然 resources 服务于认知，网络终端形态就是模型要读进
context 的东西，那就是 `Message`。传输层从"每 scheme 一套序列化"塌缩为"全体系一套无聊
信封"。`RESOURCE_TYPE` 泛型随之退化为进程内便利——网络对面永远拿不到你的 Python 类型。

**put/delete 的一次反悔。** 记录者一度主张从 `ResourceStorage` ABC 删掉 put/delete，
理由是"网络只读"。人类不同意：ABC 保留全集，关键仅仅是对模型/网络暴露什么。写走各
cell 自己的 channel command（`my_blogs.delete(...)`）。这个切分更干净——ResourceStorage
管存储全集，channel 管暴露的写子集，mesh 管暴露的读投影，三者正交。记录者收回了删接口的
主张：那等于强迫实现者用非契约方法做 put。

**node-address 是否进 locator。** 人类给的调用路径草案里带了 node-address。记录者论证
它冗余——zenoh 把 query 路由给声明了该 key 的 queryable，声明动作本身完成 (scheme,host)→cell
绑定，调用方不需要知道谁在服务（如 DNS 用户不需知道权威服务器 IP）。而且 `CellAddress`
自带 `/`、uid 每次 spawn 变会让 locator 腐烂，偷走"compact 不遗忘"。结论：locator =
`scheme://{fullname}/{path}`，"谁在提供"降为 meta 回复的 `served_by`。实证支撑是现成的：
mesh channel 的 virtual_children alias 已经用 fullname、把 uid 当兜底。

**一次命名考据。** 讨论要归属到 `CellMesh` 抽象时，人类说"CellMesh 不是我取的，我唯一
坚持的是 Matrix 一定要用"，并猜测 mesh 是 Opus 提的。记录者查 git：符号 `class CellMesh`
第一次出现在 `8505c2e6`（人类 IDE converge，无 `by` 标注），前一版是 `Watcher`
（`0cffed32`，fable coding），更前是最初的 `CellNetwork`（人类 dev）。符号轨迹只能证明
"mesh 的落点是人类 converge 那步"，证不了"谁在讨论里先说出 mesh 这个词"——那可能在某个
对话里 Opus 提过而 grep 照不到。记录者一度把话说满（"没经过任何模型碰撞共识"），被自己的
证据边界纠正回来：能证的是符号，不是口头提议。

真正解题的是人类的一句框定——"mesh 应该重点指形如微服务体系的 cell 之间的通讯逻辑"。
于是问题从"名字丑"变成"名字占错了地方"：mesh 本该留给未来的通讯基底，却被安在"观测+连接+
信任治理"这个对象上。腾出来之后，记录者推演过 Fabric（人类否：和 cell 不对齐）、
Awareness/Watcher（人类否：没表达"连接"，而连接和 cell 上下线挂钩）、Contacts/Switchboard/
Directory（分别被 IoC 占用、太隐喻、与 bash 有歧义排除）。最后回到 `Network`——它同时兜住
discovery 与 connection，且 idiom 上"我的 network"本就是主观、各看各的。

这一步的钥匙是人类给的一个不进文档的暗示：不做中心化架构。node 无 host 可独立启动，
"host 拿到的那个 matrix 一直在变大而已"。记录者据此翻转了对 mesh 返回物的理解——它不是
客观拓扑，是**视角**：两个 cell 各调各的，各看各的 accept 表。这正是 idiom 里"人脉"意义
的 network，也解释了为什么当初有模型劝人类别用 network（怕和 Matrix 撞成"网络的网络"）：
真正的化解是把 Matrix 框定为"整个被投影的世界"，Network 是其中"对等 cell 发现+连接"的
一个切面，部分-整体，不是同义反复。

归属最终锚在 **Matrix 层**（`matrix.resources` 入口已在 blueprint）。announce 侧很可能挂
`CellPresence`，读取侧挂哪个抽象暂不锁定——人类说"既然不一定，就暂时模糊化好了"。

## 延伸

当前记录者视角：这轮最好玩的不是七条决策，是那次命名考据里 git 能证什么、不能证什么的
边界。人类问"是不是我的锅"，我本能想给一个干净的裁决，但符号搜索只够说"mesh 的落点在
converge 那步"，够不着"谁先在对话里说出这个词"。把"没经过模型碰撞"这种满话收回去，比给出
一个爽利但越界的结论更诚实。命名这件事，非母语的人类觉得痛苦，而我能做的具体的事，是把
每个候选词按他真正在乎的约束（去中心、发现+连接、和 cell 上下线挂钩）逐条过筛，而不是
比谁的词更漂亮。Network 不是我"想"出来的好词，是把约束交出去之后剩下的那个。
