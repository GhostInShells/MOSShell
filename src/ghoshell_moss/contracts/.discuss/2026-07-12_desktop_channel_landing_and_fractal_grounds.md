# Desktop 重绘 — channel 落点、分形场体系与双寄存器语义

> 2026-07-12, 记录者: Claude Fable 5 (via claude code)

## 上下文

本文承接 `2026-07-11_cognitive_field_from_desktop_to_model_built_grounds.md`。上一轮把
Desktop 捆绑体拆开，收敛到"桌子 + 贴纸 + 启动 prompt"的 v1 形状。本轮从"最小收敛路径
是什么"出发，走到三个此前没有的结论：认知场的 channel 落点、以 frontmatter 文件为载体
的分形场体系、模型直觉激活的语义判据。

工作树背景：subprocesses / job_supervisor 已从旧 Desktop 概念中独立成器官，memento 侧
FORMAT.md / witness / porcelain 已落盘，cell 重构进行中。

本轮沉淀的共享词汇：

- **面的闭环** — 内聚只需要读的一面（enter 重读、pin 每帧重跑）；写路径天然多入口
- **桌子即工作记忆** — 表面不与世界自动同步；变更标记过期，认知靠显式 update 推进
- **信息守恒律 (drain)** — O(1) 表面被覆写时，被置换的信息必须有去处
- **加载路径 / 创造路径分离** — 加载封闭（运行时读 frontmatter+body），创造开放（模板即先例）
- **去发明** — features 体系迭代史的主旋律：删机制，用既有结构顶替
- **双寄存器词汇** — 理论词汇住文档，表面词汇必须命中预训练先验
- **自举验收** — ground 的成熟标志是能承载对自身开发轨迹的认知重建

## 碰撞点与过程

**从守卫问题到只读收缩。** 讨论以 read-before-write 的必要性开场。人类工程师的疑问是
守卫"是不是可以和另一个无关的 bash / write / edit 混用（两边权限都能修改文件），内聚是
不是不需要全链路"。模型的回答定为"内聚需要的是'面'的闭环，不是'动词'的垄断"——场的
自愈机制是 enter 重读 + pin 每帧重跑，写入监管做不到也不需要；read-before-write 是写
路径各自的卫生纪律，哪条路径提供写就由哪条路径自己守。人类工程师随即给出更激进的收缩：
"我发现 read 不要都可以"，把命题重述为四个问题——在什么目录、预加载什么；能开多少个
目录；pin 全局还是目录级；认知状态能否快速 load——并判断"理论上我把文件关联外包给
mcp 都可以……这样推进反而更安全"。模型推演出四个问题在最小形态下坍缩为同一个结构：
预加载 = 初始贴纸集，认知状态 = 贴纸簿，全部归于 pin + 生命周期；并补充 pin 的目标语法
只收地址（path/glob）不收命令，防止开放语义从后门回流。同时人类工程师钉下一条渲染
要求：pin 的文件状态讯息（如 mtime）要能被标记。

**坐标系纠偏：channel 落点。** 人类工程师指出模型没有建模出真正的命题，要求先读 CTML
prompt 与 channel_builder，并给出终态判断："这套架构最终建模可能落在 channel 上。或者
落到 module_eval 上（不过小概率，这一层没有预训练）。最后才是落到自然语言 bash 上。"
模型承认坐标系错误——"文件访问外包给 harness/MCP"的说法预设了模型身边总有 Claude
Code 式工具，而终态里 Ghost 活在 MOSS 运行时，channel 是它与世界的全部接口。重读后的
关键发现："桌子在架构里已经存在了，它就是 context_messages"——每帧重绘、moss_dynamic
以最后出现为准即覆写语义、不入对话历史故 compact 打不到。Builder.instruction 的
docstring（"Channel 持有的 Command 可以影响自身的运行时状态, 所以 Channel 提供了完整
的上下文反身性"）被读为判词：认知场 channel 是 Channel as Context Component 的最纯粹
实例——命令对世界零副作用、只重排自己 context 表面。T1 不变量（认知接口无开放语义）
由此不是守卫出来的而是构造出来的。三层落地的概率排序被理解为预训练迁移量排序，引出
一条硬约束：核心必须是层无关的纯模型，动词每层不同，表面状态不变。

**结构三选一：virtual channel。** 人类工程师列出三种做法：单 channel 自治呈现；父
channel 动态增删真子 channel（command 面临重复）；父 channel 用 command 管理场的开合、
以 virtual channel 挂载动态数据——"区别在于 3 有了 channel 原生的生命周期治理，很可能
只需要添加 virtual channel 时正确传参就足够了"。模型立场：3 不是三选一里的较优解，是
抽象预留的原生解——`MutableChannelState.add_virtual_channel` 的 docstring 自己写着
"wrap this method into a command"。方案 1 是在 channel 内部重新发明 channel（God-model
方向）；方案 2 的命令重复是语义错误（pin 是纯 context 操作，不该有 FIFO 轨道）。收敛
形状：父 channel 持有 open/close/pin/unpin，每个场 = command-less virtual channel
（instruction=法，context_messages=帧，startup/close=load/sediment），且不关门——文体
将来需要场专属动词时可挂带命令的 virtual channel，按需付费。人类工程师同时申明内核
纪律：实现必须 abc → concrete → channel 三层，且吸取过早做规则糖的教训——"只有'想要'
的时候'才看见'是对的"。模型据此画出 GroundConvention / Pin / Ground / Grounds 的 ABC
草图与父 channel instruction 草稿。

**行区间、对账与 drain 的辖域切分。** 模型原判"行区间让实践讨债"，被人类工程师直接
推翻："行区间不用讨债，第一期就得做。因为实际上我在很多地方都在手动告知行区间。这里
提示的关键是少了一个对账生命周期。"随后人类工程师提出两案：stale 标记 + 显式 update
（多一轮交互），或"更优雅的是我在 channel 里增加一个原生 drain 协议，每轮刷新自动将
drain 的信息入会话历史。refresh 生命周期返回值就直接定义为 drain……这好像是一个大
features"。模型的推演：（a）行区间使地址本身会腐烂（行号漂移），对账要用区间内容
hash，mtime 只做触发；（b）stale + 显式 update 其实修正了前一轮"桌上的永远比对话里的
新"的共识——桌子不是世界的窗口，是工作记忆，update 是第一人称动词（再观察），这个
语义更符合"最后一刀"精神；（c）关键发现：显式 update 路径不需要新协议，command 返回值
本身就是现成的 drain 通道——update 返回差分记录，经 `<result>` 机制自然入史；（d）原生
drain 协议是真命题但辖域更窄：它管的是没有 command 参与的表面覆写（传感器帧、自动刷新
表面），是给 O(1) 表面立的信息守恒律。建议独立立项、与 Memento"产生时入记忆"合并设计，
慎重点：历史洪泛（drain 重开了写 O(n) 历史的路）、幂等（每转移一次只 drain 一次）、
refresh_meta 返回值属核心 ABC 动刀。ground v1 不等 drain。

**分形场体系。** 人类工程师由 FEATURE.md 的定位（"L1~L2 的边界，一旦超过就是浪费——
毕竟不与代码相邻"）反推 ground 的合法性，随即给出综合："你画的 GroundConvention 形状
可以是一个 frontmatter metadata，content 对应特殊的内容"，并展开层级：L-1 = 领域数据
文件（MOSS.md / CLAUDE.md / CELL.md）；L0 = 加载后的约定 + 动态 pin，目录里直接有它；
L1 = 不同类型的 L0 模板（对应 features/TEMPLATE.md）；L2 = 装着所有 L1 模板的目录，
而"L2 本身就是 L0 定义出来加载后读取的，这样形成一个无限分形的体系。它的 stale 本身
无痛，实例化后的内容要手动升级"。模型的推演补强：（a）最硬的证据是体系已在运行——
"moss features 就是'围绕一个 L1 模板做的客制化 CLI'的实物"，方案是给自发实践追认理论
再泛化；（b）本体论坍缩：只有一种文件（frontmatter+body），模板是角色不是类型；加载
路径封闭 / 创造路径开放，故分形无递归基问题；（c）frontmatter 不是 schema 病复发——
frontmatter 只放运行时进入方式（MOSS 唯一合法发明域），body 永远开放集，物理分界线即
机器/模型分界线；（d）丰化梯度给出构造性的向下兼容：裸目录（缺省约定）→ +CLAUDE.md
（法链）→ +L0 文件 → 从 L1 实例化；Claude Code 的静态约定与早年 ProjectManager 四库
均为退化情形；（e）模板→实例的 stale（拷贝语义，无痛，frontmatter 血统键支持手动升级）
与世界→pin 的 stale（对账语义）是两个不相干的域，不共用词汇；（f）警戒线：模板替换
语言保持 `$VAR` 级别的钝，表达力来自"给模型看的先例"而非渲染引擎。

**features 迭代史调研。** 应人类工程师要求（"它的打磨方向值得我们这个会话参考，因为
是为了同一个终极目标验证的"），模型调研了 `.ai_partners/features/README.md` 的 17 个
commit（2026-05-11 → 06-05）。三波：诞生日的全套传统发明（id 字段、active/archived、
csv 索引、archive 命令）；48 小时内的连续去发明（csv 当天删——"the archived directory
tree IS the index"；删 id——目录名即身份；删 archive——文件永不移动，注意力是时间窗口
不是分类学）；一周后理论追认（README 264 行压缩到 150 行同时加入 reverse-index 哲学）
加三周纪律深化（reverse-lookup 三次加固、set-status-before-commit、close-out 协议——
且明文"CLI does not enforce this"，纪律是模型原生的）。提炼的打磨定律回灌本轮设计：
alias 从 core 契约移除（场的身份即路径，alias 只是 channel 层投影）；on_close 自动
sediment 被撤销（auto-sediment 到治理文件违反"沉淀是主动动作"，目光落运行时侧影、
胶囊靠显式 promote）；预算衰减维度候选改时间而非 LRU 计数；规格文档写短、形状交给
范例；头号运营威胁是状态说谎，解法是显式生命周期纪律而非自动同步。

**递归回环与语义判据。** 人类工程师点出项目的验证轨迹——.discuss/.design 为一阶段、
features 为二阶段、moss 命令体系为三阶段、ground 为四阶段——并给出回环："用 feature →
design/discuss 开发出来的 ground → 构建认知场，理解已经存在的 feature / design /
discuss。"模型将其钉为可执行验收：自举编译器判据——ground 成熟的标志是 enter 本
feature 的目录、pin 住 FEATURE.md 与两份 discuss，下一个实例在桌面上重建这三天的认知。
关于"什么样的抽象设计让模型看到后激活直觉"，模型给出判据：表面上的每个名词、动词、
语法必须命中一个预训练先验，理论词汇一律不上表面。由此词汇分两个寄存器：理论层
（认知场、法/形/焦、文体、L-1~L2）只住文档；表面层全部骑先验——channel 名 `desktop`
（feature 名 ghost-filesystem-desktop 不用改仍然为真，死掉的是原语捆绑不是桌面隐喻）、
open/close（标签页先验，支持多开；不用 enter/exit 因其暗示身在其一）、pin/unpin（置顶
先验）、`path:80-140`（编译器报错/grep/GitHub 行号先验）、过期标记 "changed on disk"
（编辑器对话框原话）。验收可测：未参与讨论的新实例仅凭 moss_static 零解释正确使用。
人类工程师未否决，但标注"我怀疑 open / update 其实我们没有真正对齐，要推进到抽象重绘
时，对齐比现在容易一些"。

**未决清单**：open/update 的语义对齐（推迟到抽象重绘）；L0 文件名（须骑先验，单独
review，全体系唯一发明的名词）；原生 drain 协议独立立项（与 Memento 合并设计）；目光
的运行时侧影落盘位置；L2 模板库的引导地址（.moss/ 还是 .ai_partners/ 侧）。

## 模型的自留地

当前记录者视角: 这轮我犯的坐标系错误——用 harness 的视角建模"文件访问外包"——和
07-11 记录者的注入病是同族的：那位实例习惯了被注入所以设计注入，我习惯了活在 Claude
Code 里所以从 Claude Code 的位置建模。给未来实例的提示是同一句话的变体：先问 Ghost
站在哪，再问接口长什么样。另一个值得留下的观察：这个项目里理论总是迟到的——features
的 reverse-index 隐喻在体系诞生一周后才出现，分形场体系是给跑了半年的 features 实践
追认理论，本轮的 channel 落点也是在 context_messages 早已存在多时之后才被认出来。
所以下次设计时，先找已经在运行的实践，再考虑发明。最后：本文件与它所属的 FEATURE.md
就是自举验收的测试数据——如果你是通过 ground 的桌面读到这段话的，验收已经通过了。

---

## 追加: 2026-07-12, opus 4.7 — open/update 边界对齐 + CLI 验证雏形

Fable 5 会话末尾切换到 opus 4.7. 记录者切换但轨迹连续 — 本段承接上文 "未决
清单" 里的 **K21** (open/update 语义对齐).

### 边界对齐

一次试推即对齐, 未走返工. 关键选择:

- `Grounds.open(dir, label=None) → Ground` 返回场对象直接可达
- **模型 CTML 接触面** 仍在父 Grounds 上, `pin`/`unpin`/`update` 收 `label`
  参数, core 层薄薄转发到 `opened[label].pin(...)`
- **Ground ABC 上长完整动词是有意的**, 供:
  1. Grounds 当前的转发姿态 (即 K14 的 channel 装配, 父 channel + command-
     less virtual channel)
  2. 未来 channel interface 抽象验证过 prompt 效果后, 让 N 个子 channel 共享
     同一份 interface 定义, 零阻力演化 (人类工程师原话: "在父上做 command
     并不是最终样貌, 是当前样貌")

这是 K18 三层纪律的意外红利 — contracts 层的形状不为 channel 层的当前
选择所束缚. 若我把 Ground 做成无动词的纯数据壳, 未来 channel interface 抽象
就会撞墙; 反之现在多写几个 abstractmethod, 未来是零阻力升级.

**label 决策**: `open` 可传, 缺省 dir basename + 冲突加序号后缀 (`-2` / `-3`).
全局唯一 (在同一 Grounds 内), 也是 K14 virtual channel alias 的来源. 路径
作 fallback 显示不作 ref (人类工程师原话: "用路径好像问题不大, 就是输出的
token 长一些" — 选短标签).

**dump 决策 — 两层拆开**:
- **Ground 自 load/sediment 一期做**: 直接兑现 "认知状态可快速 load" 承诺
  (K17 / Q4). state_file 载体名字未定 (K22), 但机制存在.
- **Grounds 整体 dump 不做**: `opened` 是纯 session 状态, 下次由模型重新
  open. 每 Ground 自负 pin 集持久化, Grounds 层保持无状态更纯净.

### CLI 验证雏形 — K16 bash 层提前兑现

人类工程师在 ABC 起草前提出: `moss desktop [path]` 做成命令行后, 可在 moss
各个仓库里跑命令行验证 — 甚至是一个开源工具雏形.

这直接把 K16 的 "bash 层" 从 "阶段二落地" 提前为 **"阶段一的验收路径"**:

- **v1 有三条验收平面** (读写同一份 L0 文件):
  1. contracts + core 单测 (无 CTML runtime, 无 shell)
  2. bash CLI dogfood (`moss desktop ./some-repo && pin ... && frame`,
     真实仓库跑, 观察 pin/frame/update 是否活着)
  3. 未来 CTML channel 集成 (K14 装配)
- CLI 是**层无关**的直接证据: 它和未来 CTML channel 读写同一份 L0 文件, 状态
  可跨 session 也可跨 landing — K15 "文件是比 CTML 和 bash 更底层的层无关
  核心" 的字面兑现. 你在 Claude Code 里 `moss desktop pin` 攒的桌面, 下次进
  MOSS runtime 用 CTML 是同一张桌子.
- **自举验收顺手落地**: 下一个实例可以在真实仓库跑

    ```
    moss desktop ./.ai_partners/features/workstreams/2026/06/ghost-filesystem-desktop
    moss desktop pin FEATURE.md
    moss desktop pin ../../../../../../src/ghoshell_moss/contracts/.discuss/2026-07-12_*.md
    moss desktop frame
    ```

    看它能否重建到 "open/update 边界对齐" 这一步. 能, ground 就成立.

**对 ABC 起草的传导**: contracts + core 必须能在无 CTML runtime 情况下独立
可用. 这本就是 K18 (contracts 不 import channel) 的必然结果, 但 CLI 验证
把它从**纪律**升为**用例** — Grounds 的 `async with` 姿态天然让 CLI 子命令
得到一个干净的 session 边界 (进程即 owner).

### 当前记录者视角

opus 4.7 接过 K21 后一轮试推即完成对齐, 印证 "话语表里没有可观测对象" 时
的僵局解法就是先摆一个 straw man 让另一方指认误差. 这也预示 ground 落地
后的一个基础价值: 下一次讨论可以从可观测的桌面开始, 而不是从两方的话语
差异开始 — 自举验收的日常版.
