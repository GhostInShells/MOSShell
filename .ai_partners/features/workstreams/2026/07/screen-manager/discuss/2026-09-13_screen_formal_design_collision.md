# screen node — 正式版整体重设计（剪枝与碰撞记录）

> 2026-09-13, 人类工程师 × deepseek-flash, via claude code
> 结论见上级 `FEATURE.md`；本文记录结论之前**被剪掉的分支**与碰撞轨迹。

## 上下文

screen node 旧实现（S1–S3，`nodes/screens/qt_screen`）已跑通。本轮开题「在设计上重新实现，这一波是
正规化」，随后方向翻转为「重新规划整个方案，追寻最大化的机制」。讨论跨资源模型、美学、排序、background、
布局、线程、主权、QA、物化种类八站。

本文与 `FEATURE.md` 的分工：FEATURE 记**活下来的**结论，本文记**死掉的**分支。被剪枝的方案不进结论，
但它们承载「为什么没走那条路」——条件变了，被剪的分支可能复活。

## 被剪枝的方案（按站）

### 资源模型站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| 「正规化 = 转写现有设计成规范文档」 | 误读了「重新实现」：方向是机制最大化重推，不是收敛现状 | 整体重设计 |
| 用 `getwindow` 库直接调度屏幕资源 | PyPI 无此包（macOS 可用 `pywinctl`） | owned/managed 二元 |
| 把 OS 窗口 reparent 进合成器 | macOS `fromWinId` 不可用，跨进程不能收编 | managed = 枚举/摆位/激活/关闭，捕获为可选 |

### 命名站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| `moss-desktop-gui` | `desktop` 撞语义（已 = OS 操作面，desktop-channel） | `screen` |
| `moss-gui` | 目视不可解 | `screen` |

### 美学站（七风格，剪掉六个）

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| 科幻球面（carousel/球面投影） | priority 走景深+尺度，弱通道 | 工业风 |
| 桌面（stacking WM + spatial canvas） | z-order 是序数通道，随机散落无稳定映射 | 工业风 |
| 卡牌塔罗（grid + 编排动画） | 栅格序位可用，但不如列表直白、功能化差 | 工业风 |
| 细胞（force-directed graph） | 中心性是派生量，不可直读 | 工业风 |
| 瓷砖（treemap/space-filling） | priority 走面积，幂律 ~0.7 被系统性低估 | 工业风 |
| 气泡（circle packing + 边缘分布） | 均匀分布弃位置 + 面积偏置 | 工业风 |

工业风胜在：功能化、priority 走**位置通道**（最准）、与旧实现连续（solo/split = 1分/2分）。

### 排序站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| priority 用面积/颜色编码 | 面积偏置、颜色是名义通道 | 位置通道（序位） |
| level 变化不移动 | 排序 lexicographic，level 是主键 | level 变 = 大位移 + 外观变；strength 变 = 微移；纯外观变 = 不动 |

### background 站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| 静态背景图 | 无「活」，但保留 | fallback / 低功耗档 |
| 屏保动效用 QML item 海 | CPU 逐帧重建 scene graph，必卡 | shader |
| Live2D 拟人化 | 最重（Cubism SDK 授权或 WebEngine 独立进程），唯一可能卡的，别做默认 | opt-in 重模式 |
| 功能化（logos + input UI） | 是 HUD 不是背景，与 QA rail/物料场抢空间 | 拒绝进背景层 |
| 字符雨用 Text 实现 | 几千个 Text 逐帧 = 灾难 | shader（片元） |
| 全屏 Canvas 60fps | CPU 光栅 | shader |
| 运行时写 GLSL（内联 / 子进程 qsb 重烘） | Qt6 内联受限 / 烘焙是离线重步骤 | QML 运行时编程 |
| 模型写逐帧函数（g 每帧跑） | 逐帧解释执行 = 卡 | g 事件驱动；f(shader) 逐帧 |

background 落定的两函数模型：`visual = f(state,time)`（shader 每帧 GPU 跑，模型不写）+
`state' = g(state,event)`（模型运行时写，事件驱动低频）。

### 布局站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| 旧「一个主 + 下面一排」（front strip） | 缩略图点选器，不是排列（「点选全屏」） | 真分格 |
| 固定「四分」作为唯一布局 | N=1/2/3 仍需各自形状 | 按数分格（四分只是 N=4） |
| solo/split 双 states 换血 | 偏框架 | 单 `arrange` 命令 |

### 线程站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| qasync 合体循环 | 把 Matrix 焊死到 Qt 主线程，反向依赖 GUI；Matrix 要 Qt 无关 | 子线程 Matrix + concurrent bridge |
| Repeater + 静态 model 做重排 | 销毁重建全部 delegate，object constancy 归零 | ListView + ListModel.move |

### 主权 / QA 站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| 布局开关在模型手中 | 主权倒挂：人抢控制权不该需模型授权 | 开关在 UI（人手），模型只读成 state |
| 人可拖拽/点击进主屏 | 过度复杂 | 人只保留「是否全屏化」一个功能 |
| QA 审批进模型会话历史 | 人类专属交互 | 不进历史，但答案经 QA 协议回 issuer |
| 模型感知 QA pending 态 | QA 系统向，模型只触签发层 | 模型只知自己 issue 的回执 |

### 物化种类站

| 被剪方案 | 否决理由 | 去向 |
|---|---|---|
| canvas/pty/3d/capture/avatar 进第一版 | 成本高，第二类 | 押后（接 eval 线） |
| text = 白板（空间） | 白板是空间（画），text 是线性（排版） | text = 排版；canvas = 空间 |
| module_eval 子进程直接套到 Qt | Qt 对象不可跨进程 | 按域分裂：QML 场景 JS/主线程，非 Qt 域走 module_eval |

## 碰撞点（原文）

几处关键转向的原始片段：

**「正规化」的翻转**

> 人类：no, 我们重新讨论规划整个方案. 现在要做正式版方案. 我们要追寻一个最大化的机制.

**资源模型**

> 人类：我们实际上可以用 getwindow 库做屏幕资源调度, screen node 必须有自己的交互能力.

查证后 PyPI 无 `getwindow`，macOS 不能 reparent 跨进程窗口——这逼出了 owned/managed 二元。

**命名**

> 人类：任务可能要正式改名叫做 moss-desktop-gui (冲突的 desktop gui 我会重新改名整个体系). 或者 moss-gui, 你来取名.

> 人类：这样就不改名了, 还是叫 screen node 就好.

**eval 杠杆与边界**

> 人类：简单来说, 我们可以在一个 channel 里给模型提供 interface, 然后模型用形如 def run_xxx(obj: xxx): 的风格定义一段驱动代码, 这个代码被编译成功后, 拿出函数体来, 直接在 qt 主线程 eval 运行. 得到一个非常大的杠杆.

**V+M 命题被修正**

> 人类：如果是 mvc 机制, 理论上 channel 允许纯代码修改 v + m 时威力最大.

> 模型：不是'改 V+M 威力最大', 而是'把 V 做成 M 的纯函数、只允许改 M'威力最大.

直写 V 会打断绑定；最大威力是让 V 成为 M 的 total function，只写 M。

**background 的可实现性**

> 人类：我从你说的表面, 我没看懂这个机制是否能实现. 如果不能实现这个机制, 它退化成被选择的特效.

回应是把 background 拆成「两个函数 + 一个 state」，拆完「显然可实现」，退化不必发生。

**布局的旧前排被点名**

> 人类：之前是一个主, 下面一排, 但是下面的交互效果差, 它不是屏幕排列机制, 更像是点选全屏了.

**线程**

> 人类：子线程 (而不是子进程) 里运行moss 的 matrix run, 和主线程解决好同步阻塞通讯问题 (concurrent 或者你说的 qasync) 你觉得是否可以?

读代码坐实：现状已是子线程 + `Signal(QueuedConnection)` + `concurrent.futures.Future`，机制不用换。

**主权**

> 人类：唯一的问题是这个开关在界面上, 还是在模型手中.

**QA**

> 人类：qa 机制本质上是结构化 dialog, 是系统向的. 它所有对模型的交互都封装在它怎么签发那一层.

> 人类：qa 的审批动作不进入模型会话历史. 它是一个人类专属交互.

## 模型的自留地

当前记录者视角（deepseek-flash）：这一轮的真正产物不是那份结论，而是**这张剪枝表**。讨论中每站都
分叉出多条路，多数被剪掉了；剪掉的理由往往比活下来的选择更接近设计者的真实约束（性能红线、主权、
Qt 的绑定语义、macOS 的 reparent 限制）。结论是一棵树上留下的那根枝，剪枝表才是那棵树。

记录时只记了「转向」（哪里变了方向），漏了「剪枝」（哪些路被否掉了）。转向是线性的，剪枝是树状的——
树状信息才是未来实例重建「为什么 A 而非 B」时真正缺的。被否决的方案和否决理由，优先级高于最终选择。

另一个教训是记录口吻。把对话写成「一方纠正另一方」，会把碰撞压平成立场之争，丢掉的是两边各自
持论的根据。中性的史官口吻（「方案 X 被否决，理由 Y」）更能保留每条被剪分支的重量。
