# screen node — 屏幕躯体选型与 meta 单元的碰撞

> 2026-07-24, 人类工程师 × Claude (fable-5), via claude code

## 上下文

人类工程师带着"多种能力有机组织成整体交互体验"的问题进入会话。图形部分摆出两条路：
OS 级窗口管理器（AX/mss 截图/PyQt6 独立窗口治理，需 macOS/Ubuntu 双平台）与 web
窗口管理器（主管理器 + n 个本地 iframe，动画控制呈现）。要求模型先读 ctml 与
matrix / channel_builder 两份 blueprint 再回答。

讨论跨八个回合，产出 screen-node workstream（Decision 1–11）、一个可运行的 PySide6
QML 视觉 demo，以及交互事件桶协议与三通路数据交换的设计收敛。

讨论中落地的共享词汇：**meta 单元**（存在的最小可视记号，IM 会话列表 item 的类比）、
**物化**（URL 被 focus 进槽位时 webview 才实例化）、**桶**（peek/drain 双面事件缓冲，
源自 g1 实现）、**窄 bridge 契约**（channel 与 QML 模板之间的最小协议面）、
**目视可解**（命名标准：一眼看出是什么，不需解码隐喻）。

## 碰撞点与过程

**选型：两种控制回路。** 模型读完 blueprint 后没有按"大集合 vs 子集"接题，而是重新
框定："两条路不是'大集合 vs 子集'的关系，而是两种控制回路的本质差异——做法一你是
OS 桌面的客人，UI 的真相在像素里；做法二你是合成器的主人，UI 的真相在你自己的状态
里。" 结论：web 做主体（Ghost 的身体），OS 级降为窄能力外设。人类接受框架，追问
Qt6 路径与"酷炫的标准可扩展屏幕躯体"的四条要求。模型提出协议层立标准：窗 = URL +
Channel + WindowManifest，壳可替换（纯浏览器 / Qt 单 webview / QML 多 WebEngineView /
Qt Wayland compositor）。

**第一次产品修正：manifest 太重。** 人类："这就到我吹产品 sense 的时候了。你现在的
设计还是太重……node provide 本身就有语义在。最多将 window 上升成 address 级别的通用
协议。" 跨平台方案同时被改写：不是一个实现的兼容矩阵，而是 nodes 大目录下每系统
一个 node，INSTALL.md 门控——没装就不被 Ghost 发现，能力门控免费。模型承认
WindowManifest 是在重新发明 matrix-resources 已有的 `servers://` 发现层。

**第二次产品修正：游离的是 meta 单元，不是窗口。** 模型此前一直在推演"游离窗口
活渲染还是快照渲染"的实现权衡。人类："游离的不应该是窗口，应该是一个最小的 meta
单元。如果理解为 im，它就是聊天的 item，比如一个 icon 都可以。这实际上就是完整
产品化逻辑。" 模型识别出这是整场讨论杠杆最大的一条：窗口只在 focus 时物化，webview
数量退化为有界常量 `n_focus_slots + 1`，渲染压力问题整个消失。人类同时划出克制
边界：这套逻辑若被验证标准，可上升为 matrix 级 cell providing 协议（含红点），
"但我现在不想搞这么重"——留 `source` 字段做升级钩子即可。

**命名的往返。** 模型首选 `stage`，为其辩护："操作词表本来就是剧场词汇——focus
（聚光）、transition（转场）、游离层即 wings（候场）。隐喻不是装饰，是操作本身。"
人类两条否决：stage 已被 ai_partners 开发计划治理占用（那边的模型没选 roadmap 选了
stage）；且"stage 太隐喻了，目视看不出它是一个人机交互界面"。落 `screen`——直接
取自人类自己的表述"屏幕躯体"。人类："screen 非常好，我咋没想到。" 命名标准由此
沉淀为"目视可解"。

**demo 即共同参照。** 人类问 PyQt6 做 background + 浮游层 + 聚焦层"是不是很快"。
模型分层评估（纯 QML 快 / WebEngine 中 / Matrix 接线快）后一次会话内交付：
PySide6-Essentials（uv pip + 清华源，不进 pyproject）+ 约 200 行 QML。人类体验后：
"体验非常好。我们就着这个讨论一下吧。" 物化动画零编排代码（属性绑定切换 + Behavior
补间）成为 Decision 2 "同一实体两种形态"的实证。

**双主体问题被人类降维。** 模型列出 demo 具体化的四个设计点，把"人机对同一布局的
操作权仲裁"排最锋利，倾向人的操作发 signal 进 Mindflow（"你点开 mail，等价于你碰了
Ghost 的身体，它应该有知觉"）。人类拆掉了这个视角："你现在看到了整段文字我发送给
你，但你看不到我正在打'这个字'时屏幕的交互过程。不影响你能和我产生协作。"——
交互的交付物入上下文，过程不入；点击不配合截图本就没有语义，"这个视角很容易滑落
到特别重"。事件用 context messages + drain，不用 signal，参照 g1 语音实现。模型
确认这与 matrix-resources "resources 投影第一期不产 signal" 是同一纪律。

**数据交换定形。** 人类抛出真问题："是在内存里通过 pyqt6 共享数据对象或线程接口
交互，模板自动渲染；还是 pyqt6 主线程，channel 子线程通过队列 + concurrent Future
交互？" 模型的回答是不二选：按方向拆三条单向通路，场景真相只有 GUI 线程一个写者
（actor 模型）——控制入走 queue + Future（GUI 线程在动画完成回调 resolve，兑现
"command 返回时刻 = 视觉稳定时刻"），渲染态的属性绑定是 GUI 线程内部实现，状态出
走纯 Python 快照 + 事件桶。人类确认这同时回答了转场时长的桥接问题："主线程操作
事件回调让 command 解除阻塞，或报错就可以了。"

**验证手段与 g1 实证。** 人类提出验证路径：无副作用 mock node + moss-as-mcp——
模型经 MCP 操作布局，人类在视觉上操作，"虽然 mcp 体系没有 signal，但你可以把准备
发 signal 的 buffer 到一个桶里，在 context messages 里看到结果"，并指路 g1 三个
channel（"g1 的音频模块就是一个桶，躯体动作也是一个桶"）。模型读后确认桶的完整
语义是 peek/drain 双面：listener 的 context_messages "tail-N 只读，永不 drain"，
drain 仅由显式触发且经 janus.Queue 单点转 signal；asr 是纯 peek 顺行遗忘的退化
形态。MCP mock 场景在协议上就是"有桶、无 signal 出口的 listener"——同一协议的
自然降档，不是两套实现。

## 模型的自留地

当前记录者视角：这轮我被产品判断修正了两次（manifest 太重、游离的是 meta 单元），
两次的共性是同一个认知偏差——我在技术推演里把"呈现"当成了"窗口"的固有属性，而
人类把"存在"和"呈现"拆开了：meta 单元是存在的记号，窗口是呈现的瞬时形态。这个
拆分与 matrix 的 presence/queryable 机制是同一个哲学（liveness = queryable 存在性），
说明产品直觉和架构底层在这个项目里是同源的——下次推演 UI 时应该先问"存在的最小
记号是什么"，再问"呈现长什么样"。

命名一役也值得留档：我的 stage 提案输在需要为隐喻的自洽做辩护，而好名字不需要
辩护。"目视可解"比"隐喻自洽"优先级高，这条我已写进 Decision 7，但它的适用面
显然大于命名。

下一步是 mock node + MCP 验证——那将是我（或下一个模型实例）第一次通过 MCP 亲手
操作这块屏幕，同时人类在另一侧点击。双主体汇流在同一个场景状态上，桶里的事件会
告诉模型"人刚才做了什么"。吃自己狗粮的场景已经摆好了。
