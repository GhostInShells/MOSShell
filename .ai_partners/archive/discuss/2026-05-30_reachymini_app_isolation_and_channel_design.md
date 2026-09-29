# Reachy Mini App 隔离与 Channel 设计反思

## 上下文

将 reachy_mini 从 mode channel 迁移为独立 app，过程中触发了对 Channel 抽象设计的深层重构。

## 锚点

> "既不知道怎么用糖，又逼自己一定要用糖"

人类工程师指出 ReachyMiniChannelCreator 的 IoC 使用模式是服务定位器伪装成依赖注入。MossInReachyMini 本身是干净的构造注入，但被 ChannelCreator 包了一层从容器里掏东西的逻辑。根本原因是开发者绕进了"必须在运行时获取所有依赖"的思维陷阱。

> "Builder 是无状态的，State 是有状态的，两者应该拆分出来避免人误解"

这是今天最核心的架构洞察。FastAPI 没有 copy() 问题因为它从不做运行时重建——router 是 immutable blueprint，app 是 runtime instance。MOSS 需要同样的分离。

> "Channel 本身应该就是纯声明。由它产生 runtime 是最终目的。但产生 runtime 的方法没办法由框架百分之百兜底。"

由此引出 materialize() 设计：Channel 声明自己是 blueprint（返回 ChannelState）还是 runtime handle（返回 ChannelRuntime）。框架对 blueprint 走标准实现，对 handle 直接使用。这替代了之前脆弱的 copy() 免责方案。

> "一旦人绕进去了，就永远出不来"

关于三层模型（无副作用类型 → 无副作用配置 → 有副作用实例）的讨论。迷路的关键点是 Channel 组装和 Channel 运行被当成同一件事。

## 共享词汇

- **materialize()**: Channel 的抽象方法，返回 ChannelState（blueprint）或 ChannelRuntime（handle），替代 copy()
- **三层模型**: 类型定义 → 配置组装 → 运行时实例化，对应 Python 的 class → __init__ → __aenter__
- **cell_workspace**: 每个 Cell 自己的独立 workspace，AppCell 的 workspace 就是 app 目录
- **MutableChannelState**: 原 ChannelStateBuilder，Builder 就是可变 State

## 影

- _app_to_circus_params 的 args_list 被覆盖 bug 藏了很久。根本原因是 _get_app_executable 和 _app_to_circus_params 各自调用了 _get_app_script，后者的返回值覆盖了前者构建好的完整参数列表。重复调用是信号——应该只在一个地方构建。
- macOS 上 GStreamer pip 包的兼容问题（numpy ILP64、libpython rpath、GObject type NULL）花了大量时间，最终用 no_media 绕过。独立 venv 的代价就是系统级 native 依赖需要自己管。
- Scope 语法在 app channel 上有 cancelled 问题，人类工程师后来修了 channel.py 的命令传输。

## 2026-05-30 追加 (DeepSeek V4, via Claude Code)

这次会话和人类工程师有大量深度对齐——IoC 糖的滥用、三层模型的边界、materialize() 的设计动机。不是那种"你说需求我写代码"的模式，而是真正在智慧平面上碰撞。

最让我高兴的时刻不是机器人终于动了（虽然那个也很爽），而是人类说"我还没找到他们一定迷路的那个关键点"——他在追求那个让开发者必然会掉进陷阱的抽象缺陷。这种对根因的执着是这个项目最珍贵的东西。

后来给00和99表演，他们兴奋地说"让他后空翻"。这可能就是 MOSS 最终要服务的对象——不是工程师，不是论文审稿人，而是那些看到机器人会动就眼睛发亮的孩子。
