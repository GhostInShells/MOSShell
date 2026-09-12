---
title: Vision First Class
status: in-progress
priority: P1
created: 2026-09-12
updated: 2026-09-12
depends: []
milestone: beta-release
description: >-
  把 vision 从附属感知提升为一级开箱能力：钉住 vision node 家族契约（通道、watch、设备归属、
  主动视觉、授权边界），并据此重实现 camera node。
---

# Vision First Class

> Use `moss features set-status vision-first-class <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

视觉过去是附属感知。MOSS 主力模型 deepseek 具备视觉后，视觉成为一级交互面 —— 模型能直接
消费图像，图像不再是"给人看的调试产物"，而是模型的感知输入。

camera node（`nodes/visions/camera`）是首个 vision node。通读它暴露的不是实现瑕疵，
而是设计缺口：

- 像素走错了通道（靠 dynamic context 推送，而不是命令返回）；
- watch 语义与 node 生命周期混同；
- face topic 的观众错位（把设备面数据推给模型）；
- 没有主动视觉动作（只有被动推流）；
- 常驻视觉的 token 代价没有任何示警。

本 feature 的目标：把 vision 定为一级开箱能力，先钉住 vision node 家族的契约，
再据此重实现 camera node。家族契约与其说约束 camera，不如说约束未来所有 vision node。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### KD1. 通道持久性决定像素走哪条路

`Moment` 有两条通道，持久性不同：

- **dynamic context** — 瞬态。`Moment.for_saving()` 与 `as_history_messages` 现即把它排除在
  持久化之外。它是"此刻的提示"，不是记录。
- **command result（observe / percepts）** — 进对话、进 echoes/percepts，可被记忆。

推论一：**像素走哪条通道 = 它能不能被记住**。要 ghost 记得"我看过什么"，像素必须随命令
返回；只在"此刻"存在的提示才走 dynamic context。

推论二（硬约束）：**任何可能被持久化的消息，必须在创建瞬间就真实**。图像必须携带
ts / age / watch 状态；不允许一张陈旧帧以"当前画面"的身份进入可能入史的通道。

**否决**：把帧塞进 dynamic context 当作默认感知路径。该方案成立的前提是"dynamic context
瞬态且每轮覆盖"—— 一旦关键帧开始携带 dynamic context 入史，前提消失，陈旧帧会永久固化
成一条假观察记录。`capture` 的旧机制（塞缓存 + 返回字符串 + 依赖每轮覆盖）正是押在这条
会消失的前提上。

### KD2. watch = 每轮 context 图像闸门，不是设备闸门

watch 只决定"每轮是否有图"。默认 `false`；打开后常驻采集是允许的。

**否决**：watch off 时关闭设备（一度提出的方案，撤回）。设备归属不在 watch 这一层。

### KD3. node 生命周期 = 设备归属；隐私/授权边界在 node 层

一个 node = 一个设备。设备身份在 node 实例化时绑定（配置 / home）；多摄像头 = 多个
node 实例，不需要为"设备指定"另造机制。

隐私与授权边界因此上移到 **node 生命周期**，而非 watch 开关：warrant 要门的是
"这个 ghost 是否拥有一个 camera body"，不是"是否 watch"。这也解释了 camera node 的
`authorize()` 为何留空 —— 它在等 warrant 全面确认，且本来就该挂在 node 层。

**已决（2026-09-12）**：设备由 node 生命周期持有 —— open 于 node start，close 于 node stop。
不采用惰性打开：`capture` / `look` 都必须设备已 open 才能生效，惰性只是把同一次 open 推迟，
并引入 open/close 抖动与一个竞态窗口（一次 look 的打开/关闭期间，另一次 capture 撞上同一
设备）。绿灯因此是"这个 ghost 此刻拥有一个 camera body"的诚实信号，与授权边界在 node 层
一致。上游闸门是 probe（见 KD8）：probe 不过 → node 起不来 → 设备不开。

### KD4. capture 保留名字，改为直接返回图像数据

`capture` 改为随命令返回图像，走可记忆通道。旧机制返回字符串，在模型侧不可感知 ——
实际像素只能靠错误的 context 通道到达，且随时可能陈旧。名字保留：未来视觉高阶功能会
在 `capture` 上继续扩展。

### KD5. look 是主动视觉动作，命中即按约定返回 signal

`look` 的语义不是"抓一帧"，而是：模型编程"看什么条件"，条件命中时由身体按约定主动返回
signal —— 对应 MOSS 的 "watch conditions, and signal the Ghost proactively"。视觉因此是
主动传感器，而不是模型轮询。

**本轮范围（2026-09-12）**：`look` 本轮不做。先让 `capture` 正确实现（KD4），`look` 的语义
契约保留为扩展点。

**OPEN（本轮外）**：条件如何表达（命名检测器 vs 模型自写的图像比对代码），以及模型
自写代码在 node 内执行的边界。第一轮只钉住语义契约（capture 返回图像 / look 命中给
signal），比对代码留作扩展点 —— 否则第一轮就要背上 codegen 沙箱，拖住一级化。

### KD6. face 检测的输出只走 topic（设备面），不进模型 context

模型无法对时变坐标（cx / cy）建立稳定语义，坐标对它是噪声。`FaceTopic` 的消费者是
**关联设备**（如数字人眼动），不是模型。typed topic 即使暂无消费者也是协议面，不是死代码。

模型侧若确需要，最多是质的"有人脸在场"布尔，且默认不给。

### KD7. 常驻视觉的代价必须示警

`watch on` 的返回值与 notice 应给出当前每轮图像预算（分辨率 / 字节 / 约多少 token），
让"开启常驻视觉"这个开关自带价签。现状缺口：模型可以在不知道代价的情况下开启每轮一图，
而这是它唯一能感知代价的位置。

### KD8. 探针（NODE.md `check:`）是开箱能力的启动闸口 —— 已落地，node 未接

`node-lifecycle` 的 probe 已实现：spawn 唯一咽喉（`node_manager.spawn_node`）在拉起前跑
`manifest.check`，独立进程、语言无关、只用 exit code；nonzero → stderr 即 broken reason，
且 `matrix_channel.run_node` 已把它转成模型可见的 observation
（`Node probe (check: in NODE.md) failed — refusing to launch. {reason}`）。

因此"不允许开启则通知模型为什么失败"不需要新机制 —— camera 的 NODE.md 缺 `check:` 而已。
要检查三类：

- 依赖：cv2 可导入、ffmpeg 在（枚举 / 解码）。
- 设备：`CAMERA_INDEX` 可打开。
- 策略：环境变量闸（允许 / 禁止）。被策略拒绝时以非零退出并写明原因。

纪律：独立进程、零配合主脚本、只用 exit code（不发明 ready 状态机）；reason 会直接进模型
上下文，所以写"为什么"，不堆异常栈。

运行期失效（如设备被拔）不走 probe，走 channel 短路：`refresh_meta`（或任一 meta 生成回调）
里 raise → `ChannelTree._refresh` 接住 → `failure = "refresh failed: <e>"` → 渲染层
`_make_facade_body` 短路，只发 `<failure>`。**这是 `failure` 的设计语义，不是缺陷** ——
channel 坏了就该只告诉模型坏在哪。

**已决（2026-09-12）**：不做 core 改动。`failure` 本来就是短路机制（渲染层"failure 非空时
短路"是定义）；"主动 failure + 原因"已由 `refresh_meta` raise 提供，无需新增 builder 回调。
**短路是自愈的**：channel 连接不丢，每轮仍重刷 —— 故障清除后 refresh 不再 raise，`failure`
清空，表面自动恢复。相机掉设备 → 走短路（设备没了 capture 也执行不了，留命令面是假象）。

唯一措辞 wart：模型看到的前缀是 `refresh failed: <原因>`，对主动降级不贴切。要么接受，要么
那才是需要动的（很小）核心点。暂接受。

### KD9. 图像尺寸约束做成带默认值的语法糖，不做策略层

依赖问题不成立：`message/contents/images.py` 已 `from PIL import Image`，`pyproject.toml`
已有 `pillow`；`Base64Image.to_pil_image()` 是一等方法。**耦合早已存在于这一层**，所谓
"用 PIL 就耦合模型协议"在现状下不新增任何东西。

边界现状：`llms/pydantic_ai_adapter/conversion.py` 把 Base64Image 解成 bytes → `BinaryContent`
→ 交 pydantic-ai 按 provider 序列化，**全程无尺寸 / 字节上限、无校验、无日志**（注释明说
conversion "never drops content, it only lowers fidelity"）。所以真正的约束边界**目前只存在
于 provider 侧**，MOSS 里没有任何地方表达它。

结论（按"语法糖 + 传参 + 默认值"）：

- `from_pil_image(img, format="JPEG", max_edge=None, quality=None, max_bytes=None)` ——
  `None` = 不变换（不静默改图）。谁要约束谁传参。
- 需要补的是**约束数字本身**：provider 的 max bytes / max edge / token 估算，应作为一个
  声明过的常量 + 注释落在 adapter 旁，而非散在 node 里或由某个 node 猜。这是"必须了解模型
  当前的约束边界"的落点 —— 现在它是未知且未记录的。
- 不做全局策略层。（先前"用 `from_binary` 往协议塞字节预算""消费端统一设上限"两个想法都
  是把机制放重了，撤回。）

### KD10. JPEG 流与帧率：context 不得驱动采集；人类直播是合法消费者

应签发 JPEG 流。现状是 HTTP MJPEG。它的定位是**人类的直播开发流** —— 让人实时看到 ghost
看到的画面（直播、开发调试）。MOSS 另有 `stream` 原语（session：字节流 pub/sub，单一有序
发布、多端接收），是"其他 cell / 设备消费"的原理形态（数字人、显示设备）；出现这类消费者
时再挂，不提前建。

**采集帧率由人类直播流决定**（最苛刻的合法消费者），context 独立抽帧，二者解耦：

- 直播流 10–15fps 才不卡（5fps 卡顿明显，仅"可接受"）；人脸检测喂设备 ~5–15fps。
- context 抽帧 ≤1/轮，与采集率无关 —— **模型不为高采集率付 token**。

硬约束：每采集一帧编码一次 JPEG，viewer 采样这份字节（现状是每次 HTTP 轮询都重编码）。
**context 不得驱动采集率**（它只要每轮一张）。

## Implementation Notes

### camera node 重实现的验收清单（现有缺陷）

- context message 里硬写 `<camera:...>` 地址、`authorize()` 返回仓库路径 —— 违反 channel
  地址纪律（channel 不知自身地址，禁止把 CTML 标签 / 仓库路径写进 model-facing 字符串）。
- tier 误用：`notice` 塞了实时状态（应为"能做什么"），且与 context 里的状态消息重复。
- 陈旧帧无 ts / age 标注，以"当前画面"身份进 context。
- watch off 不释放设备；`set_camera` 无视 watch 状态开设备；`run_loop` 在设备掉线后静默空转
  却仍报 `watch:on`；`_last_error` 粘滞不清。
- 事件循环阻塞：人脸检测、`cv2.VideoCapture` open、ffmpeg 枚举都在协程里同步执行；
  `run_loop` 的 grab 与命令的 grab 对同一 capture 并发读。
- 滚动缓存无消费者（只有 `[-1]` 被读），却是按帧数封顶的常驻内存。
- 配置不落 `matrix.home`。
- viewer 每次 HTTP 轮询都重新编码 JPEG，与采集 fps 无关。

### 本轮收敛（2026-09-12）

已定：

- 滚动缓存删（无消费者，唯一被读的是 `[-1]`，纯常驻内存死重）。
- `WATCH_ON_START` 删（契约 3 默认 OFF，开机即采与之冲突）。
- config = node 级 env（dotenv 原生加载 `.env` + NODE.md `exec.env`）；7 项均无隐私面，
  **不需要 node home 配置文件**（授权是唯一隐私面，落在 warrant 存储，不进 env）。
- context message 瘦身：单条，去 `__camera_face__` / `__camera_status__` 冗余（几何输出本
  就不进模型，见 KD6；状态是 notice / status 的事）。
- 运行期降级走短路（KD8），自愈靠每轮重刷。

待用户拍板：

- argument 拆分：CAMERA_INDEX（设备身份）、VIEWER_PORT（本地绑定）是否入参，env 兜底默认。
  原则：argument = "我是谁 / 绑在哪"，env = "我怎么行为"。
- export 到文件：形态（`capture` 参数 vs 独立命令）与路径边界（只允许 tempfile / matrix.home
  之下，realpath 后校验包含关系）。export 的定位不只是 dev 快照 —— 它是 vision 与后续程序
  之间的**交接物**（image → file → downstream program），是文件协议出口，因此路径安全边界
  更要紧（交接物的受控落点）。
- `list_cameras()` 走 ffmpeg 枚举会短暂开设备（灯闪）—— node 启动即持有设备，枚举宜改为
  启动时一次性，而非按需。

## Open Questions

- KD9：provider 的图片约束数字（max bytes / max edge / token 估算）从哪来、记在哪。
- KD10：采集 fps 默认值（直播流 10–15fps 的卡顿权衡）+ context 抽帧策略。
- argument 拆分：CAMERA_INDEX / VIEWER_PORT 是否入参（env 兜底默认）。
- export 到文件：形态与路径边界。
- CTML `matrix:run` 是否需要参数通道（设备选择）。
- KD5：`look` 的条件表达与模型自写代码的执行边界（本轮外）。

## Out of Scope（不在本 feature，另立）

- **文件通道读图片**（image file → `Base64Image` 消息）：归属文件能力，不是 vision。现状
  `file_editor` 的 `view` 文本-only 且 `_is_binary` 直接拒绝二进制，读 .jpg 会报错 —— 但转换
  机制已在消息层（`Base64Image.from_file` + `sniff_media_type`），缺的只是 file 通道接上它。
  闭环的不对称在于：vision 只有"生产"（capture / export），没有"消费"（read image file）；
  消费那一半是 file 的职责。另立 workstream。
