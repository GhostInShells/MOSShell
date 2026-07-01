# Channels — G1 LLM 接触面 (L4)

contrib `g1/` 四层中的 L4。每个文件 / 子包是一个 channel，站在 `runtime/` 之上，把业务能力暴露给 LLM。

```
g1/
├── sdk/         L1 — SDK 句柄 + 生命周期 + 上行信号. 无业务语义.
├── runtime/     L3 — 业务对象, 可脱离 channel 独立单测.
├── channels/    L4 — 这里. 把 runtime 暴露给 LLM. 状态机、授权、idle、Observe 在此层.
└── providers/   IoC 注入点.
```

**Code as Prompt 纪律**: 本 README 只承载**通用纪律 + 模块索引 + 集成入口说明**。每个 channel 自身的设计、命令面、授权逻辑、idle 行为全部写在**模块 docstring** 里，不在这里重复。

## 通用纪律

### 1. Channel = 进程级单例模块

每个 channel 是一个 Python 文件 (复杂的用子包)，模块顶层直接 `channel = new_channel(...)`, 装饰器挂命令 / lifecycle。参见 §6。

不搞 class wrapper、不搞多实例、不搞依赖注入 — 一种能力对应一个 channel 实例，进程内唯一。

### 2. 相互独立开发，组合在集成层做

各 channel 模块之间**不互相 import**。channel A 不调用 channel B 的函数，不做跨 channel 编排。

理由: runtime 层已经做好了能力原子化。channel 层的职责是"把这个原子暴露给 LLM"，不是"编排多个原子"。

编排 — 哪个 channel 挂在哪棵树、父子关系、启动顺序 — 是集成层 (mode `channels.py` / app `main.py`) 的事，不进 `channels/`。

### 3. 导入 runtime 用绝对路径

```python
from ghoshell_moss_contrib.unitree.g1.runtime import locomotion
from ghoshell_moss_contrib.unitree.g1.runtime import led
```

跟 runtime 层纪律一致，不做相对 import。

### 4. Channel 自包含自身依赖的 runtime 生命周期

每个 channel 在自己的 `@build.startup` 里启动**直接依赖的 runtime**。幂等: runtime.start() 内部自带 `_running` 检查, 重入直接 return, 所以放心调。

```python
@face_led.build.startup
async def _on_startup() -> None:
    led.start()   # 幂等
```

**边界**: 只启自己直接用的 runtime, 不启同伴的。整机级基建 (`sdk.bootstrap()` — DDS + 三 client + monitor) 是**所有 g1 channel 的共同上游**, 不属任何单一 channel, 由集成层 (mode `channels.py`) 在加载路径顶部一次性 bootstrap。

**为什么不集中在集成层启 runtime**: channel 层不自包含 runtime 生命周期 = channel 复用时上游必须知道"这货背后要先启哪个 runtime", 是隐式契约, 会踩雷。self-contained 让 channel 可 drop-in 到任意集成路径 (mode / 独立测试 / repl 直连), 上游只管拓扑不管顺序。

**stop 不写**: runtime 生命周期 = 进程级 (runtime README §2), channel 销毁不该 stop runtime — 别的 channel 可能还在用, 且 runtime 不支持 stop 后再 start。清理由 daemon 线程随进程死。

### 5. Logger: `moss.g1.channels.<name>`

```python
logger = logging.getLogger("moss.g1.channels.led")
```

跟 runtime 层 `moss.g1.runtime.<name>` 对齐。

### 6. Channel 是模块级实例, 不是工厂函数

每个 channel 模块顶层 `channel = new_channel(...)` 生成实例, 装饰器直接挂命令 / lifecycle / instruction. 集成层直接 `from ... import channel` 用。

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel

face_led = new_channel(
    name="face_led",
    description="...",
)
face_led.build.instruction("...")

@face_led.build.command()
async def blink(color: str) -> None: ...

@face_led.build.startup
async def _on_startup() -> None:
    led.start()
```

不做 `new_xxx_channel() -> Channel` 工厂函数 — 一种能力对应一个 channel 实例, 进程内唯一, 工厂签名是为不存在的多实例需求付费。

## 必要前置知识

进入 channels 工作前必读，三份缺一不可，顺序即优先级：

1. **`runtime/README.md`** (本目录同级下层) — L4 从 L3 继承的通用纪律：模块级单例、绝对路径 import、logger 命名 (`moss.g1.<layer>.<module>`)。runtime 生命周期本身是**进程级单次启停**, channel 层在自己的 startup 里启动**直接依赖的** runtime (幂等), 不 stop。不读这份，会重复踩 L3 已经踩过的坑。

2. **`moss --ai codex blueprint channel_builder`** — 契约面。`Channel` / `Builder` / `Command` / `CommandUtil` / lifecycle (`startup` / `idle` / `close` / `running` / `refresh_meta`) / `available` 门控 / `import_channels` 组树 / `with_binding` IoC 注入，全在这。

3. **`moss --ai ctml read`** — 调度语义。channel FIFO occupy 与异通道并行、`text__` / `chunks__` / `ctml__` 三类流式参数、scope `until=flow|all|any`、`@nonblocking`、`Observe` 与 `raise ObserveError` 的中断分层、`__content__` 非标记文本、原语 (`__main__` only)。不懂 CTML，就理解不了 channel 该长什么样 —— channel 的命令面和 idle 面都是 CTML 调度模型的映射。

**走最小知识路径 —— 不要提前旁观其它 channel 的实现**。读完上面三份就动手，**不要**顺手翻 `channels/` 已有模块或其它 contrib，把它们当"参考实现"抄。已有实现常混入历史包袱、场景偏见、临时妥协，会在你形成契约理解之前先污染它。契约的源头在 blueprint + ctml，不在别人的代码里。只有当你已经独立形成设计、需要验证一个具体拿捏 (e.g. "别人怎么处理依赖 runtime 状态的 available") 时，再定点去参考。

## 模块索引

| Channel | 依赖 runtime | 一句话 |
|---------|-------------|--------|
| `g1_root` | `system_info` | 根 channel — 身体自我认知 instruction + vitals (电量/温度) |
| `face_led` | `led` | 面部 LED 灯条 — idle 底色 (solid/breath/set_idle) + 有限表现动画 (fade/blink/pulse/rainbow/police) + 清理 (clear/off), context_messages 上报当前 idle 状态 |
| `locomotion` | `locomotion` | 空间移动 — 前后 / 横移 / 转身 / 强停 7 命令, context_messages 上报 active session (无 active 不打扰) |
| `fsm` (`g1_fsm`) | `story_202607_fsm` | 授权状态三元组 + AI 模式按键规则. change callback → LED/TTS 播报; X 键 → InterruptSignal + `locomotion.stop()` |
| `asr` (`g1_asr`) | `asr` | G1 远场麦克风 ASR 纯感知 — `peek_window(3)` 每帧进 context, 不发命令不发 signal |
| `listener` | `listener` + `headphone_buttons` + `story_202607_fsm` | 蓝牙耳机近场流式 ASR — 无命令面 (开关提交在硬件按键侧), context_messages 走 tail-N 只读. 头戴按键 → toggle 聆听; Y 键 (需 AI 模式) → 自由对话切换; A 键 (需 AI 模式) → 强制 drain + NotifySignal |

新 channel 落地后在表里加一行。

## 集成入口

集成层可以是 mode `channels.py` (workspace 里, e.g. `.moss_ws/src/MOSS/modes/unitree_g1/channels.py`) 或 app `main.py` (e.g. `.moss_ws/apps/bodies/g1/main.py`).

集成层职责有两件:

1. **加载路径顶部一次性 bootstrap 整机基建** (`sdk.bootstrap()`) — 所有 g1 channel 的共同上游, 不属任何单一 channel.
2. **拓扑组装** — import channels, `parent.import_channels(child)`, `main.import_channels(root)`.

```python
# .moss_ws/src/MOSS/modes/unitree_g1/channels.py
from ghoshell_moss_contrib.unitree.g1 import sdk
sdk.bootstrap()  # 整机基建, mode 加载时启动

from ghoshell_moss_contrib.unitree.g1.channels.g1_root import g1_root
from ghoshell_moss_contrib.unitree.g1.channels.face_led import face_led

g1_root.import_channels(face_led)
main.import_channels(g1_root)
```

各 channel 依赖的具体 runtime (led / locomotion / asr / ...) **不在集成层启动**, 由 channel 自己的 startup 承担 (§4).

## 测试纪律

待第一个 channel 落地时建立。预期方向: channel 层单测以"命令契约 + idle 行为 + available 闸门"为主，mock runtime 层接口，不依赖 G1 实机。

## 参考

- `runtime/README.md` — 下层通用纪律。channel 层的许多约定 (logger 命名、绝对路径 import) 直接从 runtime 继承。
- `.ai_partners/features/workstreams/2026/06/unitree-g1-integration/story-2026-07.md` — 用户故事弧线，channel 划分的上位参考。
- `.ai_partners/features/workstreams/2026/06/unitree-g1-integration/design/2026-06-30_g1_arms_animation.md` — arms channel 的设计起点。
