# Runtime — G1 业务对象层 (L3)

contrib `g1/` 四层中的 L3. 每个文件 / 子包是一个"业务对象"模块, 站在 `sdk/` 之上、`channels/` 之下.

```
g1/
├── sdk/         L1 — SDK 句柄 + 生命周期 + 上行信号. 无业务语义.
├── runtime/     L3 — 这里. 把 sdk 原料组合成可独立调用的能力模块.
├── channels/    L4 — 把 runtime 暴露给 LLM. 状态机、授权、Observe 在此层.
└── providers/   IoC 注入点.
```

**Code as Prompt 纪律**: 本 README 只承载**通用纪律 + 模块索引 + 实机引导 SOP**. 每个模块自身的设计、接口、安全要点、引导步骤、调试钩子全部写在**模块 docstring + 对应脚本 docstring** 里, 不在 README 里重复. 跟着代码走的信息长效, 文档跟代码脱钩就会沉没.

## 通用纪律 (持久有效, 进 runtime 工作前必读)

### 1. 模块 = 单例服务, 不做 OO 封装

每个 runtime 模块是一个 Python 文件 (复杂的用子包). 内部用模块级私有状态 (`_dq` / `_lock` / `_thread` 等下划线开头), 公开只暴露函数. 模块本身是 Python 单例 (import 系统保证), 无需"谁持有实例"的争论.

**反例**: 用 `class FooModule` 包装. G1 上每种能力实例唯一, 不需要继承、多实例、mock 注入. 不为不存在的需求买单.

### 2. 生命周期 = 进程级, 由 main.py 显式启停

`main.py` 在 `sdk.bootstrap()` 之后逐个 `start()` 每个 runtime 模块. channel 层只**使用** runtime, 不启停. channel 销毁不影响 runtime — runtime 跟着进程生死.

```python
# .moss_ws/apps/bodies/g1/main.py (将来的形态)
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
from ghoshell_moss_contrib.unitree.g1.runtime import asr, arms

bootstrap(nic)
asr.start()
arms.start()
# ... build channel + matrix.provide_channel ...
```

理由: channel 是 LLM 接触面, channel 销毁 / 重建是高频事件 (例如 LLM 切模式). runtime 服务跟着 channel 启停 = 频繁震荡, 数据丢失. 解耦后 channel 只读不启停, runtime 跟进程生死, 简单且鲁棒.

模块**生命周期 = 进程级单次启停, 不支持 stop 后再 start**. 如需重启请重启进程. 下游模块不应自加可恢复状态机.

### 3. 标准接口形态

runtime 模块分两类, 接口形态不同:

**A. 上行感知类** (asr / listener / control_pad / motion / imu / arm_joints / vision): 持续收数据, 模型偶尔 drain 看累积. 内部分三种数据形态子范式:

- **累积式 drain** (asr / listener / control_pad): utterance / chunk 序列累积, drain 取走 + forgotten 告知
- **覆盖式 latest snapshot** (motion / imu / arm_joints): 单帧当前态, 直接覆盖, 无累积, 无 forgotten
- **滑动窗口** (vision): 连续帧固定时间窗, refresh 时取走 + 清空, 不告知 forgotten (被挤是覆盖不是丢失)

三种子范式共享下表接口, 差异体现在数据形态和 drain 语义, 详见各模块 docstring.

| 函数 | 语义 |
|------|------|
| `start(*, **opts) -> None` | 启动. 幂等 — 重入直接 return. |
| `stop(timeout: float = 2.0) -> None` | 停止 + join 子线程. 幂等. |
| `is_running() -> bool` | 当前是否运行. |
| `drain() -> Batch` | 拿走累积数据 + 清 buffer. "自上次 drain 后我错过了什么". |
| `peek_latest() -> Result \| None` | 不出栈看最新一条. |
| `register_listener(cb) -> str` / `unregister_listener(handle)` | 新数据回调注册. |
| `health() -> dict` | 暴露内部状态, 供 monitor / channel debug. |

**B. 动作执行类** (locomotion, 未来 arms / led / 各类轨迹): 接 LLM 命令 → 调底层 RPC/DDS → 占用一段物理时间. 无累积数据, 无 listener.

| 函数 | 语义 |
|------|------|
| `start() -> None` | 启动. 幂等. 通常只是 set _running + 状态清零, 无子线程. |
| `stop_runtime(timeout: float = 2.0) -> None` | 停止 + 强制取消 active session + 物理兜底 (e.g. StopMove). 命名避开跟命令面 `stop()` 撞. |
| `is_running() -> bool` | 当前是否运行. |
| `current_command() -> str \| None` | active session 的命令名, 给 channel.idle / debug. |
| `health() -> dict` | 暴露 version / current_command / elapsed 等. |
| **`async <cmd_name>(...) -> str`** | 命令面. 每个命令一个 async 函数, **返回 Observe 文本** (含 reason: duration / preempted_by:X / stopped / exception). |
| `async stop() -> str` | 强停, 独立接口, 不参与互斥队列, 立即生效. |

不需要的语义不实装. 比如 `arms` / `locomotion` 是动作执行器, 无"drain accumulated data"概念, 不需要 drain.

**互斥语义** (动作执行类): 单一 `_current_version` + 抢占切换 + version 化的底层 wrapper. 详见 `locomotion.py` 顶部 docstring.

### 4. 子线程 = daemon, 容错 = log + 保持活

reader / publish loop 等子线程**全部 daemon=True**, 随进程退出而死, 不阻塞 shutdown.

线程主循环用 `try / except Exception: logger.exception(...); time.sleep(0.1)` 包裹, **不允许 raise 出主循环**. 信号断流比异常传播严重得多 —— 一个 reader 死掉 = 整个数据流哑火.

错误累计计数走 `health()` 暴露, 不刷屏 log warning.

### 5. Ring buffer 满 = feature, 不是 bug

数据上行类模块用 `collections.deque(maxlen=N)`. **满了自动挤掉最旧的, 不告警, 不阻塞.** 这跟人类记忆一样, 短期遗忘是常态.

模型需要被告知"我忘了多少" — `Batch` 对象里维护 `forgotten: int` 字段, drain 时跟数据一起返回, drain 后归零. 类似"我刚才没听清几句话"的诚实告知.

### 6. Listener callback 跑在 reader 线程

回调注册接口约定: cb 同步触发在 reader 线程内. **cb 不能阻塞, 只做最轻量的事** (计数 / 转发到 queue / print 等). 复杂逻辑放别处.

理由: 让 listener 在数据到达同一线程触发是最低延迟方案. 当前阶段我们故意避免复杂 listener — 跨线程 / asyncio 推回的设计模式留给 channel 层.

### 7. Logger 命名: `moss.g1.<layer>.<module>`

```python
logger = logging.getLogger("moss.g1.runtime.asr")
```

`logging.Logger` 沿 `.` 向上传播 (`propagate=True` 默认), `moss` 顶层配置的 handler / level / formatter 自动应用. 不需要持久引用 `get_moss_logger()`. `logging.getLogger(name)` 内部维护 dict 单例, 每次调用 O(1), 无性能压力.

### 8. 数据契约: pydantic BaseModel + ulid id

`Result` / `Batch` 用 pydantic BaseModel. **字段必带 `Field(description=...)`** —— BaseModel 字段描述会被 channel 层通过 `model_json_schema()` 灌入 prompt, **measure twice 写 description**, 这就是 prompt.

记录 id 用 `ghoshell_moss.message.unique_id()` (ulid). G1 上行报文一般无原生 id 字段, 由 runtime 自生成保证去重和追溯.

### 9. Helper = 无状态转换函数

runtime 不直接产 `ghoshell_moss.message.Message` —— Message 是 channel 层"何时 / 怎么 / 要不要喂模型"的事. runtime 提供无状态 helper:

```python
def to_xml_text(r: Result) -> str: ...
def batch_to_xml(b: Batch) -> str: ...
def to_message(r: Result) -> Message: ...
def batch_to_message(b: Batch) -> Message: ...
```

channel 拿现成 helper 包装, runtime 自身可独立测试 (不依赖 Message 体系即可单独运行).

### 10. `__init__.py` 保持空 (或仅 re-export 子模块名)

`runtime/__init__.py` **不平铺各模块的函数**. 各模块语义独立, `start` / `stop` / `health` 同名会冲突. 调用方走子模块路径:

```python
from ghoshell_moss_contrib.unitree.g1.runtime import asr
asr.start()
asr.drain()
```

这跟 `sdk/__init__.py` 平铺不同 — `sdk/` 是紧密协作的子系统组 (bootstrap + state + buttons), 平铺无冲突且语义聚合. runtime/ 各模块是平行能力, 必须命名空间隔离.

### 11. 模块间 import 一律绝对路径, 不用 `from ..`

跨模块 (跨子包 / 跨层) 的 import 全部写完整路径:

```python
# OK
from ghoshell_moss_contrib.unitree.g1.sdk import get_audio_client

# 禁止
from ..sdk import get_audio_client
from ...g1.sdk import get_audio_client
```

理由: 相对路径在 IDE 重构 / 跨目录移动时静默失效, 是经验性的长期负债. 绝对路径在 IDE 全文替换、grep、跨仓库迁移时一致可见.

例外: 同包内的私有同级 import (例如 `sdk/_bootstrap.py` 里 `from . import _monitor`) 不强制改 — 那是 Python 包内"私有协作"的惯例, 整个目录一起搬时不会断. 但**新写代码倾向绝对路径**, 不增量欠债.

## 认知误区 (显式偏航)

后续实例进入 runtime 工作前, 必须先建立对以下误区的抗性. 独立列出而不放在通用纪律里, 因为它们是**具体的错误认知**, 不是正面规则.

### drain 直接喂 context_messages — 视场景对错, 别一刀切

**症状表述** (老): "周期性 drain 拿历史轨迹喂进 context_messages, 让模型理解'刚才发生了什么'" (原句在 `_imu_sen_listen_and_drain.py:12`).

**关键区分**: `context_messages` 有两种被调用节律, 分工不同:

- **每帧刷新** (高频, 感知型 channel 默认): 每个 shell tick 都会重装配上下文, drain 一次下一帧 buffer 空, 顺行性遗忘退化成即时遗忘 — 老结论仍成立, **每帧型 context_messages 不能 drain, 只走 peek 只读 (peek_latest / peek_recent_n / snapshot / current_state)**.
- **每回合触发 (signal-driven)**: ghost 回合装配时才读一次, 不是每帧. 这种情况下 drain 语义正确 — buffer 累积到 signal 触发, 一次性交给 ghost, 下回合从空开始. 类似"你按提交键之前, 我一直在累积你说的话". listener 这类"按键 / signal 触发交付"的通道适用此模型, 但**装配路径必须证实是每回合一次不是每帧一次**, 否则退化成即时遗忘.

**判定流程** (新写 channel 时):

1. 我这个 channel 的 context_messages, ghost 每一次感知 tick 都会 rebuild 吗? → 是: 只走 peek. 否: 走 drain 也 OK.
2. 不确定? → 先 peek + tail-N, 稳妥. 未来确认是每回合装配后再改 drain, 成本很低.
3. 想要"平时看历史, 触发时交付整批"的双路径: peek 走 context_messages, drain 走 signal payload — 两条不互斥, 各司其职. listener channel 就是这个组合的样板.

**正确分工汇总**:

- `drain()` 合法宿主 4 类 (第 4 类为本次补充):
  - **listener callback / 定时 task**: 消费后转 `Signal` 推给 mindflow (事件驱动)
  - **一次性事件 Memory 消息**: 消费后在会话历史里落一次, 不再刷新
  - **主动查询 command**: LLM 用一条 command 显式消费, 一次返回
  - **每回合装配一次的 context_messages** (确认非每帧): 累积交付语义, 与 peek 型互补
- `peek_latest()` / `peek_recent_n()` / snapshot / `current_state` / `health`: 每帧型 context_messages 的合法数据源. 需要历史窗口就 runtime 内 ring buffer + 只读 accessor, channel 每帧读最近 N 条, 不消费.

**感染范围** (老): 除病灶起点外, 相关论述搞错分工的位置至少包括 `vision.py` (顶部 docstring), `asr.py` (~L427), `listener.py` (~L133), `control_pad.py` (顶部 docstring + ~L141/L370/L605) 等. 后续接手 runtime 层修订工作的实例应逐个 review 这些位置, 把 drain 与 context_messages 的关系按上述 "每帧 vs 每回合" 二分改写, 不要延续原表述. 修订每一处时同步更新本节的感染范围清单.

## 测试纪律

**测试只走 PC2 实机**. macOS 上 `from .sdk import get_audio_client` 会触发 `unitree_sdk2py` 顶层 import, 跑不通 — 不做 mock, 不维护双轨.

### 验证脚本与模块同目录

每个 runtime 模块的实机验证脚本**放在 runtime/ 内**, 跟模块代码物理相邻. 命名约定:

| 前缀 | 范畴 | 编号 | 用法 |
|------|------|------|------|
| `_<module>_sen_<scenario>.py` | **场景类** — 模拟 channel 真实使用 scenario, 双工体验 | 不编号 | `python -m ghoshell_moss_contrib.unitree.g1.runtime._asr_sen_xxx <nic>` |
| `_<module>_tes_<id>_<name>.py` | **单测类** — 验证模块内某条机制 / 边界 / 契约 | 必须编号 | 同上 |

下划线开头 = 模块私有, `__init__.py` 不导出, 不进 `from X import *`. 但**可被 `python -m` 直接执行**, 这是它的存在形态.

### 场景类脚本设计原则

场景脚本是"模拟 channel 真实使用"的小剧场, 不是步进式实验探针:

- **每个脚本独立完整** — 不要"先跑 sen_001 再跑 sen_002"这种顺序依赖. 那种 step-by-step (sdk/00-27 那批) 是探索性脚本的形态, 在功能测试里很糟糕 — 想测 step_4 要从 step_1 走起.
- **双工体验** — listener 后台持续打印 + 主线程 PromptSession 等人输入. 用 `prompt_toolkit.patch_stdout` 防止打印破坏输入框 (参考 `ghoshell_moss.host.tui` 范式).
- **真实交互** — 让人对着 G1 说话 / 摇杆 / 按按键, 然后看 runtime 模块的真实响应. 不要 mock 数据.
- **干净 finally** — `try/finally` 保证 `module.stop()` 必跑. 实机不清理 = DDS 资源泄漏.

### 单测类脚本设计原则

单测脚本验证"模块该满足的某条契约":

- **每个脚本聚焦一条断言** — start/stop 幂等 / listener 异常隔离 / forgotten 计数正确等
- **编号决定推荐执行顺序** — 不强制顺序, 但低编号通常是高阶测试的前置
- **assert + 明显成功/失败输出** — 退出码 0 = 通过, 非 0 = 失败

### Code as Prompt

每个脚本顶部 docstring 必须能让一个未读过 README 的模型实例**一眼看懂场景 + 用法**:

1. **一句话场景** — 这个脚本模拟什么 / 验证什么
2. **Usage** — `python -m ...` 完整命令 + 参数
3. **前置** — G1 状态要求 (开机 / 任何模式 / Sport 模式 / 有人配合 / ...)
4. **预期** — 跑通时人/模型看到什么 (实时刷新行 / drain summary / health 输出 ...)
5. **安全要点** — 涉及物理动作的模块, 引导用户前必须告知的禁区 (L2+B / L2+R2 / 推摇杆 等)

脚本运行时也是 prompt — print 引导人类下一步做什么 (`"对着 G1 说几句话"` / `"按 Enter drain"` / `"Ctrl+C 退出"`), 不要让人盯着空屏幕猜.

## 实机验证流程 — 模型引导用户的 SOP

当人类用户说 "我想测 <模块>" / "怎么验证 control_pad" 这类话, 你 (模型) 按以下流程引导他们. 这是给后续模型实例的 SOP — 让你不必每次重新发明流程.

### Step 1 — 分清是单测 (tes) 还是场景 (sen)

| 类别 | 文件名 | 是否要 G1 开机 | 是否要人按按键/说话 |
|------|--------|----------------|---------------------|
| 单测 | `_<m>_tes_NNN_*.py` | 否 (只需 PC2 上 import 通) | 否 (内部 hook 注入) |
| 场景 | `_<m>_sen_*.py`     | 是 | 是 |

**强烈建议: 先跑通全部单测, 再开场景**. 单测全 PASS = 模块逻辑无 bug; 场景失败更可能是 G1 物理状态 / 操作时序问题, 不是代码问题.

### Step 2 — 前置环境确认 (单测 + 场景共有)

1. 用户 SSH 到 PC2 (路径见 `.moss_ws/apps/bodies/g1/docs/hardware.md`)
2. cd 到 MOSS 仓库根目录 (PC2 的 git clone 位置)
3. `.venv` 就绪 (没有则 `uv sync`)

### Step 3 — 跑单测 (按编号顺序)

低编号通常是高阶测试的前置, 按顺序跑.

```bash
.venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._<module>_tes_001_xxx
.venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._<module>_tes_002_xxx
# ...
```

- 退出码 0 + stdout `PASS: tes_NNN_xxx` → 通过
- 退出码非 0 + stderr `FAIL: <reason>` → 失败, 让用户复制完整输出反馈给你

### Step 4 — 跑场景

单测全 PASS 后才跑场景:

```bash
.venv/bin/python -m ghoshell_moss_contrib.unitree.g1.runtime._<module>_sen_xxx <nic>
```

- 引导用户做对应物理操作 (按某键 / 说话 / 看 G1 反应)
- **物理操作 + 终端预期输出 + 安全要点, 全部见脚本顶部 docstring** — 那是脚本作者写给执行人的, 你照着引导即可, 不要凭记忆造步骤. 不在本 README 里重复.

### Step 5 — 失败时

- 让用户复制完整 stderr + assert / exception trace 反馈
- **不要凭猜测诊断, 不要让用户改代码**. 流程: 反馈 → 你判断 → 改代码 → 重跑.
- 反复失败考虑 G1 物理状态问题 (mode 不对 / 网络断流 / DDS 没连上 / sdk 没装), 让用户跑 `scripts/sdk/00_import_verify.py` 检验 sdk 链路

### Step 6 — 实测结果落地

实测发现新事实 (字段意义 / 物理行为 / 时序常数), 沉淀到:
- 模块 docstring 里的 "实测发现" / "TODO" 注释 (跟代码近)
- `.ai_partners/features/workstreams/.../FEATURE.md` 的 Session Log + "物理事实" 节
- 本 README "待实测 / 待回填" 节 (索引粒度)

不要散落在临时笔记里.

## 模块清单 (索引)

每个模块的设计 / 接口 / 安全要点 / 引导步骤 / 调试钩子, **全部在模块文件 docstring + 对应脚本 docstring**. 模型引导用户时:

1. 看模块 docstring (设计 / 接口契约 / 物理事实 / 调试钩子)
2. 看 sen / tes 脚本 docstring (场景 / 用法 / 前置 / 预期 / 引导步骤 / 安全要点)
3. 跟用户对话引导

### 上行感知类

| 模块 | 一句话定位 | 关键脚本 |
|------|----------|----------|
| `asr.py` | G1 内置远场 ASR (整句 VAD) | `_asr_sen_listen_and_drain` |
| `listener.py` | 蓝牙耳机近场流式 ASR (与 asr 对称, 需先跑 setup 生成 config) | `_listener_sen_setup` (首次配置), `_listener_sen_dialog` (端到端) |
| `control_pad.py` | 遥控器按键语义层 (binding 精确匹配 + fallthrough + 双层 debounce) | `_control_pad_sen_listen_and_drain`, `_control_pad_tes_001..006` |
| `motion.py` | G1 FSM 模式当前快照 + 切换事件轨迹 (10Hz polling) | `_motion_sen_fsm_transitions` |
| `imu.py` | 机身姿态当前快照 + 2Hz 定时采样 (rpy/gyro/accel, 不存 quat) | `_imu_sen_listen_and_drain` |
| `arm_joints.py` | 双臂 10 关节当前快照 + 2Hz 定时采样 (rad, 跟 arms keyframe 单位对齐) | `_arm_joints_sen_listen_and_drain` |
| `system_info.py` | 电池 + 主板状态 stateless query (无 daemon, 无 ring buffer) | `_system_info_sen_read` |
| `vision.py` | 摄像头滑动窗口式视觉感知 (fps × window 严格 token 预算, 第三变体范式) | `_vision_probe` (硬件路径), `_vision_sen_window` (待实装) |

### 动作执行类

| 模块 | 一句话定位 | 关键脚本 |
|------|----------|----------|
| `locomotion.py` | G1 空间移动 (前后/横移/转身), 七 async 命令面 + version 互斥 + Observe reason | `_locomotion_sen_basic` |
| `led.py` | 眼条 RGB LED 关键帧动画 (三轨道优先级 + 4 easing + 20Hz daemon) | `_led_sen_factory_showcase` |
| `audio.py` | G1 内置 TTS + PCM 流播放 + 音量 (跟 audio_player 角色分工) | `_audio_sen_speak_and_play` |

### 辅助 / 工具

| 文件 | 用途 |
|------|------|
| `audio_player.py` | StreamAudioPlayer ABC 适配器, 服务 `ghoshell_moss.contracts.speech` 流水线 |
| `audio_provider.py` | G1StreamPlayerProvider, mode providers.py 装配用 |
| `story_202607_fsm.py` | FSM 模式枚举 (motion.py 引用; story-2026-07 用户故事的代码化身) |
| `_headphone_buttons_probe.py` | 蓝牙耳机按键事件发现工具 (为后续 `headphone_buttons.py` 模块准备输入) |

### 样例参考 (新模块入门先读这些)

- **上行感知类范式**: `asr.py` (简单, 首选样例) / `control_pad.py` (复杂度: binding + 双层 debounce)
- **动作执行类范式**: `locomotion.py` (version 互斥 + async 命令面 + Observe reason)

新模块先对照这两类选一个, 再决定是"沿用范式"还是"因 X 原因偏离 — 在代码注释明文化".

## 实施过程与复盘 (2026-06-30)

**过程**: 一个母会话 (claude-opus-4-7) 设计四层骨架 + 写 `asr.py` 作为首个样例 + 起 `runtime/README.md` 通用纪律. 多个并行会话照此范式扩展到 12 个 runtime 模块 + 24 个验证脚本. 期间动作执行类作为第二范畴从 `locomotion.py` 自然涌现, 与上行感知类分立. README 一度膨胀到 775 行 (各并行会话往中央文档堆模块细节), 治理回 331 行 + 明文化 Code as Prompt 纪律在 header.

**给后续模型实例的复盘**:

- **不要照抄已有模块** — `asr.py` / `locomotion.py` 是两类范式样例, 不是模板. 新模块进来时, 先确认你的物理事实跟样例是否同构, 再选范式. 当前矩阵里 `forgotten` 字段被全部上行感知类照抄, 但低频模块 (motion 10Hz / imu 2Hz / arm_joints 2Hz / system_info 无 daemon) 永不触发 — 是范式一致性的代价之一. 你接手新模块时, 应根据自身物理事实裁减无意义字段, 而不是 copy-paste.
- **`health()` schema 跨模块未统一** — 各模块独立设计字段名和粒度. 不在本期强行对齐, 等 channel 集成阶段倒推 (channel 真用上才知道什么字段必要).
- **范式偏离要在偏离模块自身的 docstring 明文化** — 例如 `listener.py` 偏离 `asr.py` 的对称范式 (独立 setup probe / `Utterance` ≠ `AsrResult` / 流式 partial 与整句 VAD), 因为耳机近场流式跟 G1 远场整句的物理事实不同. 这类理由必须写在偏离模块的 docstring, 不要让后续模型实例误判成"历史负债待清理".
- **未完成**: `arms` keyframe animation 引擎. 7-01 讨论后设计路径重估 (详见 FEATURE.md "能力路线图" + "arms 能力金字塔" 节), 6-30 设计文档 `2026-06-30_g1_arms_animation.md` §3/§5 命令面部分已被推翻, 上位范式 §0 仍成立. 本期目标降级为 L1 (闲时呼吸), L2 起 (ExecuteAction 包装) 依赖 action state probe 实测. arms 高级形态 (LLM 写动画 / 复杂中断 / 稻草人) 全部需要 "中断三基础" (碰撞反馈/脱力 + 复位 + 首帧过渡) 达成才能开始 — 这不是本期范围.
- **未完成**: `vision` 摄像头感知. 设计 docstring 已定型 (滑动窗口 + fps × window 严格 token 预算, 见 `vision.py`), 实现待 Jetson 硬件路径 5 分钟脚本验证 (7-02 早晨) 后落地. 数值 (默认 fps / window / max 上限) 依赖实测调, 不能在虚拟机里定死.

## 待实测 / 待回填 (打开 README 时的 TODO)

- `asr.py`: DDS subscriber `Close()` 后能否 `Init()` 重启. 不行则 stop() 改成"不关 sub 只停 reader 线程".
- `asr.py`: filter topic `rt/audio_msg/filter` 是否并入 (monitor_asr 双订). 实测后决定.
- `asr.py`: `angle` 字段正值方向 (G1 右侧 or 左侧). 实机说话标定.
- `listener.py`: 蓝牙 HFP 实际采样率 (mac AirPods / PC2 BlueZ 各自报多少). setup probe 拿到后回填到本节.
- `listener.py`: 蓝牙断连 → 重连 → capture 是否能自动恢复 (capture supervisor 设计如此, 实机未验证).
- `listener.py`: `drain(force_finalize_partial=True)` 后, 后续 partial 是否真被丢弃, 新 session 是否干净开 — dialog 脚本可复现验证.
- `listener.py`: 单测体系待补 (force drain session abort / health change 跳变 / config 解析 / forgotten 计数).
- `_headphone_buttons_probe`: 用户跑完后, 把 summary 贴回模型, 据此写 `headphone_buttons.py` (跟 `sdk/_buttons.py` 同范式).
- `motion.py`: Dance (R1+B) / Debug (L2+R2) 等模式的 FSM ID 实测确认, 回填到 `story_202607_fsm.FsmMode` 枚举.
- `imu.py`: roll / pitch 零位 + 正方向坐标系标定 (做一次"向前倾 10°"实测). 标定完去掉 Field description 里的"未校准"警告.
- `imu.py`: yaw 漂移速率实测 (静止放置 5min 看 yaw 变化), 决定 channel 周期 dump 时是否需要给 LLM 警告"yaw 已漂移 N°".
- `imu.py`: 静止折叠阈值 (0.5° yaw / `_DQ_MOVING_THRESHOLD` 0.05 rad·s⁻¹) 是否合理, 实测噪声水平后调整.
- `arm_joints.py`: 10 个手臂关节 rad 零位 + 正方向标定 (一个一个手动摆到极限位看 q 值). 标定完去掉 Field description 里的"未校准"警告.
- `arm_joints.py`: `_HISTORY_DELTA_THRESHOLD` (0.05 rad) 是否合理 — 实测一段 wave 动画看是否过滤掉了应该看见的关节.
- `arm_joints.py`: Sport 模式下手臂自然摇摆/呼吸时, sampler 是否会把空闲噪声当成"在动" — 实测后可能要调 `_DQ_MOVING_THRESHOLD`.
- `vision.py`: Jetson 摄像头硬件路径 (cv2.VideoCapture 直出 / GStreamer V4L2 / CSI nvargus 四种 fallback 哪条成立). 5 分钟脚本验证.
- `vision.py`: cv2.read() 在 Jetson 上单帧耗时 (估 5-15ms). 影响能否吃满 5fps 上限.
- `vision.py`: max_fps / max_window / resolution 合理上限, 跟当前部署 LLM token 预算挂钩. 起点建议 fps=2.0 × window=1.0, 实测调.
- `vision.py`: deque 内 PIL.Image 长期持有的内存开销 (5 帧 × 640×480 RGB ≈ 5MB, 应无压力, 实测确认无泄漏).
- `vision.py`: 摄像头子线程 OpenCV 异常容错策略 (跟 asr 一致 log + sleep + 保持循环).
- `system_info.py`: `last_update_seconds_ago` 实际反映的是 LowState (1052Hz) 健康度, 不是 bmsstate / mainboardstate 本身的新鲜度. 想区分需在 sdk 层 per-topic 记录 last_update, 不在本期范围.
- `system_info.py`: 电池温度阈值 (>50°C 警告) / 主板温度阈值 (>70°C 警告) 是从官方文档抄的, 实机正常范围待实测回填到 Field description.
- `audio.py`: TtsMaker 能否被 PlayStop 中断 (大概率不能 — TTS 是独立通道). 长文本 TTS 中途 `:cancel` 实测验证.
- `audio.py`: speaker_id 不同值 (0/1/2/...) 对应的音色. 实机一次性扫几个数值标定.
- `audio.py`: Volume 范围 — 0-9 (官方文档暗示) 还是 0-100 (_archived 老代码). `:vol N` 扫一遍确认有效区间.
- `audio.py`: TTS 时长估算公式 (当前一刀切 0.15s/char). 用 `:status on` 看 `rem=` 衰减跟耳朵听到的结束时刻偏差, 中英文分别标定.
- `audio.py`: TtsMaker 是否互斥 (RPC 是否在 TTS 未播完时返回 7401/3104). 短间隔连发实测.
- `locomotion.py`: `V_YAW low/medium/high` 全是猜值 (0.3 / 0.6 / 1.0 rad/s). 跑 `tl 5 low` 数实测转角算回真实角速度, 回填模块顶部常量.
- `locomotion.py`: `LocoClient.Move` 是否需要 keepalive 重发 — 现 publish loop 20Hz 每 tick 重发. 实测改 "仅首发" 看 G1 是否还能走完 duration, 决定是否 strip 重发.
- `locomotion.py`: 抢占切换是否平滑 — `f 5` 第 2 秒立刻 `tl 1 medium`, 看 G1 物理是否抽搐. 抽搐说明 G1 主板对 Move 过渡处理跟预期不同, 需另谋方案.
- `locomotion.py`: 强停后 G1 物理状态 — 应立即静止 + 回 stand idle. 如保持最后一帧速度或乱走, StopMove 行为跟想象不同.
- `locomotion.py`: 单测体系待补 (version 抢占切换 / stop 优先级 / finally StopMove 版本校验). 跟 control_pad 的单测体系对齐.
- `led.py`: LedControl 调用率上限实测 — 想把 driver fps 上调到 30/50 看 RPC 是否 timeout/丢帧.
- `led.py`: HSV easing 在 G1 眼条上是否真的比 LINEAR 视觉更柔和 (red→green 中段). 实机不同距离 (1m/3m) 各感受一次.
- `led.py`: breath 默认参数 (period_ms=2000, 余晖 1/16) 视觉自然度, 不自然就调回填默认.
- `led.py`: SDK 抽离后改驱动线程内 `client.LedControl` 为 sdk 新路径 (TODO 已标在代码就近).
- `led.py`: 单测体系待补 — 离线可跑 (start/stop 幂等, 轨道优先级覆盖, 工厂渲染契约), 不依赖 sdk.
- 通用: `health()` 返回是否需要标准化 schema (跨模块统一). 第二个 runtime 模块进来后再决定 → 目前未统一, 等 channel 集成时倒推.
- 通用: motion / imu / arm_joints / system_info 单测体系待补 (start/stop 幂等, ring buffer forgotten, listener 异常隔离 等). 跟 control_pad 的单测体系对齐.
