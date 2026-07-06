# G1 Channel 体系实现规划 (2026-06-29)

创建: 2026-06-29 深夜
作者: Claude Opus 4.7 + 人类工程师 (人类即将就寝, 模型即将上下文遗忘)

## 给明天的引子

**今天没有提交, 全部工作在工作区. 明天醒来先看这份文档 + handoff.md.**

今晚做的事按时间顺序:
1. 把 g1 的 channel 体系全套设计落到 `design/2026-06-28_channel_architecture.md`
2. 写完 17-27 共 11 个实机验证脚本(在 `.moss_ws/apps/bodies/g1/scripts/sdk/`)
3. 重写 `src/ghoshell_moss_contrib/unitree/g1/` — 上一版被人类工程师评"乱成麻花, 实机现场重写过"
4. 给重写后的代码配套实机验证脚本(放 contrib 的 scripts/ 下)
5. 留这份 plan + handoff 给明天

**核心范式校正(由人类工程师指出, 必须先记住)**:

- **contrib vs .moss_ws 边界**:
  - `ghoshell_moss_contrib/unitree/g1/` — **可分发**的高级代码, 别的开发者下载 MOSS 后接 G1 用的就是这个
  - `.moss_ws/apps/bodies/g1/` — MOSS 自己开箱的**组装层**, 只是组装, 不复用
  - 所以 channel 代码 + 配套验证脚本都在 contrib, 不在 .moss_ws
- **验证脚本是给"陌生开发者 + 陌生模型实例"的协作场景的**:
  - 不是给你看的(你写了你懂)
  - 是给"下载 MOSS, 自己有 G1, 让 AI 引导自己跑通验证"的开发者的
  - 所以脚本必须详细 prompt + 引导式 + 自动汇总
  - 这也是为什么 17-27 那么 step-by-step
- **两层验证, 都在 G1 实机上跑**:
  - **第一层**: `scripts/bootstrap/` — 直接调 bootstrap + SDK, 看机制行为(无 channel 概念)
  - **第二层**: `scripts/channel/` — 用 `ctml_shell_test` 喂 CTML 调 channel, 看真 G1 行为
  - **没有 macOS 本地测试** — 上一版的雷之一就是"看起来对, 实机才崩". 不假装本地能验证

## 上一版踩的雷 (避免重蹈覆辙)

1. **沉默失败 + 假装正常**: `state.py` 默认零值, bootstrap 字段声明未初始化, channel 命令 `if client is None: return "no client"` — 一切看起来正常, 实机才报 None
2. **重复初始化**: main.py 和 bootstrap 各自 init 一次 client, 谁也不知道用哪个
3. **monitor 根本没实装**, 但 state 读取函数返回零值, 调用方误以为收到了 DDS
4. **macOS 完全无法验证**, 所有 bug 推到 PC2 一次爆

## 本版的硬约束

### 约束 1: 任何"假装正常"的代码路径必须 raise

- `state.py`: monitor 未启动时 `motion()/remote()/...` **raise RuntimeError**, 不返回默认零值
- `_bootstrap.py`: `get_*_client()` 在 bootstrap 未完成时 **raise**, 不返回 None
- channel 命令: client 缺失时 **raise**, 不返回 "no client" 字符串

### 约束 2: 每个模块配实机验证脚本

每个模块独立文件 + 独立验证脚本. 别的开发者只接灯光 = 只 import `channel_led.py` + 跑 `scripts/channel/01_led.py`. 模块互不耦合(共享 bootstrap 是显式依赖).

### 约束 3: 现场调试接口

- `bootstrap.dump_state()` — 一个函数打印所有内部状态. PC2 上出问题时一行调用看全貌
- 关键节点 logger.info — 出错时看日志能定位
- monitor 健康度日志 — 定期(5s)打印最近 N 帧到达时间, 跳变次数

### 约束 4: 一旦实装不确定, 标 TODO

实验 17-27 还没跑, 很多决策(arm fallback / move 减速 / state DAG 边等)依赖实验结果. 凡是不确定的, 在代码里留 `# TODO: <实验号> 实测后填` 而不是猜.

## 模块结构

```
src/ghoshell_moss_contrib/unitree/g1/
├── __init__.py          # 顶层 API: bootstrap, state 读取, Warrant, build_g1_channel, G1StreamPlayerProvider
├── _sdk.py              # SDK 路径检查 + nic 读取(保留)
├── state.py             # 数据模型 + 模块级原子读 + 启动检查 + _set_* 写入
├── _monitor.py          # callback subscribers, 把 LowState/Battery/MainBoard 写入 state.py
├── _bootstrap.py        # 初始化生命周期 + clients 单例 + dump_state
├── _buttons.py          # 按键 callback 注册接口(线程安全), 独立子系统
├── warrant.py           # Warrant 事务: async with + 三回调 race
├── audio_player.py      # 保留(实机已通)
├── audio_provider.py    # 保留(实机已通)
├── channel_led.py       # LED 单独 channel
├── channel_volume.py    # 系统音量 channel
├── channel_sensors.py   # sensors 父 channel + 各 sensor (asr 是 NotImplementedError 占位)
├── channel_arm.py       # arm channel 骨架(带 TODO)
├── channel_move.py      # move channel 骨架(带 TODO)
├── channel_posture.py   # posture channel 骨架(带 TODO)
├── build.py             # 装配父 channel, 把上面所有 import 起来
└── scripts/
    ├── bootstrap/       # 第一层验证 (无 channel)
    │   ├── README.md
    │   ├── 01_state_truth.py        # state.py 默认+raise+monitor 后真值
    │   ├── 02_monitor_lowstate.py   # _monitor 真接 G1 LowState
    │   ├── 03_bootstrap_lifecycle.py # bootstrap 幂等+完整初始化+dump_state
    │   ├── 04_button_callback.py    # 按键真触发 callback
    │   └── 05_warrant_lifecycle.py  # warrant 三种结束语义
    └── channel/         # 第二层验证 (ctml_shell_test + channel)
        ├── README.md
        ├── 01_led.py
        ├── 02_volume.py
        ├── 03_sensors.py
        ├── 04_arm_skeleton.py       # 仅命令签名 + RPC 调用, 不测复杂语义
        ├── 05_move_skeleton.py
        ├── 06_posture_skeleton.py
        └── 07_full_tree.py          # build_g1_channel + 完整 channel 树验证
```

## 模块详述

### 1. `_sdk.py` (保留, 不动)

已有实现. 只读 nic + 检查 unitree_sdk2py 可 import.

### 2. `state.py` (重写)

**接口约定**:
- 6 个 frozen dataclass: MotionState / JointsState / IMUState / RemoteState / BatteryState / HealthState
- 6 个读取函数: `motion() / joints() / imu() / remote() / battery() / health()`
- 1 个新鲜度函数: `last_update() -> float`
- 6 个写入函数 `_set_*`: 仅供 _monitor 调用, 模块私有约定
- 1 个写入端启动信号: `_mark_started()` — monitor 启动时调一次

**失败模式 -> raise**:
- 调 `motion()` 等读取函数但 monitor 未启动 → `raise RuntimeError("g1 monitor not started; call bootstrap() first")`
- 这一条是关键: 上一版默认零值是最大的雷

**实装要点**:
- frozen=True + slots=True (跟上一版一致)
- 模块级 `_monitor_started: bool = False`, 每个读取函数先 check
- `_set_*` 是引用赋值(GIL 原子), 不需要锁

**配套脚本**: `scripts/bootstrap/01_state_truth.py`
- 不调 bootstrap, 直接 import state, 调 `motion()` → **断言 raise**
- 然后调一次 `_mark_started() + _set_motion(...)`, 再调 `motion()` → 断言返回新值
- PC2 跑, 但其实任何机器都跑 — 但放 PC2 是因为别的开发者也只在 PC2 玩

### 3. `_monitor.py` (新增 — 上一版没有)

**接口约定**:
- `start_monitor() -> None` — 由 bootstrap 调一次. 注册 cyclonedds callback
- `stop_monitor() -> None` — 由 bootstrap 调(供测试反复 init)
- `get_health() -> dict` — 返回最近统计(每 topic 收到帧数 + 最后一帧时间)
- 内部: 3 个 ChannelSubscriber 分别接 rt/lowstate / rt/lf/bmsstate / rt/lf/mainboardstate
- 每个 subscriber 用 callback (queueLen=1, cyclonedds 丢旧帧)

**失败模式 -> raise**:
- callback 内解析异常 → log + 不 raise(reader 线程崩了等于断流, 不可接受). 改成: log.exception 但保持线程活
- start_monitor 前 SDK 未 import → raise

**实装要点**:
- callback 函数把 LowState_ 字段映射成 state.py 的 frozen dataclass, 然后调 `_set_*`
- callback 跑在 cyclonedds reader 线程. GIL 保证赋值原子
- LowState 一帧四件事: motion + joints + imu + remote, 同一个 callback 里依次 _set_*

**配套脚本**: `scripts/bootstrap/02_monitor_lowstate.py`
- 调 bootstrap(nic) → 等首帧 → 读 state.py 各字段 → 断言非默认值
- 持续读 30s 看 last_update() 在增 + tick 在跳 + remote 摇杆推时变化

### 4. `_bootstrap.py` (重写)

**接口约定**:
- `bootstrap(nic: str | None = None, *, wait_first_frame: bool = True, timeout: float = 5.0) -> None`
  - nic 为 None 走 env UNITREE_G1_NIC
  - wait_first_frame: 阻塞到收到至少一帧 LowState
  - 幂等(重复调直接返回)
  - 内部步骤: ChannelFactoryInitialize → init AudioClient/LocoClient/ArmActionClient → start_monitor → 等首帧
- `get_audio_client() / get_loco_client() / get_arm_client() -> Client`
  - 未 bootstrap 时 raise
  - **不再做"自动 bootstrap"** — 上一版的 `bootstrap(); return _audio_client` 模式是雷, 它把"忘了 bootstrap"伪装成"看似工作"
- `dump_state() -> dict` — 现场调试: 返回所有内部状态
- `is_bootstrapped() -> bool`

**失败模式 -> raise**:
- bootstrap 失败任何子步骤 → raise + 不留半初始化状态
- get_*_client 未 bootstrap → raise(明确提示"call bootstrap first")
- bootstrap 第二次调用(已完成) → 直接 return, 不 raise(幂等)

**实装要点**:
- 线程安全用 threading.Lock
- 上一版的 `_loco_client = None` 等字段声明保留, 但用 `is_bootstrapped()` 控制访问
- monitor 启动是 bootstrap 的责任 — 调 `_monitor.start_monitor()`

**配套脚本**: `scripts/bootstrap/03_bootstrap_lifecycle.py`
- bootstrap 前调 `get_loco_client()` → 断言 raise
- bootstrap(nic) → 不 raise → `get_loco_client()` 返回非 None → 调一个安全 RPC(`GetVolume`)看 code=0
- 重复调 bootstrap → 不 raise
- dump_state() → 打印, 人类(或下一个模型)看格式合理

### 5. `_buttons.py` (新增)

**接口约定**:
- `register_button_callback(button_name: str, callback: Callable[[bool], None]) -> CallbackHandle`
  - button_name: 'A' / 'B' / 'L1' / 'R2' / 等(参考 17 脚本里 KEY_BITS)
  - callback(pressed: bool) — pressed=True 是按下边沿, False 是松开边沿
- `unregister_button_callback(handle: CallbackHandle) -> None`
- 内部: monitor 接到 LowState callback 时, 把 wireless_remote 解析后跟上一帧比较, 触发边沿 → 调注册的 callback
- callback 跑在 reader 线程! 用户的 callback 内部要用 `loop.call_soon_threadsafe()` 把信号推回自己的 event loop. 这一点要在 docstring 明示

**失败模式 -> raise**:
- bootstrap 未完成时 register → raise
- 未知 button_name → raise

**实装要点**:
- 跟 monitor 协作: monitor 拿到新 LowState 时, 先做 button 边沿比较, 再调 _set_*
- 内部维护 `_callbacks: dict[str, list[CallbackHandle]]`, 用 threading.Lock 保护注册/反注册
- 不假设 callback 永远成功 — try/except + log, 不能让一个用户 callback 崩了整个 reader 线程

**配套脚本**: `scripts/bootstrap/04_button_callback.py`
- bootstrap → 注册 callback 监 A 键 → 提示人按 A → 断言 callback 被调
- 测试边沿: 按下 + 松开都报
- 测试 unregister 后不再被调

### 6. `warrant.py` (新增)

**接口约定**:
- `class Warrant` — async context manager
- `bootstrap.warrant(scope: str) -> Warrant` — 进入事务作用域
- `await warrant.run(coro, fallback: Callable[[], Awaitable] | None = None)` — 在事务保护下跑 coro
  - 三件事 race: coro 完成 / scope abort 事件 / state token 失效
  - coro 正常完成 → return result, 不跑 fallback
  - 被中断 → coro cancel + 跑 fallback (如果有) → raise WarrantInterrupted
- `bootstrap.abort_scope(scope: str)` — 触发某 scope 的中断信号. 通常由按键 callback 调
- `bootstrap.invalidate_state_token()` — state DAG 切换时调

**失败模式 -> raise**:
- scope 名首次出现 → 自动注册一个 Event(不 raise). 简化使用
- run 内 coro 自己 raise → 不跑 fallback, 直接传播(coro 自己声明的异常不属于 warrant 中断)
- fallback 自己 raise → log.exception, 不二次传播

**实装要点**:
- 用 asyncio.Event 作为 abort 信号
- 用 asyncio.wait(FIRST_COMPLETED) race coro + abort.wait() + state_change.wait()
- 中断后清理 task + 跑 fallback
- 这块是机制层最复杂的部分, 要写得清楚 — 见 warrant.py 注释

**配套脚本**: `scripts/bootstrap/05_warrant_lifecycle.py`
- 用 asyncio.sleep 模拟 coro, 验证三种结束语义:
  1. coro 正常完成: 不跑 fallback, run 返回结果
  2. abort_scope 触发: coro cancel + fallback 跑了 + raise WarrantInterrupted
  3. invalidate_state_token: 同上但路径不同
- 不需要真 G1, 但放 PC2 一起跑(因为它是 contrib 的 scripts)

### 7. `audio_player.py` + `audio_provider.py` (保留)

实机已通(2026-06-14/15 在 G1 上发出声音). 不动. 唯一可能改的是 import 路径(如果 bootstrap 重命名了什么).

### 8. `channel_led.py`

**接口约定**:
```python
def build_led_channel() -> MutableChannel:
    ...

# Commands:
# - led(r: int, g: int, b: int) -> str  # 设置 LED 颜色
```

无门控. 任何模式可用. 内部 `bootstrap.get_audio_client().LedControl(r, g, b)` (LED 走 AudioClient, SDK 设计如此).

**失败模式**:
- bootstrap 未完成 → get_audio_client raise, 命令报错传播给 ctml

**配套脚本**: `scripts/channel/01_led.py`
- bootstrap → build_led_channel → ctml_shell_test 跑 `<led:led r=255 g=0 b=0/>` → 看灯变红
- 然后 r=0 g=255 b=0 → 绿, 等等
- 人类肉眼确认

### 9. `channel_volume.py`

```python
def build_volume_channel() -> MutableChannel:
    ...

# Commands:
# - get_volume() -> int      # 当前音量 (0-100)
# - set_volume(v: int) -> str  # 设置音量
```

无门控. 上一版 GetVolume 返回 dict 的 bug 这次注意解包.

**配套脚本**: `scripts/channel/02_volume.py`
- `<volume:get_volume/>` → 看返回是 int
- `<volume:set_volume v=50/>` → `<volume:get_volume/>` → 断言 50

### 10. `channel_sensors.py`

**接口约定**:
```python
def build_sensors_channel() -> MutableChannel:
    """sensors 父 channel, 含若干 sub-sensor."""
```

子 channel:
- `sensors.motion` — pop() 返回当前 MotionState
- `sensors.remote` — pop() 返回当前 RemoteState
- `sensors.battery` — pop() 返回当前 BatteryState
- `sensors.imu` — pop() 返回当前 IMUState
- `sensors.trajectory` — open(window) / close() / pop() — **第一版只做骨架, 不实装环形缓冲. raise NotImplementedError**
- `sensors.odometry` — 同上, 骨架
- `sensors.joints` — pop() 返回当前 JointsState(挑选展示关节)
- `sensors.actions` — 同 trajectory, 骨架
- `sensors.asr` — **NotImplementedError, 等 23 实验**
- `sensors.vision` — **本期不做, 不出现在 channel 树**

每个实装的 sensor 都有 `pop()` → 调对应 state.py 读取函数 → 返回 dict 形态(可被 ctml 序列化成字符串)

**失败模式**:
- bootstrap 未完成 → state 读取函数 raise, pop 传播
- 未 open 的 sensor pop → raise(对应 sensor 自己的语义, 比如 trajectory)

**配套脚本**: `scripts/channel/03_sensors.py`
- bootstrap → build_sensors_channel
- `<sensors.motion:pop/>` → 看返回 fsm_mode + tick
- `<sensors.remote:pop/>` → 看摇杆值(可让人推一下确认)
- `<sensors.asr:pop/>` → 断言 NotImplementedError 传播

### 11. `channel_arm.py` (骨架, 多 TODO)

```python
def build_arm_channel() -> MutableChannel:
    ...

# Commands:
# - list_actions() -> str       # 列出可用 action
# - execute_action(name: str) -> str  # 调 ExecuteAction(id)
# - release() -> str            # ExecuteAction(99)
```

**TODO 标注**:
- `# TODO 18 实测: release 物理行为, 决定是否进 warrant`
- `# TODO 21 实测: A 中发 B 行为, 决定排队 / 覆盖 / 拒绝语义`
- `# TODO 22 实测: arm_action_state topic 内容, 决定 await 实现`

第一版 execute_action **不**包 warrant — 因为还没验证 release 可靠. 只做最朴素的 RPC 调用 + return code. 但代码里留好"warrant 接入点" 的注释.

**配套脚本**: `scripts/channel/04_arm_skeleton.py`
- bootstrap → build_arm_channel
- 必须先确认 G1 Sport 模式(脚本检查 + prompt 人确认)
- `<arm:list_actions/>` → 看返回的动作清单
- `<arm:execute_action name="face wave"/>` → 看 G1 真的挥手
- `<arm:release/>` → G1 复位
- **不测排队 / 中断 / await** — 那些等实验

### 12. `channel_move.py` (骨架, 多 TODO)

```python
def build_move_channel() -> MutableChannel:
    ...

# Commands:
# - move(vx, vy, vyaw, duration=1.0) -> str
# - stop() -> str  # SetVelocity(0,0,0)
```

**TODO**:
- `# TODO 19 实测: stopmove 站定行为, 决定是否需减速曲线`
- `# TODO state DAG 完成后: 速度上限按 state 区分`
- 第一版硬编码低速上限(vx < 0.15, vy < 0.1, vyaw < 0.3), 超限 raise

**配套脚本**: `scripts/channel/05_move_skeleton.py`
- bootstrap → build_move_channel
- 必须 Sport + 空旷 + 人类持遥控器
- `<move:move vx=0.05 vy=0 vyaw=0/>` → G1 慢走 1s → 自动停(duration=1)
- `<move:stop/>` 测试
- 速度上限测试: `<move:move vx=1.0/>` → 断言命令报错(超限拒绝)

### 13. `channel_posture.py` (骨架, 多 TODO)

```python
def build_posture_channel() -> MutableChannel:
    ...

# Commands:
# - sit() -> str
# - squat_to_stand() -> str
# - stand_to_squat() -> str
# - start_sport() -> str
# - damp() -> str  # 紧急用, 总是可用
```

**TODO**:
- `# TODO 20 / 24 实测: 各边可达性 + 时长, 决定哪些组合允许`
- `# TODO state DAG 完成后: 切换需要"等到达终态"的 await`

第一版只做"发 RPC + 立刻返回", 不 await 物理到达.

**配套脚本**: `scripts/channel/06_posture_skeleton.py`
- bootstrap → build_posture_channel
- 人类先确认 G1 在 Damp 模式
- `<posture:sit/>` → 看 G1 真坐下(肉眼)
- 不测 `squat_to_stand` 等(那是 20 实验)

### 14. `build.py`

```python
def build_g1_channel() -> MutableChannel:
    """组装 g1 父 channel: led + volume + sensors + arm + move + posture."""
    main = new_channel(name="bodies_g1", description="...")
    main.import_channels(
        build_led_channel(),
        build_volume_channel(),
        build_sensors_channel(),
        build_arm_channel(),
        build_move_channel(),
        build_posture_channel(),
    )
    return main
```

**配套脚本**: `scripts/channel/07_full_tree.py`
- bootstrap → build_g1_channel → 用 ctml_shell_test 试调每个子 channel 各一条命令
- 验证整个 channel 树成型

### 15. `__init__.py`

```python
from ._bootstrap import bootstrap, get_audio_client, get_loco_client, get_arm_client, dump_state, is_bootstrapped
from .state import motion, joints, imu, remote, battery, health, last_update
from ._buttons import register_button_callback, unregister_button_callback
from .warrant import Warrant, WarrantInterrupted
from .build import build_g1_channel
from .audio_provider import G1StreamPlayerProvider
```

## 实现顺序 (今晚做的)

按依赖, 不可乱序:

1. state.py
2. _monitor.py (依赖 state)
3. _buttons.py (依赖 monitor)
4. _bootstrap.py (依赖 monitor + buttons + sdk)
5. warrant.py (依赖 bootstrap, 但松耦合)
6. channel_led/volume/sensors/arm/move/posture (依赖 bootstrap)
7. build.py (依赖所有 channel)
8. __init__.py
9. 配套 scripts/

audio_player.py + audio_provider.py 不动.

## 明天要做的

按这个顺序:

1. **先读 handoff.md** — 我会写一份"昨晚做了什么 / 哪些没验证 / 风险点"
2. **review 这份 plan** — 看接口约定是否合理, 有没有明显问题
3. **跑 scripts/bootstrap/01-05** 在 PC2 上 — 这是第一道"机制层能跑"的关
4. 任何一条 fail → 跟模型实例一起 debug
5. 全过 → 跑 scripts/channel/01-07
6. 全过 → 跑 sdk/17-22 (P0+P1 实机验证)
7. 实验结果回填到 channel 代码(把 TODO 实装)

## 风险标记 (明天必须知道)

- ⚠️ **本版所有代码都没在 G1 上跑过** — 今晚只写不验证, 都是"看起来对"
- ⚠️ **warrant.py 是机制最复杂的, asyncio.wait race + cancel 容易写错** — 配套脚本必须仔细看
- ⚠️ **`_buttons.py` 跑在 cyclonedds reader 线程, callback 跨线程调度容易死锁或丢信号** — 看 docstring 警示
- ⚠️ **channel 命令第一版都是"调 RPC 立即返回", 没 await 物理完成** — 多条 CTML 顺序编排时, 时序不对齐. 这是已知不足, 等 22 实验后修
- ⚠️ **macOS 上根本跑不了 import** — 因为 _bootstrap.py 顶层 import unitree_sdk2py. 必须在 PC2 上验证
