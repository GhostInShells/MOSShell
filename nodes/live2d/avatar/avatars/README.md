# avatars/ — 形象套件契约

一套形象 = `avatars/<name>/` 一个自包含目录：

```
avatars/<name>/
├── AVATAR.md      可选：角色人设 + per-group instruction 覆盖 + idle 配置
├── channel.py     可选：这个形象的命令面（模型是唯一的作者）
├── animations.py  可选：动画轨迹（纯代码，async def 一条一命令）
└── model/         模型资产（不入库，见 INSTALL.md），必须有 *.model3.json
```

**没有 `channel.py`** → 驱动从模型的 `cdi3.json` / `model3.json` 自动映射一份示范命令面
（`src/avatar_node/mapper.py`）。自动映射只是兜底示范，不是主路径——真想给一个形象设计
好用的命令面，写 `channel.py`。

`AVATAR.md` 是 frontmatter markdown（同 NODE.md 惯例），描述"它是谁、该怎么说话、idle
怎么跑"：

```yaml
---
name: Hiyori
description: 开朗活泼的少女，动作轻快，说话带点撒娇   # 一句冷人设
voice: 可爱女生                                      # 推荐音色 (TTS tone 名)
groups:                                              # 可选：覆盖某 group 子 channel 的 instruction
  face:
    instruction: "头部角度 angle_x/angle_y/angle_z，cheek 脸红"
idle:                                                # 可选：待机配置
  delay: 3.0                                         # 空闲多久才进待机 (秒)
  loop: { group: Idle, index: 0 }                    # 全身待机循环动作组
  parts: { blink: false, breath: true }              # 部件级 idle (SDK 原生)：眨眼/呼吸
---
正文 = 更完整人设，不进 instruction，留给 channel.py 作者/人类
```

缺字段一律默认值；`groups.<g>.instruction` 缺省用自动冷描述。模型包自带眨眼的（动作曲线
驱动眼开闭），应把 `idle.parts.blink` 设为 `false`，避免和 SDK 默认眨眼双重闪烁。

## animations.py — 动画轨迹编程

`animations.py` 是给模型"写代码编排动作"的出口：每个 `async def` 函数是一条动画轨迹，
自动编译反射成主 channel 上的一条命令。函数签名即接口，函数体是纯 Python：

```python
# avatars/<name>/animations.py
async def wave():
    """打招呼: 抬左手挥一挥."""
    avatar = get_avatar()          # 已注入, 无需 import
    await avatar.play("Tap@Body", 0)
    await asyncio.sleep(0.2)       # asyncio 已注入
    avatar.param("ParamArmLA", 0.8, manual=True)
```

- 注入 `get_avatar() -> Avatar` 与 `asyncio`，无需 import。
- `await avatar.play(...)` / `avatar.param(...)` / `await asyncio.sleep(...)` 是积木。
- 每条动画命令 `blocking=True`：它的 await 序列就是时间轨迹，占主轨。
- 编辑后调 `reload_animations` 热更新（同名 module 覆盖旧命令；编译失败保留上一版）。

## channel.py 的入口

驱动发现套件后，若存在 `channel.py`，会调用它唯一要求的入口：

```python
# avatars/<name>/channel.py
from ghoshell_moss.core.blueprint.states_channel import new_prime_channel

async def build(avatar):            # avatar: avatar_node.Avatar
    chan = new_prime_channel(name="hiyori", description="...")

    @chan.build.command()
    async def look(x: float = 0.0, y: float = 0.0) -> None:
        """让眼睛转向 (x, y)，取值 -1 到 1"""
        avatar.params({"ParamEyeBallX": x, "ParamEyeBallY": y})

    return chan
```

`build` 必须返回 `PrimeChannel`（用 `new_prime_channel` 构建），因为驱动要 `with_module`
挂动画轨迹模块。

`build` 返回的 channel 就是这个形象的膜，直接挂进 Matrix。命令签名即接口，不用再写
清单（channel_builder 的红线：不在 instruction 里复述命令）。

## Avatar 事件面（形象作者只用这一个对象）

驱动把"我要它做什么"事件化封装在 `Avatar` 上，作者不需要碰 WS、页面、渲染：

| 方法 | 语义 |
|---|---|
| `avatar.param(id, value, manual=False)` | 推一个参数到某值。`manual=True` 表示模型显式命令，若目标是唇形参数会关掉自动唇动（模型输出优先） |
| `avatar.params({id: value, ...})` | 批量推参数 |
| `avatar.motion(group, index=0)` | fire-and-forget 播一个动作（不占时、不自动复原） |
| `await avatar.play(group, index=0, hold=0.0)` | 占时前景动作：播一个动作，占满时长（默认 `Meta.Duration`，`hold` 覆盖），结束复原 |
| `avatar.expression(name)` | 切一个表情 |
| `avatar.reset()` | 表情/参数回默认，并恢复自动唇动 |
| `avatar.set_backdrop(url)` | 换背板（通用，与模型包解耦） |
| `avatar.set_idle_loop(group, index=0)` | 设定全身待机循环动作；传 None 回自动 |
| `avatar.stop_motion()` / `avatar.clear_motion()` | 停动作 / 停动作+参数回默认再重下发状态 |
| `avatar.idle_loop()` / `await avatar.run_idle_manager()` | 待机循环原语 / 待机仲裁循环（跑在 `build.running`） |
| `avatar.set_lip_sync(on)` | 开关自动唇动（默认自动，模型一般不用碰） |
| `avatar.zoom(factor)` / `avatar.move(x, y)` | 缩放 / 平移（视口层，非模型参数） |
| `avatar.spec` | `ModelSpec`：`.groups`、`.motions`、`.expressions`、`.lip_sync`、`.eye_blink`、`.motion_duration()` |
| `avatar.persona` | `Persona`：`.name`、`.description`、`.voice`、`.group_instructions`、`.idle` |

约束（见 NODE.md）：**command 即真相**——驱动不向页面回读，`avatar.state()` 返回的是
被命令过的参数。

## 默认待机、唇动与交互感知

这三件是框架的默认行为，作者在 `build()` 里可覆盖：

- **默认待机**：`build()` 里调 `avatar.set_idle_loop("Idle", 0)` 即把该动作设为这个形象的
  默认 idle。不设则自动（优先名字含 Idle 的动作组，否则第一个）。启动延时与部件 idle 由
  `AVATAR.md` 的 `idle` 段配置。
- **自动唇动**：驱动默认订阅 `audio/sample`（说侧）驱动唇形。模型显式对唇形参数下
  命令（`avatar.param(..., manual=True)`）即关闭自动唇动，`set_lip_sync(True)` 恢复。
- **交互感知面**：人类点击/拖拽形象会被记录进 `avatar.interactions()`（有界 buffer），
  notice 自动 tail 给模型。点击还会触发 Tap 反馈动作（模型有 `Tap*` 动作组才播）。
  其余状态不进模型 context，按需用命令查。

## 命名与发现

- 套件名 = 目录名。启动用 `--avatar <name>` 选定，换形象 = 重启 node。
- 参数命令名由 `lexicon.param_ident` 从参数 id（必要时从 cdi3 的 Name）归一，共享词表
  见 `src/avatar_node/lexicon.py`——它描述 Cubism 参数词汇本身，不是 per-model 配置。
