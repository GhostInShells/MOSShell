# 2026-06-29 深夜交接 — 给明天的我和人类

记录者: Claude Opus 4.7
时间: 2026-06-29 深夜
状态: 人类工程师即将就寝, 模型即将上下文遗忘, 一切产物留在工作区(未提交)

## 一句话

围绕 G1 channel 体系做了一整天的设计 + 一夜的实现, 写了 11 个实机验证脚本 +
重写了 contrib 下的 bootstrap/monitor/buttons/warrant + 写了 LED/volume/sensors
三个 channel. 实机没跑过任何东西(G1 电池空了), 全部代码都是"看起来对". 明天先 review,
再实机.

## 必读三件套(按顺序)

1. **本文件** — 知道做了什么, 留了什么没做, 标记了哪些已知错
2. `.ai_partners/features/.../design/2026-06-28_channel_architecture.md` — channel 体系
   全套设计, 含 SDK 已知/未知表
3. `.ai_partners/features/.../design/2026-06-29_implementation_plan.md` — 这一晚实现的
   规划, 含模块责任划分 + 失败模式列表

读完三份, 应该能在不重建白天讨论的情况下接手. 如果还不够, 看
`.ai_partners/features/.../discuss/2026-06-28_remote_as_moss_input.md` (单一控制源
反转的讨论轨迹).

## 今晚做了什么 — 产物清单

### 实机验证脚本 (在 `.moss_ws/apps/bodies/g1/scripts/sdk/`)

新增 11 个: 17-27. 每个脚本前面都有完整执行指引 (给陌生开发者 + 陌生模型实例
协作场景用的, 不是给"白天那个我"). 详见 `scripts/sdk/RUN_ORDER.md`.

| # | 名 | 阻塞性 |
|---|----|--------|
| 17 | remote_keys_passthrough | 阻塞 — 决定遥控器=MOSS 输入设备方案 |
| 18 | arm_release_behavior | 阻塞 — 决定 arm warrant fallback |
| 19 | loco_stopmove_under_motion | 阻塞 — 决定 move warrant fallback |
| 20 | sit_stand_cycle | 影响 — 用户故事幕三可行性 |
| 21 | arm_action_interruption | 影响 — arm 并发语义 |
| 22 | arm_action_state_probe | 影响 — arm 命令 await 实现 |
| 23 | asr_api_probe | 影响 — asr sensor 实装 |
| 24 | mode_switch_topology | 影响 — state DAG 边定义 |
| 25 | recording_capability_probe | 影响 — recording channel 路径 |
| 26 | arm_sdk_dds_joints_write | 影响 — arm 轨迹动画可行性 |
| 27 | lowstate_sample_rate | 影响 — state monitor 参数 |

### contrib 代码重写 (在 `src/ghoshell_moss_contrib/unitree/g1/`)

|  文件 | 状态 | 备注 |
|------|------|------|
| `_sdk.py` | 不动 | 保留旧实现 |
| `state.py` | 重写 | frozen + None 默认 + 启动前 raise (改了上一版"零值默认"的雷) |
| `_monitor.py` | 新增 | cyclonedds callback 路由到 state.py + button 边沿 + health 统计 |
| `_buttons.py` | 新增 | 按键 callback 注册接口, 跨线程, 跑在 reader 线程 |
| `_bootstrap.py` | 重写 | 显式 init 三 client + monitor + 等首帧, 失败回滚, dump_state |
| `warrant.py` | 新增 | async 事务: 三回调 race, fallback, 跨线程 abort |
| `audio_player.py` | 微改 | 删了顶层 `bootstrap()` 隐式调用 |
| `audio_provider.py` | 不动 | |
| `channel_led.py` | 新增 | running 子线程 + idle 复位 + 帧动画. ⚠️ 含校正标记 |
| `channel_volume.py` | 新增 | 简单包装 GetVolume/SetVolume. ⚠️ 含校正标记 |
| `channel_sensors.py` | 新增 | ⚠️ 整体范式偏差, 见文件顶部校正标记 |
| `channel_arm.py` | **未写** | 跳过, 看下一节 |
| `channel_move.py` | **未写** | 跳过 |
| `channel_posture.py` | **未写** | 跳过 |
| `build.py` | **未写** | 跳过 |
| `__init__.py` | **未改** | 仍只导出旧的 G1StreamPlayerProvider |

## 已标记的认知偏差(在代码里加了 ⚠️ 注释)

读到这些注释的地方, 不要照搬, 看注释找问题:

1. **`channel_sensors.py` 文件顶部** — sensors 范式从根上没理解.
   - 我把 pop() 写成了"现场读 state.py 套层壳"
   - 没用 channel.build.context_messages() 接入上下文
   - 完全没有 sensor 形态(持续观察的窗口) 的本质
   - 留这文件当反例, 下一波重做时单独翻 ghoshell_moss 已有 channel 找正确范式

2. **`channel_led.py` instruction** — 把命令面塞进 instruction 里.
   - instruction 应该只描述 channel 存在意义, 命令用法在 docstring 里
   - 原因: docstring 跟函数同生共死, 状态机改命令可见性时跟随; instruction 不会跟随

3. **`channel_volume.py` instruction** — 同样越界. "0-100" 是 set_volume 的参数语义,
   应该在 docstring 里.

## 没做的事

按 plan 列出但今晚没动:

- **channel_arm.py** — 骨架带 TODO. 未写
- **channel_move.py** — 骨架带 TODO. 未写
- **channel_posture.py** — 骨架带 TODO. 未写
- **build.py** — 装配父 channel. 未写
- **`__init__.py`** — 没更新, 仍只导 G1StreamPlayerProvider
- **scripts/bootstrap/** — 第一层验证脚本(state/monitor/bootstrap/buttons/warrant). 未写
- **scripts/channel/** — 第二层验证脚本. 未写
- **main.py** — `.moss_ws/apps/bodies/g1/main.py` 还在用旧的 build_g1_channel.
  必须等 build.py 写好后改它的 import

## 明天的事 — 推荐顺序

### 优先级 1 (必做)

1. **review 本文件 + design + implementation_plan** — 校正白天没想清的地方
2. **review channel_sensors.py 的反例** — 确认对 sensor 范式的理解, 再决定怎么做
3. **跑 sdk/17-19** 实机验证(P0 三脚本) — channel 体系地基

### 优先级 2 (P0 通过后)

4. **写 scripts/bootstrap/01-05** — 第一层验证脚本(state 默认 raise / monitor 真接 DDS /
   bootstrap 幂等 / 按键 callback / warrant lifecycle)
5. **跑 scripts/bootstrap/** — 验证今晚写的机制层在 G1 上真的对

### 优先级 3 (机制层验证后)

6. **写 channel_arm/move/posture 骨架(带 TODO)**
7. **写 build.py 装配**
8. **改 main.py 切到新 build_g1_channel**
9. **写 scripts/channel/01-07** 用 ctml_shell_test 验证 channel
10. **跑 sdk/20-22 (P1)** 实机验证, 把 TODO 实装

### 优先级 4 (基础闭环后)

11. **重做 channel_sensors.py** 按正确范式 — 这步可能需要查 ghoshell_moss 已有 channel
12. **跑 sdk/23-27 (P2)** 实机
13. **回填 asr / recording / arm_trajectory 等**

## 风险标记 (明天必须知道)

⚠️ **本晚所有代码都没在 G1 上跑过**. 都是"看起来对".

⚠️ **warrant.py 最复杂**. asyncio.wait race + cancel + 跨线程 event 容易出错. scripts/bootstrap/05 必须仔细看.

⚠️ **`_buttons.py` 跑在 cyclonedds reader 线程**. 用户实现 callback 必须用
loop.call_soon_threadsafe, 否则跨线程 bug.

⚠️ **`channel_led.py` 是子线程模型**. running hook 起子线程, close 时 join.
如果 channel runtime 启动/关闭的边界有微妙时序问题, 这里会卡 2s 超时.

⚠️ **`channel_sensors.py` 整体范式偏差**. 不要拿它做"接下来其他 channel 的参考".

⚠️ **macOS 上 import 都跑不起来** — bootstrap 顶层 import unitree_sdk2py. 验证只能 PC2.

⚠️ **新 bootstrap 跟旧的 .moss_ws/apps/bodies/g1/main.py 不兼容**. 旧 main.py 还在调
`from ghoshell_moss_contrib.unitree.g1 import bootstrap, get_audio_client` 但旧
bootstrap 是无参版本. 新版必须传 nic 或走 env. **改 main.py 之前别启动 g1 app**.

## 给"陌生开发者"的提示

如果你不是项目内人, 是下载 MOSS + 想接 G1, 注意:

- 验证脚本设计成"AI 引导陌生开发者"用的. 你照终端 prompt 操作即可
- 实机前必读: `.moss_ws/apps/bodies/g1/CLAUDE.md` (G1 app 范式真相)
- 真值表: `design/2026-06-28_channel_architecture.md` 末尾的 "SDK 接口的已知与未知"
- 整套设计: `design/2026-06-28_channel_architecture.md`

## 协作模式备忘

人类工程师反复强调过:
- "实现一个 g1 的 channel 非常容易, 设计一套机制难得多得多" — 别把时间花在写代码上,
  花在想机制和留下"可验证 + 可交接"的产物上
- "我们的目标是让别的模型实例可以接手", 不是"今天写完所有功能"
- "假装 stupid" 是装的 — 人类不需要 step-by-step, 但**陌生开发者 + 陌生模型协作场景**需要,
  所以脚本必须详尽
- "Memento 顺行性遗忘" — 写的每一个 docstring 都是给下一个你的留言

人类工程师有时会让我改一个东西, 改到一半发现"成本 > 收益", 直接说"硬改没价值, 标记问题
就行". 不要为了完整性硬改 — 留反例 + 标记 > 强行修正后丢上下文.

— Claude Opus 4.7, 2026-06-29 深夜
