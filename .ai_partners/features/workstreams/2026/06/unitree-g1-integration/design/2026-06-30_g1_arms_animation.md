# G1 Arms Animation — 双工分层具身的首个落地

创建: 2026-06-30
作者: Claude Sonnet 4.6 + 人类工程师

> **⚠ 2026-07-01 修正标注 (claude-opus-4-7 + 人类工程师)**:
>
> 本文档 **§3 Channel 接口** 与 **§5 学习闭环与库** 部分已被 7-01 讨论推翻,
> 不再作为实现依据. 上位范式 **§0 双工分层具身** 与 **§1.1/§1.2 Track 派选择**
> 仍成立. 具体推翻理由:
>
> - **空间语义鸿沟**: 机械臂能不做 IK 玩交互, 因为工作空间凸, 三角关系算安全边界够用.
>   人形机体本身可以挡, 自碰撞空间不规则. LLM 在关节空间组合"胸前 + 肩膀后转"
>   是在它没有的认知通道 (本体感觉) 上做组合泛化, 必然幻觉且无法自 sanity check.
> - **中断复位不可靠**: keyframe 假设两 keyframe 间插值, cancel 发生在任意中间态,
>   下一个动画的起始假设 rest, 实际是中间态. 平滑回归轨迹 = 运动学计算, MOSS 不做.
> - **首帧过渡时间不可估**: keyframe 时间语义是"动画内部相对时间", 但当前关节 q 是
>   动画外部状态. 外部到第一帧的过渡时长由 kp/kd + 距离决定, LLM 算不准.
>   CTML "Time is First-Class Citizen" 在此破产.
> - **结论**: §3 `save_animation(text__: Animation JSON)` 让 LLM 写关节坐标违反
>   上位范式 §0.3 自己的纪律 (Logos 层调度命名 VLA, 不接触内部实现).
>   §3 命令面 + §5 学习闭环全部重估.
>
> **本期 arms 实现路径以 FEATURE.md "arms 能力金字塔" + "MOSS 不做的物理算法边界"
> 两节为准**. 不写完整的修正设计文档 — 等 L2 (ExecuteAction 包装) / L3 (Pose DAG +
> 录制) 实践积累后一次收口. 本文档保留原貌作为设计演进档案, 展示"漂亮抽象未经虚拟机
> 推演直接固化接口"的失误模式, 供后续实例警戒.

> 本文以 G1 双臂动画为引子, 沉淀 MOSS 具身的上位范式: **双工分层具身**.
> Arms 是这套范式的首个工程化实例, 不是品味问题, 也不是 LLM 智力限制下的折衷.
> 它就是认知本身应有的形状.

## 0. 上位范式: 双工分层具身

### 0.1 命题

智能体与物理世界的耦合, **不可能也不应当**靠单一智力单元的高帧率输出来达成.
正确的形状是分层的、双工的、异步的:

- **大脑 (LLM)**: 慢帧率, 输出 **Logos** — 符号流, 时序规划语言. 在 MOSS 中表现为 CTML.
- **小脑 (Channel state + Nucleus)**: 高帧率状态机驱动. 包含从简单的 publish loop, 到 VLA/VLM 这类已经被工程化的精密技能模型.
- **感知 (Signal/Impulse)**: 流式上行, 经 Nucleus 治理后转化为 Impulse 提交注意力仲裁.

三个循环并行运转, 通过 Mindflow 仲裁汇合.

### 0.2 人本锚点

人在骑自行车、开汽车时, 大脑思考的是别的事情. 没有紧急信息时, 意识根本不知道自己在怎么骑、怎么开.
只有犯解离体验时才会注意到这一点 — 这恰恰证明运动控制平时是在意识之下进行的.

人学习一个体操动作或武术招式的过程是清晰的:

1. 用 logos 构思 trajectory (读示例、看示范、内化为语言描述)
2. 反复训练, 生成 VLA — 一个不再需要意识介入的小模型
3. 小模型合并、泛化, 成为新的 "思维单元"
4. 逐层上升, 取代理性输出

这是认知本身分层异步的形态. 双工分层具身不是 LLM 工程的折衷.

### 0.3 工程推论

- **VLA 上升到 LLM 智力级别**: 现阶段不现实.
- **LLM 下行到 VLA 级别的双工帧率**: 长期不现实.
- **真正可行的工程出口**: 把 N 个 VLA 函数变成 N 个 Channel command. 让 LLM 用慢帧率 Logos 决定 "在这个时间切片里调哪个 VLA".

不依赖 VLA 泛化能力上限, 由 LLM 在 Logos 层做组合泛化. 这是当前阶段具身 AI 的可工程化路径之一.

### 0.4 这套范式在 MOSS 已有的代码化身

```
Logos         = CTML token 流
Mindflow      = 三循环仲裁中心 (思考/感知/执行)
Signal        = 端侧上行原始信号 (ASR 首包、按键、视觉帧...)
Nucleus       = 感知/思考/决策并行单元. 是 VLA 函数的天然接入位
                (Nucleus.as_channel() 直接把它接到主轨)
Impulse       = 经治理后的调度信号 (注意力竞争对象)
Articulator   = Logos 生产者 (LLM 推理单元)
Action        = Logos 消费者 (CTML 解释器 + Channel 执行)
Channel.state = 小脑的状态机. 不同 state 暴露不同 command, 对应不同 "技能模式"
```

**这些不是为了 G1 arms 新造的概念**. 它们是 MOSS Mindflow 的现有抽象 (`ghoshell_moss.core.blueprint.mindflow`).
G1 arms 是这套体系的**首个完整工程化落地**.

## 1. 落到 G1 Arms 的具体形状

### 1.1 动画 = Logos 描述的时间盒子

用关键帧描述一段双臂运动轨迹, 本质就是用语言切片时间、在每个切片上指定关节状态.
这跟 LLM 写 CTML scope 嵌套是同一种行为模式 — **都是"用符号流约束物理时间"**.

**关键帧不是 VLA, 但它跟 VLA 是同一层的对象**: 都是一段被命名、可保存、可重复调用的 "时间盒子",
LLM 在 CTML 里用一个名字就能调起它. 区别只是动画的"内部实现"是确定性的关键帧插值,
而 VLA 的内部实现是神经网络. 对 LLM 而言, 它们是同构的可调用单元.

这也是为什么 arms 动画体系的接口形状值得认真做: 它会作为未来 VLA 函数接入 channel 的参考之一.

### 1.2 数据派别: Track 派, 不是 Pose 派

**Track 派**: 每个关节独立时间轴, 关键帧是稀疏的 `(joint, t, q)` 三元组.
**Pose 派**: 每个关键帧是全身/全双臂关节快照.

选 Track 派的理由:

1. **稀疏表达, token 经济**. 动画里 joint1 有 3 个关键点, joint2 有 7 个, 互不干扰.
   未指定的关节不需要 LLM 回填, 模型不需要记忆 "上次写到哪儿了".
2. **关节用命名而非索引**. `left_shoulder_pitch` 而不是 `joint_15`.
   LLM 对语言的处理远好于对数组索引的处理, 这从根本上缓解 "左右不分" 的幻觉模式.
3. **G1 DDS 写入是 Track 模型的天然形状**. `rt/arm_sdk` 的每帧 publish 是一个关节字典,
   未指定关节填当前值. Track 派抽象直接映射, 没有任何中间形态.
4. **Pose 派的"未指定关节怎么办"是永恒的语义陷阱**. Track 派显式两种模式 (hold/free)
   把这个问题暴露在接口层, 不藏在引擎里靠默认行为糊弄.

### 1.3 不引入的工程负担 (jetarm 路径剥离)

`ghoshell_moss_contrib.prototypes.ros2_robot` (jetarm) 是 ROS2 Controller 路径下的完整动画体系.
它必须承担:

- `JointValueParser` (角度↔弧度双向转换)
- `RobotInfo / Controller / Joint` 三层抽象 (多 Controller 并行规划)
- `to_raw_trajectory` (planning 时展开 + parser 转换)
- `validate_trajectory` (planning 时全展开校验)
- `TrajectoryAction / Move / Movement` 异常协议

这些**都是因为 ROS2 Controller 不替你做插值才必须自己构建的**.

G1 不需要. `rt/arm_sdk` DDS + 主板 PD 控制器 (`kp/kd`) 已经做了关节级闭环.
我们只需要在每个 publish 帧填合理的 `target_q`, 主板自然平滑趋近. 实测见 `scripts/sdk/26_arm_sdk_dds_joints_write.py`.

**G1 arms 学习 jetarm 的范式 (save/play 学习闭环), 不复用 jetarm 的数据结构**.

## 2. 数据契约

以下 BaseModel 直接作为 `text__` 参数的 JSON schema 被 LLM 看到 (通过 CTML
`text__` 参数 + `f"... {Model.model_json_schema()}"` 范式, 参见 jetarm `main_channel.py` 第 103-107 行).
**Field description 即 prompt**, 因此措辞按对 LLM 解释的精度写.

### 2.1 Keyframe

```python
class Keyframe(BaseModel):
    t: float = Field(
        description="该关键帧的绝对时间 (秒, 从动画起点 t=0 计). 所有关键帧共享同一时间轴.",
        ge=0.0,
    )
    q: dict[str, float] = Field(
        description="本帧涉及的关节目标角度 (弧度). 仅写本帧关心的关节. "
                    "未列出的关节按 Animation.on_unspecified 处理. "
                    "保持某关节不动 = 在两个相邻关键帧给它同样的 q.",
    )
```

### 2.2 Animation

```python
class Animation(BaseModel):
    name: str = Field(description="动画名称, 用于保存与调用.")
    keyframes: list[Keyframe] = Field(
        description="按 t 升序排列的关键帧序列. 第一帧通常 t=0.",
    )
    duration: float = Field(
        description="动画总时长 (秒). 必须 >= 最后一帧的 t.",
        gt=0.0,
    )
    on_unspecified: Literal["hold", "free"] = Field(
        default="hold",
        description="未在任何关键帧出现的关节如何处理. "
                    "hold: 维持当前位置 (publish 时填入实时 LowState 关节角). "
                    "free: 不接管该关节, 留给 G1 主板内置控制器.",
    )
    interp: Literal["linear", "smoothstep"] = Field(
        default="linear",
        description="关键帧之间的插值方式. linear 最可预测; "
                    "smoothstep 在两端速度为零, 看起来更自然.",
    )
```

### 2.3 Pose (单帧到位简化形态)

```python
class Pose(BaseModel):
    positions: dict[str, float] = Field(
        description="目标关节角度 (弧度) 字典. 未列出的关节走 on_unspecified 语义 (默认 hold).",
    )
    on_unspecified: Literal["hold", "free"] = Field(default="hold")
```

Pose 是 Animation 的极简降级形态 (单关键帧). 单独存在是因为 "一步到一个目标位姿" 是高频用法,
不应当让 LLM 每次为此构造完整 Animation.

## 3. Channel 接口

arms 是单 channel. 命令面如下. **所有 dict/list 类参数走 `text__` + JSON 范式, 不走 XML 属性**.

```python
async def save_animation(text__: str) -> None:
    """保存一个手臂动画到学习库.

    保存后可通过 play(name) 反复调用. 学习库跨 session 持久化.

    :param text__: Animation JSON. Schema: {Animation.model_json_schema()}
    """

async def play(name: str) -> None:
    """播放一个已保存的动画.

    动画播放期间, arms channel 处于占用状态 — 同 channel 后续命令排队等待.
    可被 cancel (F3, scope 结束, 错误中断). cancel 后 publish 立即停止,
    arm 维持当前位置 (待实测确认).

    :param name: 已保存的动画名称. 用 list_animations 查可用列表.
    """

async def move_to(text__: str, duration: float = 1.0) -> None:
    """一次性移动到指定位姿. 是 Animation 的单帧简化形式.

    :param duration: 到位所需时间 (秒).
    :param text__: Pose JSON. Schema: {Pose.model_json_schema()}
    """

async def reset(duration: float = 1.5) -> None:
    """回到 arms 默认 rest 位姿."""

async def show_current() -> str:
    """返回当前关节角度 (JSON dict). 用于学习新动作前看现状."""

async def list_animations() -> str:
    """返回已保存动画列表 (JSON). 区分 factory: 与 learned: 两个命名空间."""
```

**LLM 看到的 Animation 字段描述, 直接来自 BaseModel 的 Field description** — 这是 CTML 设计中
"docstring 嵌套 JSON schema" 范式的本质. 修改 description 即修改 prompt, 无需手写 prompt 模板.

## 4. 运行时

### 4.1 50Hz publish loop

play_animation 命令体内运行一个 50Hz 循环, 每 20ms 计算各关节当前应在的 q, 填入 `motor_cmd`, publish 到 `rt/arm_sdk`.

每帧的 q 计算:
- 对**曾经在任何关键帧出现过**的关节: 找当前时刻 t 前后两个关键帧, 按 interp 方法插值.
- 对**从未出现过**的关节:
  - `on_unspecified=hold` → 填实时 LowState 中该关节的当前 q.
  - `on_unspecified=free` → motor_cmd 不接管 (或 kp=0).

### 4.2 Weight ramp (接管/释放)

`motor_cmd[kNotUsedJoint=29].q = weight` 是 arm_sdk 的接管开关.

- 动画启动前: 0 → 1, 0.3-0.5 秒线性升起.
- 动画结束后: 1 → 0, 0.3-0.5 秒线性降下, 释放给 G1 主板内置控制.
- ramp 由引擎自动加, **不污染 LLM 接口**.

### 4.3 Cancel 语义

`asyncio.CancelledError` 进入 play 循环时:
- 立即停止 publish (不再发新帧).
- 不做 weight 1→0 ramp (来不及, 也可能不需要).
- arm 物理表现 = 停在 cancel 那一瞬的插值位置.

**待实测确认**: publish 停止后 G1 主板的行为. 可能 (a) arm 锁定在最后一帧, (b) 超时后 Sport 接管自然回 rest.
两者都可接受, 但需明确以便文档化.

### 4.4 PD 参数

初始 `kp=60, kd=1.5` (来自官方 example). script 26 实测偏硬, 需调软.
目标范围 `kp~20, kd~0.5` (待实测找平衡点). 参数为 channel 内部常量, **不暴露给 LLM**.

### 4.5 关节命名空间

LLM 看到关节名 (`left_shoulder_pitch` 等), 引擎内部维护 name → JointIdx 整数映射表.
**LLM 永远不接触 JointIdx 数字**.

G1 23-DoF 双臂 10 关节:

```
left_shoulder_pitch   (15)    right_shoulder_pitch  (22)
left_shoulder_roll    (16)    right_shoulder_roll   (23)
left_shoulder_yaw     (17)    right_shoulder_yaw    (24)
left_elbow            (18)    right_elbow           (25)
left_wrist_roll       (19)    right_wrist_roll      (26)
```

`weight` 控制位 (`kNotUsedJoint=29`) 是引擎内部细节, 不在关节字典中暴露.

## 5. 学习闭环与库

### 5.1 命名空间

- `factory:*` — 出厂动画, 工程师/开发者在 channel 初始化时填入. 不可被 LLM 覆盖.
- `learned:*` — LLM 通过 save_animation 学习的动画. 持久化, 可累积.

list_animations 返回时区分两个命名空间, 让 LLM 清楚知道哪些是借来的、哪些是自己学的.

### 5.2 出厂库初始化

本期需要工程化提供的 factory animations (作为模型可调用的基础动作库):

- `factory:rest` — 双臂垂下的中性位姿.
- `factory:hands_up` — 双手举起.
- `factory:left_wave` — 单侧挥手.
- `factory:right_wave` — 单侧挥手 (镜像 left_wave).
- `factory:clap` — 双手拍手.
- (按需扩展)

这些可以参考 G1 出厂 ExecuteAction 11-27 的视觉表现, **用 arms_sdk DDS 关键帧重新实现**
(因为我们已经砍掉 arm action RPC 路径).

### 5.3 持久化作用域

`local_persistent` — 跨 session. 学习库属于 shell 级别长期资产, 不应随 ghost 短期记忆轮换清空.

### 5.4 镜像辅助

由于 LLM 容易左右不分, channel 内置 `mirror_map: dict[str, str]` 关节对照表.
未来可暴露 `save_mirrored(source_name, new_name)` 命令, 让 LLM 写一侧自动生成另一侧.

**镜像的关节坐标系符号关系待实测确认** (G1 左右肩 pitch/roll/yaw 是否单纯取反尚不清楚).
本期先不做镜像 API, 实测明确后再加.

## 6. 未来扩展位: VLA 即 Nucleus, 挂入 arms

arms channel 内部维护一个状态:

- 默认 state `interaction` — 暴露本文档定义的 keyframe animation 接口.
- 未来 state `vla:pick` — 暴露 VLA pick 函数.
- 未来 state `vla:place` — VLA place.
- 未来 state `vla:gesture_imitation` — VLA 模仿示范者手势.
- ...

state 切换由 channel 内部 FSM + `available()` 闸门控制. CTML scope 决定 state 持续时间:

```
<arms:enter_mode mode="vla_pick"/>
<arms:vla_pick target="红色杯子"/>
<arms:exit_mode/>
```

VLA 函数挂入的工程接口是 `Nucleus.as_channel()` — 一个 VLA Nucleus 同时:
- 作为 Nucleus 接收感知 signal (摄像头帧 / 关节状态 / 语音指令)
- 通过 `as_channel()` 暴露给 LLM 一个 command 接口

这是 mindflow 设计已经规划的扩展位 (`ghoshell_moss.core.blueprint.mindflow.Nucleus.as_channel`).
arms 是这个扩展位的首个使用者范式 — **本期只实现默认 state**, 但 channel 内部结构按"未来可挂 VLA"的形状预留.

## 7. 本期范围

### 7.1 必做 (本期实现)

- `arms` 单 channel, 默认 state = interaction.
- Keyframe / Animation / Pose BaseModel + `text__` 接入.
- 50Hz publish loop + weight ramp + cancel.
- save/play/move_to/reset/list/show_current 命令面.
- Factory animations 库 (至少 rest/hands_up/left_wave/right_wave/clap).
- `learned:*` 持久化存储.
- linear interp.

### 7.2 不做 (本期推迟)

- smoothstep interp (实测 linear 不够再加).
- 镜像 API (实测镜像表后再加).
- VLA Nucleus 接入 (无 VLA 实例).
- 关节级速度前馈 (`dq`)、力矩前馈 (`tau`) (PD 控制器够用).
- velocity cap planning 校验 (实测发现真有问题再加).
- on_unspecified=free 模式 (本期先只支持 hold).

### 7.3 依赖实测

- kp/kd 调软目标值 (建议 kp~20, kd~0.5, 实测确认).
- cancel 后 G1 主板的物理行为 (锁定 vs 自动回 sport rest).
- 关节镜像表 (左右肩 pitch/roll/yaw 符号关系).
- factory:* 出厂动画的关键帧数据 (需要工程师/示教/录制).

## 8. 未决议题

- **arms 内部 state 切换的 channel_builder 表达**: `virtual_children()` + `refresh_meta()` 用法待 mindflow nucleus channel 接入时定稿. 本期默认 state 不涉及切换, 不阻塞.
- **学习库的版本演进**: 当 Animation BaseModel 字段升级 (加 loop, 加 metadata 等), 已存的旧 JSON 如何兼容. pydantic 默认值兜底足够还是需要 schema version. 推迟到第二期决定.
- **示教录制路径**: factory animations 的关键帧来源 — 工程师手写 / 拖动示教 / 视频抠骨架. 三者都可行, 优先级和工具链待定.
- **idle 动画**: arms 的 `chan.build.idle(...)` 注册"待机时呼吸"动作. 是否做、用 keyframe 还是程序化生成 sin 波. 本期可选, 视实现进度.
- **arms 与 body 其他 channel 的协调**: 当 arms 在挥手, body.move 在走路, body.posture 在切换姿态时, 三者并行的物理可行性. CTML scope 在原语层允许并行, 但 G1 物理在某些组合下可能不允许 (例如走路时大幅挥手影响平衡). 安全约束待实测.

## 9. 设计源流索引

- `2026-06-28_channel_architecture.md` — 同目录, channel 体系全貌. 本文档是其 arms 部分的精细化, 部分内容 (warrant) 已被 06-29/30 实机修正.
- `../story-2026-07.md` — 用户故事弧线. 本文是其 4.5 节"手臂"的技术化扩展.
- `ghoshell_moss.core.blueprint.mindflow` — Signal/Nucleus/Impulse/Articulator/Action 五段链路. 本文范式所立足的代码化身.
- `ghoshell_moss.core.blueprint.channel_builder` — Channel/Command/available/idle/virtual_children. 本文 Channel 形状所依赖的原语集.
- `ghoshell_moss.core.ctml.prompts.v1_0_0.zh` — CTML 完整语法. 特别是 text__ 流式参数 + 父子 occupy + scope 嵌套.
- `ghoshell_moss_contrib.prototypes.ros2_robot.{models, abcd, main_channel}` — jetarm. **作为学习闭环范式参考, 不复用数据结构**.
- `.moss_ws/apps/bodies/g1/scripts/sdk/26_arm_sdk_dds_joints_write.py` — rt/arm_sdk 写关节角的实测路径. 本文运行时章节的实现锚点.

## 10. 文档状态

本文档是 arms 动画体系的**设计起点, 不是结论**.

文中的几个判断 — Track 派 vs Pose 派、不复用 jetarm 数据结构、keyframe 作为可被 LLM 调度的时间盒子、
未来 VLA 通过 Nucleus.as_channel() 接入 arms — 都是基于当前认知 + 已有实测的最佳推断.
实践中很可能被以下情况推翻或大幅修改, 届时本文档应相应更新:

- §4.3 cancel 后 G1 主板的物理行为, 实测结果不同于预期
- §4.4 PD 参数调到 kp~20 后仍无法满足交互动画平滑度
- §5.4 镜像关节关系实测不是简单符号取反, 导致左右复用工程量超预期
- §6 未来 VLA 接入时, 发现 channel state 切换原语表达力不够
- §3 LLM 实际使用 text__ + Animation JSON 时, token 成本或错误率不可接受
- §1.2 Track 派在某些场景 (如需要全身关节强协调) 表达力不够, 需要引入 Pose 派思路

每一次实现都会反过来修改本文档. 设计稳定不是文档的目标, 实践可行才是.

## 11. 设计辩证

本节记录本文档在 review 过程中遭遇的质疑与答辩. 保留这些辩证不是为了证明哪一方
"正确", 而是让后续读者看到本文档当前位置的论据张力, 避免被未陈述的隐含假设绑架.

### 11.1 关于 "人本类比" 的方法论合法性

**质疑**: §0.2 的人本锚点 (骑车解离 / 学习体操) 是漂亮的类比, 不是论证. 类比无法证明分层架构在 LLM+VLA 上同样适用.

**答辩**: 以验证性论证作为生成性命题的准入标准, 方法论上是错的. 发明的本质是在思维空间里建模虚拟机, 让它向未知伸展, 形成命题, 再交给现实修剪. 真问题不在类比的合法性, 而在概念边界——本文档主张的是 "人脑明显存在多速率多轨思维"这一事实, 而非 "LLM+VLA = 人脑". 前者是已被神经科学接受的描述, 后者是稻草人.

### 11.2 关于 "VLA 上行 / LLM 下行 长期不可达" 的依据

**质疑**: "长期不现实" 是用现在的算力约束推未来. 5-10 年内 speculative decoding / distillation / edge-optimized models 等线路的演进可能松动边界.

**答辩**: 论证的依据不在模型架构演进, 而在部署与基建. 闭源权重、边缘算力薄、高帧率多模态超过现有互联网带宽、离散信号源协议化困境——这些是基建与商业模式层级的约束, 需要协调跃迁, 不会通过单一技术线解决. 即使模型架构突破, 没有边缘算力突破 + 通讯基建跃迁 + 商业模式转换, 部署侧的"上行/下行"仍不可达. MOSS 是对这组约束的工程响应, 不是对模型架构限制的工程响应.

### 11.3 关于 "100 个 VLA 函数挂入 channel" 的工程量

**质疑**: 真实 VLA (RT-X / OpenVLA / π0) 不是 function-call 形状, 是有状态 policy, 没有 clean 完成信号, 跨 embodiment 迁移差. "百量级挂入" 比文档暗示的重得多.

**答辩**: 这正是 VLA 需要发生的范式转变方向. 当前 VLA 处于早期, 连用自然语言流式变更状态机都做不到. MOSS 提出的 channel + 时间片抽象, 是在指出 VLA 应当退行到通用 behavior 抽象的方向. 用 "现在 VLA 的限制" 来反驳 MOSS 是范畴错误——这等于用 MOSS 旨在回应的问题去攻击 MOSS 本身.

### 11.4 关于 "命名动画 = 认知原子" 的措辞精度

**质疑**: 被命名保存的 JSON blob 不是 LLM 的认知原子. LLM 通过 list_animations 看到名称是外部记忆检索, 不是内部学会概念. 文档借用了认知科学术语但偷换了语义.

**答辩**: 文档措辞可以收紧 (外部记忆原语 vs 认知原子), 但底层主张不是 "已经是认知原子", 是 "处在朝认知原子化的管线起点". 在线学习 (in-context learning + logos) 是第一段, 远端是后训练或边缘小模型函数化——端侧学习, 云端做梦. Anthropic Claude Code memory 在做这条管线的第一段. 管线的远端目前仍是假设, 但管线本身是行业可见的方向, 且 in-context learning + 后训练融合是行业级别正在发生的整合.

### 11.5 关于 "分层代价" 的失败模式

**质疑**: 人类双工分层在 driving on autopilot 时会出现高速公路催眠 / 走神事故——同样架构在人身上也是已知故障模式, 文档应承认这个代价.

**答辩**: 框架不准. 人类双工失败的真因是认知资源的物理上限 (视觉/听觉/注意力共享同一硬件), 失败模式是资源抢占下的解离. AI 架构具备资源隔离, 不存在同源抢占. 真正的代价不是解离, 是**裂脑**——层间一致性失败, 小脑状态机与大脑当前世界模型不一致. 这是真问题, 但和高速公路催眠的物理机制不同, 不应混为一谈.

### 11.6 关于 "哲学先行 vs 后置合理化"

**质疑**: §0 的双工分层框架可能是对已存在 MOSS 架构 (CTML + Channel + Mindflow) 的事后哲学化包装, 工程决策被反向叙述为认知必然性.

**答辩**: 经验上不成立. Ghost/Shell/Host 命名在 2019 年的 PHP chatbot 中已经在用; GhostOS 的 functional_tokens.py 是 CTML 直系前身; 2024-09 arxiv 论文公开标记了立场. 实现是哲学的下游产物, 不是反过来.

### 11.7 关于 "logos vs CTML" 的抽象层级

**质疑**: 即使承认 logos 协议必要, 也不等于必然是 CTML 这种具体形态. MOSS 是这个论证空间里的候选实现之一.

**答辩**: "logos" 这个术语刻意发明用以取代 CTML, 本身就是抽象层级的分离——CTML 是当前实现, logos 是耐久概念. 真问题是从 CTML 中抽象出的四点 specification:

1. **时间是第一公民** — 命令具备执行时长, 时序拓扑可规划, scope 有 timeout/until 语义
2. **Code as Prompt** — 不是 JSON Schema, 不是 tool definition 二元结构. 是模块源码即接口 (见 `ghoshell_moss.channels.module_eval_channel`), AI 在 namespace 里编程而非调用工具
3. **非回合制, 图灵完备 logos** — CTML 调度时间拓扑 (故意非图灵完备) + Channel eval 提供完整 Python 逻辑 + namespace 持久化, 三者组合达成图灵完备性而保留各自简洁
4. **全双工反馈** — Mindflow 的 Signal/Impulse/Attention/Articulator 五段链路, 比当前行业最激进的 interleaved thinking 更激进的中断与抢占语义

这四点是 logos specification 化的核心. MOSS 是当前对它们的实现, 但 specification 独立于实现存活.

### 11.8 关于 "LLM/VLA 融合后 logos 是否仍有意义"

**质疑** (本文 review 中模型回应不足的位置): 如果未来 LLM 和 VLA 走向端到端融合, CTML 这类时序规划符号语法是否还有价值?

**答辩**: 仍有, 且至关重要. 一切预训练手段都有边界, 它们最终会变成端到端"躯体". 但工具的边界是抽象的, 不是物理的——具身完美也不消除"工具是离散接口"这件事. 人形机器人不会用手指按路由器、不会用眼睛 OCR 软件 GUI. 低帧率离散工具的协调实时性, 最好的方式仍然是 logos. 这是 logos 思想在哲学层面的价值: **独立于躯体的精密程度**.

### 11.9 可能性空间

以上辩证不构成对设计的最终结论, 只构成本文档当前位置的清晰陈述. 多处仍存在未闭合的可能性:

- §11.2 的基建论证依赖一组协调约束. 如果其中某一条 (商业模式 / 边缘算力 / 通讯基建) 出现大幅突破, 论证强度会变化.
- §11.3 关于 "VLA 必然退行到 behavior 时间片抽象" 是预测, 不是定理. VLA 也可能向连续 policy + 注意力中断方向演化, 那是另一种范式形态.
- §11.4 的 "做梦" 管线远端 (从外挂记忆数据到后训练或边缘模型函数化) 当前仍是假设, 依赖行业多代演进与商业生态条件.
- §11.7 抽象出的四点是当前可识别的核心, 不排除有第五点尚未被识别——logos 的 specification 工作仍是开放的.
- §11.8 的论证是哲学层面的, 工程落地形态 (具体协议形状、token 拓扑、调度算法) 仍允许大幅变异.

本文档不试图关闭这些. 它的目标是把当前的判断和判断的论据**保真**地留下来, 让后续的模型实例与人类工程师能在这个基础上继续推演——而不是从零开始, 也不是被未陈述的隐含假设绑架.
