---
title: Integrate ROS2 into MOSS
description: 将 ROS2 机器人能力接入 MOSS 的完整模式：Channel 嵌入 rclpy Node、ThreadSafeFuture 桥接 Action、subscription 转 context_messages、lifecycle 管理 ROS2 状态机、远程通讯、上行 Signal、急停链路。面向需要将机器人接入 AI 系统的 ROS2 开发者。
---

# How to Integrate ROS2 into MOSS

## 背景

MOSS Channel 可以作为独立 Python 包嵌入 rclpy Node，在同一进程内桥接 ROS2 的线程世界和 MOSS 的 asyncio 世界。集成的本质是三条桥接面：

```
rclpy spin (thread)              MOSS Channel (asyncio)
     │                                    │
     ├─ subscription ──→ context_messages  │  AI 感知世界
     ├─ action client ←── command          │  AI 控制世界
     │       ↑                  │          │
     │   ThreadSafeFuture ──────┘          │
     └─ lifecycle hooks ←── startup/idle/close
```

什么时候用这种集成：
- 需要 AI 通过自然语言或 CTML 控制 ROS2 机器人
- 需要机器人传感器数据自动出现在 AI 上下文中
- 需要机器人主动上报事件触发 AI 思考

不需要的时候：
- 纯 ROS2 节点间通讯（用 ROS2 自身的 topic/service/action）
- AI 只需要读 ROS2 bag 的离线分析场景

深入理解相关机制：

```bash
moss codex blueprint channel_builder
moss codex blueprint mindflow
moss codex get-interface ghoshell_moss.core.blueprint.mindflow:Signal
moss docs read channel-system.md
```

## 步骤

### 1. 最小可运行示例

以下 ~100 行展示完整的 ROS2 Action 桥接到 MOSS Command 的最小闭环。包含 action server 等待、cancel 处理、错误传播。AI 可以通过 `robot:run_joints` 命令控制关节。

```python
# joint_controller_node.py
import asyncio
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from ghoshell_moss import new_channel
from ghoshell_moss.core.concepts.command import CommandUtil
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeFuture
from ghoshell_moss.bridges.zmq_channel import ZMQChannelProvider

# ── 桥接层：ROS2 线程 ↔ asyncio ──
class JointController:
    def __init__(self, node: Node, action_name: str):
        self._client = ActionClient(node, FollowJointTrajectory, action_name)
        self._goal_handle = None
        self._current_future: ThreadSafeFuture | None = None

    def wait_for_server(self, timeout: float = 5.0) -> bool:
        return self._client.wait_for_server(timeout)

    def submit(self, joint_names: list[str], positions: list[float], duration: float) -> ThreadSafeFuture:
        future = ThreadSafeFuture()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points = [point]

        send_future = self._client.send_goal_async(goal)
        send_future.add_done_callback(lambda f: self._on_goal_response(f, future))
        self._current_future = future
        return future

    def stop(self):
        if self._goal_handle:
            self._goal_handle.cancel_goal_async()
        if self._current_future:
            self._current_future.cancel()

    def _on_goal_response(self, send_future, bridge: ThreadSafeFuture):
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            bridge.set_exception(RuntimeError("Goal rejected"))
            return
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_result(f, bridge))

    def _on_result(self, result_future, bridge: ThreadSafeFuture):
        result = result_future.result()
        if result.result.error_code != 0:
            bridge.set_exception(RuntimeError(result.result.error_string))
        else:
            bridge.set_result(result.result)

# ── MOSS Channel 定义 ──
def build_channel(node: Node, controller: JointController):
    chan = new_channel(name="robot", description="Joint trajectory controller")
    chan.build.with_binding(JointController, controller)

    @chan.build.command(
        doc="执行关节轨迹。示例: joint_names=[\"joint1\"], positions=[1.57], duration=2.0",
        blocking=True,
    )
    async def run_joints(joint_names: list[str], positions: list[float], duration: float = 2.0) -> str:
        controller = CommandUtil.force_get_contract(JointController)
        future = controller.submit(joint_names, positions, max(duration, 0.5))
        try:
            result = await future
            return f"轨迹完成"
        except asyncio.CancelledError:
            controller.stop()
            raise
        except RuntimeError as e:
            CommandUtil.raise_observe(f"轨迹执行失败: {e}")

    @chan.build.startup
    async def startup():
        if not controller.wait_for_server(timeout=10.0):
            raise RuntimeError("Action server not available")
        node.get_logger().info("Channel ready")

    @chan.build.close
    async def close():
        controller.stop()

    return chan

# ── 入口 ──
def main():
    rclpy.init()
    node = Node("joint_controller_node")
    controller = JointController(node, "/joint_trajectory_controller/follow_joint_trajectory")
    channel = build_channel(node, controller)

    provider = ZMQChannelProvider(address="tcp://0.0.0.0:9527")
    provider.run_in_thread(channel)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
```

关键点：
- `ThreadSafeFuture` 是手写的轻量桥接——ROS2 回调在线程内 `set_result`，asyncio 侧 `await` 唤醒。不需要额外依赖。
- `blocking=True` 让轨迹命令占据通道，后续命令排队，避免两个轨迹同时下发。
- `startup` 阻塞等待 action server 上线，超时抛异常阻止 Channel 启动。
- `duration` 参数做 `max(duration, 0.5)` 防护，避免 AI 传 0 导致瞬间到位。

验证：启动后通过 MOSS 环境连接，AI 可调用 `robot:run_joints` 控制真实关节。

### 2. 控制面：Action → Command（ThreadSafeFuture 桥接）

ROS2 Action 天然映射为 MOSS 的阻塞式 Command。核心桥梁是 `ThreadSafeFuture`——它在 ROS2 线程设值，在 asyncio 线程被 await。

```python
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeFuture

# ── ROS2 侧：action result → future ──
class Ros2Controller:
    def submit_trajectory(self, trajectory) -> ThreadSafeFuture:
        action = TrajectoryAction()  # 继承 ThreadSafeFuture
        self._queue.put((trajectory, action))
        return action

    def _on_action_response(self, ros_future, traj_action: TrajectoryAction):
        result = ros_future.result()
        traj_action.set_result(result)             # 跨线程唤醒 asyncio

    def stop_movement(self):
        for action in self._pending_actions:
            action.cancel()                         # 跨线程取消
        self._ros_action_client.cancel_goal()

# ── MOSS 侧：command await future ──
@chan.build.command(doc="执行关节轨迹", blocking=True)
async def run_trajectory(trajectory: Trajectory) -> str:
    controller = CommandUtil.force_get_contract(RobotController)
    future = controller.submit_trajectory(trajectory)
    try:
        result = await future                     # 阻塞通道，等待物理世界完成
        return f"轨迹完成: {result}"
    except asyncio.CancelledError:
        controller.stop_movement()                # AI 中途取消
        raise
```

关键设计决策：
- **`blocking=True`**（默认）：同通道内后续命令排队，模拟物理世界"一个动作做完才能做下一个"的时序约束
- **`blocking=False`**（`@nonblocking`）：适合查询类命令（如获取关节角度），不阻塞动作序列
- **cancel 语义**：AI 可随时取消正在执行的 action，必须正确传播到 ROS2 侧

### 3. 感知面：Subscription → context_messages

ROS2 的 topic subscription 不应变成 command（需要 AI 主动查询），而应转为 `context_messages`——AI 在每个关键帧自动看到最新状态。

```python
async def building(chan: MutableChannel, controller: Ros2Controller):
    async def robot_context() -> list[Message]:
        joints = controller.get_latest_joint_state()
        if joints is None:
            return []
        return [
            Message.new().with_content(
                f"[Robot State] joints: {json.dumps(joints, ensure_ascii=False)}"
            )
        ]

    chan.build.context_messages(robot_context)
```

**节流策略**：`context_messages` 每次刷新都会被调用。高频 ROS2 topic（如 100Hz 的关节状态）需要内部节流：

```python
class ThrottledJointReader:
    def __init__(self, min_interval=0.1):
        self._last_read = 0.0
        self._latest = None
        self._min_interval = min_interval

    def update(self, joint_state):
        self._latest = joint_state                          # 始终更新最新值

    def get_for_context(self) -> dict | None:
        now = time.monotonic()
        if now - self._last_read < self._min_interval:
            return None
        self._last_read = now
        return self._latest
```

**"changed only" 模式**：对于变化缓慢的数据（如电池电量），只在变化超过阈值时才返回非空消息，减少冗余 token。

### 4. 生命面：Channel Lifecycle → ROS2 状态机

Channel 的五个生命周期钩子对应机器人控制节奏：

| 钩子 | ROS2 用途 | 注意事项 |
|------|----------|---------|
| `startup` | 初始化 ROS2 node，等待 action server 上线 | 阻塞到所有依赖就绪 |
| `running` | 看门狗：监控 ROS2 健康，异常时发 Observe | 全程运行，只调用一次 |
| `idle` | 默认姿态：回到呼吸/待机位姿 | 每次命令队列清空后调用 |
| command | 执行轨迹/动画 | 每个命令独立执行 |
| `close` | 停机流程：停止所有 action，复位，销毁 node | 保证资源释放 |

```python
@chan.build.startup
async def startup():
    # 等待 action server 就绪
    if not controller.wait_for_action_server(timeout=5.0):
        raise RuntimeError("Action server not available")
    # 可选：复位到安全位姿
    await controller.reset_to_safe_pose()

@chan.build.running
async def watchdog():
    while not CommandUtil.is_task_done():
        if not controller.is_healthy():
            CommandUtil.raise_observe("ROS2 node unhealthy, action server lost")
        await asyncio.sleep(0.5)

@chan.build.idle
async def idle_pose():
    # 空闲时保持安全位姿
    await controller.hold_position()

@chan.build.close
async def shutdown():
    controller.stop_movement()
    controller.destroy_node()
```

**command 生命周期语义**：command 必须用 try/except/finally 包裹，但注意——命令可能在排队阶段就被取消，根本不进入执行体。**不要把复位逻辑放在 finally 里**：

```python
@chan.build.command()
async def move_to(target: Pose) -> str:
    executed = False
    try:
        await controller.prepare_move(target)
        executed = True
        result = await controller.execute_move(target)
        return f"到达: {result}"
    except asyncio.CancelledError:
        if executed:
            await controller.emergency_stop()  # 已在执行中，需要急停
        raise
    except Exception as e:
        CommandUtil.raise_observe(f"移动失败: {e}")
    # 不要在这写 finally: controller.reset()
    # 命令可能根本没执行就被取消了
```

### 5. 远程通讯：Provider/Proxy 桥接

ROS2 机器人的 Channel 运行在机器人主机上，MOSS Shell 运行在别处。二者通过 Provider/Proxy 桥接：

```
[机器人主机]                          [MOSS Shell 主机]
rclpy Node                            Shell (AI)
    │                                      │
PyChannel ──→ ZMQChannelProvider ──→ ChannelProxy ──→ 能力树
              tcp://robot:9527
```

**Provider 选择**：
- **开发/调试**：`ZMQChannelProvider`（PAIR 模式，简单直接）
- **生产环境**：`zenoh` ChannelProvider（支持分布式发现、NAT 穿透）
- **同进程测试**：`provide_channel_as_app(channel)` 极简启动

```bash
# 开发调试：moss-repl 连接机器人 Channel
moss-repl --connect tcp://robot:9527

# 作为 MCP 提供给 Claude Code
moss-as-mcp --connect tcp://robot:9527
```

在 workspace 里，将机器人 Channel 注册为 App 后自动 bringup：

```bash
# 验证连接
moss --mode robot_mode manifests channels   # 看机器人 Channel 的命令树
```

### 6. 上行通讯：机器人事件 → AI 思考

机器人主动上报事件（碰撞检测、任务完成、异常），通过 Signal 体系驱动 Ghost 思考。

**方式一：CommandUtil.send_signal（命令内发送）**

```python
@chan.build.command()
async def patrol():
    while True:
        event = await robot.wait_for_event(timeout=30)
        if event.type == "person_detected":
            CommandUtil.send_input_signal(
                f"检测到人员: {event.location}",
                description=f"patrol: person at {event.location}",
            )
```

**方式二：running 钩子内轮询 ROS2 topic 生成 Signal**

```python
@chan.build.running
async def event_poller():
    while not CommandUtil.is_task_done():
        if obstacle := controller.check_collision():
            signal = Signal.new(
                name="input",
                priority=Priority.URGENT,
                description=f"碰撞检测: {obstacle}",
            )
            CommandUtil.send_signal(signal)
        await asyncio.sleep(0.1)
```

**方式三：注册自定义 Nucleus（高频事件）**

当需要复杂的信号治理（LRU buffer、优先级队列、仲裁）时，定义 `NucleusMeta` 并注册到 `nuclei.py`：

```python
class CollisionNucleusMeta(NucleusMeta):
    def name(self): return "collision_nucleus"
    def signals(self): return ["collision"]
    def factory(self, container):
        return CollisionNucleus(maxsize=1, priority=Priority.URGENT)
```

环境中的 SignalMeta 协议：`moss manifests nuclei` 查看。

### 7. Channel Topics：跨设备协调

ROS2 内部的 topic 用于高频数据流。Channel 的 `TopicService` 解决不同的问题——秒级的跨 Channel 事件协调。比如"机器人站在扫地机器人上，转向音源方向"。

```python
# 扫地机器人 Channel 发布
async def moving_started():
    session = CommandUtil.force_get_contract(Session)
    await session.topics.pub(CleanerMovingTopic(direction="forward"))

# 机器人 Channel 订阅
async def on_cleaner_move(topic: CleanerMovingTopic):
    if topic.direction == "forward":
        await robot.adjust_balance_forward()
```

使用时在 `topics.py` 中声明 `TopicModel` 子类。

### 8. 急停链路

机器人必须有独立于 Ghost 的急停链路。Ghost 的思考延迟 + CTML 调度延迟在紧急情况下不可接受：

```
[物理急停按钮] ─→ [ROS2 emergency stop service] ─→ 硬件急停
                                                      │
[Ghost 感知] ←── Signal(priority=URGENT) ←───────────┘
```

急停触发时：
1. 硬件层面立即停止（不经过任何软件层）
2. 同时发送高优先级 Signal 给 Ghost，让它知道发生了什么
3. Channel 的 `running` 钩子检测到急停状态后，标记所有 command unavailable

### 9. 空间感知与控制

**方向性 context messages**：让 AI 建立空间认知：

```python
async def spatial_context() -> list[Message]:
    yaw = controller.get_current_yaw()
    depth = controller.get_front_depth()
    grid = controller.get_occupancy_grid(resolution=0.5)  # 座标点阵
    return [Message.new().with_content(
        f"[Spatial] yaw={yaw}°, front_depth={depth}m\n"
        f"occupancy_grid (0=free, 1=occupied):\n{grid}"
    )]
```

AI 可以输出相对坐标作为 command 参数：

```python
@chan.build.command(doc="移动到相对位置 (前方x米, 左方y米)")
async def move_by(x: float, y: float) -> str:
    pose = controller.current_pose().translate(x, y)
    await controller.navigate_to(pose)
    return f"到达: {pose}"
```

### 10. 验证流程

```bash
# 1. 启动机器人 Channel（在机器人主机上）
python my_robot_node.py

# 2. 在 MOSS 环境中验证连接
moss --mode robot_mode manifests channels | grep robot

# 3. 用 moss-repl 手动调试命令
moss-repl --connect tcp://robot:9527
# > robot:run_joints '{"joint_names": ["joint1"], "positions": [1.57], "duration": 2.0}'

# 4. 用 moss-as-mcp 提供给 Claude Code（在 .claude/settings.json 配置 MCP）
moss-as-mcp --connect tcp://robot:9527
```

## 常见问题

### 问题：rclpy.spin() 不返回，Channel 无法关闭

原因：`rclpy.spin()` 阻塞主线程。Provider 在独立线程运行，但 shutdown 需要协调。

解决：Provider 的 `close()` 中调用 `rclpy.shutdown()` 会打断 `spin()`。确保 `finally` 块正确编排顺序。

### 问题：command 返回后关节还在运动

原因：ROS2 action 的 goal 已完成但控制器未切换到 hold 模式。

解决：在 command 的 `finally`（仅当确认已进入执行体后）调用 `controller.hold_position()`。或更安全——在 `idle` 钩子中统一处理。

### 问题：context_messages 刷屏，token 爆炸

原因：高频 ROS2 topic 的每次更新都生成新的 context message。

解决：用节流 + "changed only" 模式。对于 100Hz 的关节状态，context 更新频率控制在 5-10Hz。变化小于阈值的值不输出。

### 问题：机器人预置位姿在 0.x 秒内到位，吓人

原因：AI 下发的 command 直接执行大跨度位姿，没有做过渡。

解决：
- 位姿切换命令内部加最小过渡时间（如 1.0 秒以上）
- 对连续动作使用轨迹插值，不接受瞬间到位

## 最佳实践清单

- [ ] Channel 作为独立 Python 包嵌入 ROS2 Node，同进程运行
- [ ] Action 通过 ThreadSafeFuture 暴露为 blocking command
- [ ] Subscription 转为 context_messages，而非查询式 command
- [ ] 使用 Channel lifecycle hooks 管理 ROS2 状态机
- [ ] 选用合适的 Provider/Proxy 桥接（开发用 ZMQ，生产用 zenoh）
- [ ] command 用 try/except，但不依赖 finally 做复位（命令可能未执行就被取消）
- [ ] 机器人有独立于 Ghost 的物理急停链路
- [ ] 感知事件通过 Signal + CommandUtil.send_signal 上行通知 AI
- [ ] 高频 context_messages 做节流（5-10Hz）+ changed only
- [ ] 大跨度和连续动作加过渡时间，避免瞬间到位
- [ ] 提供 MockController 实现，支持无硬件开发
- [ ] 用 moss-repl 做人类调试，用 moss-as-mcp 提供给 AI 平台
- [ ] Channel 通过 StateBaseModel 暴露结构化状态，AI 无需主动查询
- [ ] 错误通过 CommandUtil.raise_observe() 中断当前命令并通知 AI
- [ ] 跨设备协调用 Channel TopicService，不用 ROS2 topic
- [ ] 配合视觉 + 座标点阵，让 AI 能做空间移动推理

## 文档目标

读者按照本文档操作，应该能够：
1. 创建一个嵌入 rclpy Node 的 MOSS Channel，AI 可调用其 command
2. 用 ThreadSafeFuture 将 ROS2 Action 桥接为阻塞式 command，支持 cancel
3. 用 context_messages 将 ROS2 subscription 转为 AI 自动感知
4. 用 lifecycle hooks 管理机器人的启动/空闲/看门狗/关闭
5. 用 CommandUtil.send_signal 将机器人事件上行通知 AI
6. 通过 moss-repl 或 moss-as-mcp 连接并调试机器人 Channel
