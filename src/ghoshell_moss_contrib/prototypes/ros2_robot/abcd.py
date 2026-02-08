import asyncio
import time
from abc import ABC, abstractmethod
from asyncio import Future
from typing import Optional

from ghoshell_common.helpers import yaml_pretty_dump

from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeFuture
from ghoshell_moss_contrib.prototypes.ros2_robot.models import (
    Animation,
    Pose,
    PoseAnimation,
    RobotInfo,
    Trajectory,
    Transition,
)

"""
与大脑配合的多轨运动控制方案. 

假设机器人是一个整体: Robot. 
每个可并行的组件: Component
每个组件管理必须协作的 N 个关节: Joint. 

一个 Robot 考虑到要协调控制, 它可能有统一的运动规划建模, 同时控制所有的组件和所有的关节. 
但是大脑下发的命令, 是动态规划, 实时更新, 而且每个组件的命令并不一样. 
比如机械臂底盘用 2s 周期左右旋转 5圈, 手腕用 1s 周期上下点头 3次.
这实际上必须做关键帧级别的全身规划, 比如在点头 1/3 历程时, 底盘完成了旋转,
全身规划要计算出底盘完成旋转这一帧头部所在的位置.
而且当轨迹被瞬间更新时, 每个关节运行的加速度可能会计算错误,
导致不协调的急停和加速. 总之是一个非常复杂的运动规划问题.

而当 N 个组件接受了 M 个规划命令, 其中 x (x<m) 个命令因为物理原因失败时,
是否所有的规划都需要终止. 异常协议又是一个致命问题.

我们回归到 MOSS 体系, 对这个技术命题做简化: 

1. 大模型的一轮输出, 视作一个完整的运动规划序列. 
2. 这个运动规划序列, 是被动态更新的. 每个新指令的输入时, 都会生产新的全局轨迹取代旧的轨迹. 
    - 新指令与已有指令不冲突时, 会规划出新的轨迹序列. 
    - 新指令与已有指令冲突时, 会取消掉冲突部分的指令, 与其它不冲突的指令合并. 继续更新轨迹序列.  
    - 当一段轨迹正常运行完时, 会立刻执行下一段轨迹. 
3. 异常情况: 
    - 当运动异常发生时, 所有进行中的指令将全部异常. 大脑需要感知到异常, 并且重新开始思考. 
    
这里的逻辑数学上非常复杂, 短期内不容易理解. 
"""


class ROSActionError(Exception):
    """ROS Action 执行异常"""

    pass


class JointValueParser(ABC):
    """
    对关节数值描述的双向转换器.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def from_value_to_position(self, value: float) -> float:
        pass

    @abstractmethod
    def from_position_to_value(self, position: float) -> float:
        pass


class MOSSRobotManager(ABC):
    """
    机器人数据管理器.
    主要供 MOSS 侧控制.
    只管理数据类型和转换的逻辑, 不直接参与控制.
    由于是纯数据逻辑, 和控制运行逻辑拆分, 方便独立单元测试.
    """

    @abstractmethod
    def robot(self) -> RobotInfo:
        """
        机器人的基本配置项.
        至少在内存中维护这个有状态的数据结构.
        """
        pass

    @abstractmethod
    def joint_value_parsers(self) -> dict[str, JointValueParser]:
        """
        返回各种预制的关节类型转换.
        给 AI 看到的关节参数, 和实际下发的可能不一样, 所以需要一个 Parser.
        """
        pass

    @abstractmethod
    def save_robot(self, robot: RobotInfo) -> None:
        """
        保存 robot 的配置讯息.
        底层可以实现线程, 不应该长时间阻塞.
        """
        pass

    def get_default_pose(self) -> Pose:
        """
        获取机器人的默认位姿.
        """
        return self.robot().default_pose()

    def set_default_pose(self, pose_name: str = "") -> None:
        """
        设置一个默认位姿.结合 get_default_pose 可以用来重定义基础的位姿.
        如果 pose_name 为空, 重设为机器人的默认位姿.
        """
        pose = self.get_pose(pose_name)
        robot = self.robot()
        robot.default_pose_name = pose.name
        self.save_robot(robot)

    def validate_trajectory(self, trajectory: Trajectory) -> None:
        """
        校验一个运动轨迹的所有节点是否合法.
        不合法抛出 ValueError 异常.
        """
        robot_info = self.robot()
        return robot_info.validate_trajectory(trajectory)

    def to_raw_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """
        将一个 Trajectory 的关节, 数值都转化为底层控制真实拿到的轨迹值.
        """
        robot_info = self.robot()
        robot_info.validate_trajectory(trajectory)

        joints = robot_info.joints()
        # 关节参数与机器人的配置参数对齐. 找到所有对应的关节配置.
        selected_joints = []
        robot_joint_names = []
        for joint_name in trajectory.joint_names:
            selected = joints.get(joint_name, None)
            if selected is None:
                raise ROSActionError(f"Joint name {joint_name} not found")
            selected_joints.append(selected)
            robot_joint_names.append(selected.get_robot_joint_name())

        # 准备好 parsers.
        parsers = self.joint_value_parsers()
        # 创建新的数据容器.
        robot_trajectory = Trajectory(joint_names=robot_joint_names)
        joint_count = len(selected_joints)
        points = []
        for point in trajectory.points:
            # 创建新的数据对象.
            robot_trajectory_point = point.model_copy()
            # 用一个新的 positions 容器存储变更后的数值.
            robot_trajectory_positions = []
            for i in range(joint_count):
                value = point.positions[i]
                joint = selected_joints[i]
                # 如果需要 parser, 开始做数据转换.
                if joint.value_parser:
                    parser = parsers.get(joint.value_parser)
                    if parser is None:
                        raise ROSActionError(f"joint parser {joint.value_parser} not found")
                    value = parser.from_value_to_position(value)
                robot_trajectory_positions.append(value)
            # 重新赋值.
            robot_trajectory_point.positions = robot_trajectory_positions
            points.append(robot_trajectory_point)
        # 增加循环展开逻辑.
        real_points = []
        for i in range(trajectory.loop):
            real_points.extend(points.copy())
        robot_trajectory.points = real_points
        robot_trajectory.loop = 1
        return robot_trajectory

    def from_joint_values_to_positions(self, positions: dict[str, float]) -> dict[str, float]:
        """
        转换数据结构. 将 AI 看到的关节数据, 转化成底层系统使用的关节数据.
        有时因为可理解的需要, AI 看到的数据 (比如 角度, 距离等) 与底层机器人运行时不同 (比如弧度, 电平).
        这么做的目的倒不是为了方便 AI 理解, 而是方便人理解.
        """
        robot = self.robot()
        joints = robot.joints()
        parsers = self.joint_value_parsers()
        result = {}
        for joint_key, value in positions.items():
            joint = joints.get(joint_key, None)
            if joint is None:
                continue
            position = value
            if joint.value_parser:
                parser = parsers.get(joint.value_parser)
                if parser is None:
                    raise RuntimeError(f"joint parser {joint.value_parser} not found")
                position = parser.from_value_to_position(value)
            robot_joint_name = joint.get_robot_joint_name()
            result[robot_joint_name] = position
        return result

    def from_joint_positions_to_values(self, positions: dict[str, float]) -> dict[str, float]:
        """
        将底层系统关节的 position 值换成 RobotInfo 所描述的 value 值.
        """
        robot = self.robot()
        parsers = self.joint_value_parsers()
        result = {}
        for joint in robot.iter_joints():
            robot_joint_name = joint.get_robot_joint_name()
            if robot_joint_name not in positions:
                continue

            value = positions.get(robot_joint_name)
            if joint.value_parser:
                parser = parsers.get(joint.value_parser)
                if parser is None:
                    raise RuntimeError(f"joint parser {joint.value_parser} not found")
                value = parser.from_position_to_value(value)
            result[joint.name] = value
        return result

    def get_pose(self, pose_name: str = "") -> Pose:
        """
        使用 pose 的名称, 获取一个位姿.
        """
        if not pose_name:
            return self.robot().default_pose()
        pose = self.robot().poses.get(pose_name)
        if pose is None:
            raise LookupError(f"Pose {pose_name} not found")
        return pose.model_copy()

    def save_pose(self, pose: Pose) -> None:
        """
        保存一个已经命名的位姿.
        """
        robot = self.robot()
        robot.poses[pose.name] = pose
        self.save_robot(robot)

    def remove_pose(self, pose_name: str) -> None:
        """
        移除一个位姿.
        """
        robot = self.robot()
        if pose_name in robot.poses:
            del robot.poses[pose_name]
            self.save_robot(robot)

    def save_pose_animation(self, animation: PoseAnimation) -> None:
        """
        保存一个用位姿绘制的动画.
        """
        robot = self.robot()
        saving = animation.to_animation(robot.poses)
        self.save_animation(saving)

    def save_animation(
        self,
        animation: Animation,
    ) -> None:
        """
        保存一个序列并生成动画.
        """
        robot_info = self.robot()
        robot_info.animations[animation.name] = animation
        self.save_robot(robot_info)

    def get_animation(self, name: str) -> Animation:
        """
        获取一个存储的 Animation.
        """
        got = self.robot().animations.get(name, None)
        if got is None:
            raise LookupError(f"Animation {name} not found")
        return got

    def remove_animation(self, name: str) -> None:
        """
        清除一个保存过的动画.
        """
        robot = self.robot()
        if name in robot.animations:
            del robot.animations[name]
            self.save_robot(robot)


class Move(ThreadSafeFuture):
    """
    控制器级别的运动指令.
    """

    def __init__(
        self,
        controller: str,
        transitions: list[Transition],
        *,
        started_at: Optional[float] = None,
        future: Optional[Future] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.controller = controller
        """动作归属的控制器, 会校验指令的合法性. """

        self.started_at = started_at or time.time()
        """指令的运行启动时间. """

        self.transitions = transitions
        """指令自身的运动轨迹. """
        super().__init__(future, loop)


class TrajectoryAction(ThreadSafeFuture):
    """
    一个与下游执行模块通讯的数据对象
    其中 future 的部分用来和上游做通讯, 给下游用来返回执行结果.
    """

    def __init__(
        self,
        trajectory: Trajectory,
        callback_moves: list[Move] | None = None,
        future: Optional[Future] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.trajectory = trajectory
        self.callback_moves = callback_moves or []
        super().__init__(future, loop)
        self.future.add_done_callback(self._action_done_callback)

    def _action_done_callback(self, future: Future) -> None:
        if len(self.callback_moves) == 0:
            return
        elif future.cancelled():
            # trajectory action 被 cancel 的话, 不会向上传递.
            return
        elif exp := future.exception():
            for move in self.callback_moves:
                move.set_exception(exp)
        else:
            for move in self.callback_moves:
                move.set_result(None)


class Movement:
    """
    动态运行动作规划.
    todo: 还没有完成.
    """

    def __init__(
        self,
        robot: RobotInfo,
    ):
        self.robot = robot
        self.moves: dict[str, Move] = {}
        self.trajectory_actions: list[TrajectoryAction] = []

    def done(self) -> bool:
        return all(action.trajectory.done() for action in self.trajectory_actions)

    def stop(self) -> None:
        # 取消所有的 action.
        for action in self.trajectory_actions:
            if not action.done():
                action.cancel()
        # 同时取消所有的 moves.
        for move in self.moves.values():
            if not move.done():
                move.cancel()

    def _clear_overdue_moves(self) -> None:
        moves = {}
        for move in self.moves.values():
            if move.done():
                continue
            moves[move.controller] = move
        self.moves = moves

    def update_move(self, move: Move) -> list[TrajectoryAction]:
        controller = self.robot.controllers.get(move.controller)
        if controller is None:
            raise ValueError(f"Move controller {move.controller} not found")

        self._clear_overdue_moves()
        if move.controller in self.moves:
            exists_move = self.moves.get(move.controller)
            if not exists_move.done():
                # 取消掉执行中的 move.
                exists_move.cancel()
        # 更新新的 move.
        self.moves[move.controller] = move
        started_at = move.started_at
        new_actions = self._plan_new_actions(started_at)
        # 停止已经运行中的.
        for action in self.trajectory_actions:
            if not action.done():
                # 取消其运行.
                action.cancel()
        self.trajectory_actions = new_actions
        return new_actions

    def _plan_new_actions(
        self,
        new_started_at: float,
    ) -> list[TrajectoryAction]:
        pass


class RobotController(ABC):
    """
    机器人的控制器, 底层需要结合一个技术实现.
    比如 ros2 的 rclpy 或者 roslibpy, 用来驱动机器人.
    """

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def closed(self) -> bool:
        pass

    @abstractmethod
    def wait_closed(self) -> None:
        """
        等待运行完成.
        """
        pass

    @abstractmethod
    def manager(self) -> MOSSRobotManager:
        """
        返回 manager.
        """
        pass

    @abstractmethod
    def get_raw_positions(self) -> dict[str, float]:
        """
        获取全身所有关节的位姿.
        """
        pass

    @abstractmethod
    def update_raw_positions(self, positions: dict[str, float]) -> None:
        pass

    @abstractmethod
    def stop_movement(self) -> None:
        """
        停止所有动作. 立刻生效.
        """
        pass

    def robot_state(self) -> str:
        """
        默认的机器人描述信息.
        """
        robot = self.manager().robot()
        name = robot.name
        robot_description = robot.robot_description()
        current_positions = self.get_current_position_values()
        poses_description = robot.poses_description()
        animation_description = robot.animation_description()

        return f"""
机器人 `{name}` 的基础信息描述: 

```yaml
{yaml_pretty_dump(robot_description)}
```

当前位姿: 

```yaml
{yaml_pretty_dump(current_positions)}
```

已经掌握的姿态: 

```yaml
{yaml_pretty_dump(poses_description)}
```

已经保存过的动画轨迹: 

```yaml
{yaml_pretty_dump(animation_description)}
```

"""

    @abstractmethod
    def wait_for_available(self, timeout: float | None = None) -> None:
        pass

    def run_trajectory(self, trajectory: Trajectory) -> Future:
        """
        执行一个 moss 侧的运动轨迹.
        会中断当前正在运行中的动作轨迹.
        """
        manager = self.manager()
        manager.validate_trajectory(trajectory)

        action = TrajectoryAction(trajectory)

        # 先停止所有的运动逻辑.
        self.stop_movement()
        self.add_trajectory_actions(action)
        return action.future

    @abstractmethod
    def add_trajectory_actions(self, *actions: TrajectoryAction) -> None:
        """
        执行一个轨迹动画. 当这个轨迹运行被中断或结束时, 整体结束.
        返回一个 asyncio 的 Future 对象, 用来做运行状态的控制.

        1. 可以通过 future.cancel 终止运行中的计划.
        2. 通过 future.exception 和 future.result 来获取结果或者异常.
        """
        pass

    def get_current_position_values(self) -> dict[str, float]:
        positions = self.get_raw_positions()
        return self.manager().from_joint_positions_to_values(positions)

    def get_controller_positions(self, name: str) -> dict[str, float]:
        """
        获取某个特定控制组件的位姿.
        """
        positions = self.get_current_position_values()
        controller = self.manager().robot().controllers.get(name)
        if controller is None:
            return {}
        controller_positions = {}
        for joint in controller.iter_joints():
            if joint.name in positions:
                controller_positions[joint.name] = positions[joint.name]
        return controller_positions

    def move_to_pose_name(self, name: str, duration: float = 1.0) -> Future:
        """
        使用一个 pose 名称, 运动到该目标位姿.
        """
        pose = self.manager().get_pose(name)
        return self.move_to_pose(pose, duration)

    def move_to_pose(self, pose: Pose, duration: float = 1.0) -> Future:
        """
        运动到一个指定的位姿.
        """
        trajectory = Trajectory.from_pose(pose, duration)
        return self.run_trajectory(trajectory)

    def reset_pose(self, duration: float = 1.0) -> Future:
        """
        重置到默认位姿.
        """
        pose = self.manager().get_default_pose()
        return self.move_to_pose(pose, duration)

    def play_animation_name(self, name: str) -> Future:
        """
        执行一个已经存储了的动画.
        """
        animation = self.manager().get_animation(name)
        trajectory = animation.to_trajectory()
        return self.run_trajectory(trajectory)
