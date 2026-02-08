from collections.abc import Iterable
from typing import Any, Optional

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "Animation",
    "Controller",
    "Joint",
    "Pose",
    "PoseAnimation",
    "PoseTransition",
    "RobotInfo",
    "Trajectory",
    "TrajectoryPoint",
    "Transition",
]


class Joint(BaseModel):
    """
    机器人关节的定义.
    考虑到关节的数值机制各不相同 (角度, 距离等), 需要配合不同的计算方法, 将数值进行转换.

    不是正式的技术实现, 正式版本应该从 ROS2 Control 或者 Moveit 中对齐, 通过模板来获取.
    Constraints 也应该从配置读取.
    """

    name: str = Field(description="joint name")
    origin_name: str = Field(default="", description="底层机器人系统里的关节名称. 和底层系统对齐, 并非必要. ")
    description: str = Field(
        default="",
        description="关节的默认描述, 需要重点解释数值单位. ",
    )
    default_value: float = Field(default=0.0, description="默认的运动参数")
    max_value: float = Field(
        default=0.0,
        description="最大的参数值",
    )
    min_value: float = Field(default=0.0, description="最小的参数值")
    value_parser: str = Field(default="", description="将参数 value 转换成轨迹值 (通常是弧度) 的算法. 空表示默认. ")

    def joint_description(self) -> dict[str, Any]:
        return self.model_dump(exclude={"value_parser", "origin_name"})

    def get_robot_joint_name(self) -> str:
        """
        返回机器人侧的关节命名.
        """
        return self.origin_name or self.name

    def validate_value(self, value: float) -> None:
        if value > self.max_value or value < self.min_value:
            raise ValueError(f"joint {self.name} not allow value {value}")


class Pose(BaseModel):
    """
    位姿的数据结构.
    可以定义出一个自欺人的姿态.
    """

    name: str = Field(default="", description="位姿的 id.")
    description: str = Field(default="", description="关于机器人姿态的文字描述")
    positions: dict[str, float] = Field(
        description="所有关节的目标位置",
    )


class TrajectoryPoint(BaseModel):
    """
    运动轨迹中的一个点.
    """

    positions: list[float] = Field(
        description="所有关节的位点, 值的顺序要严格与 Trajectory 的 JointNames 对齐. "
        "注意这里的值不一定是底层真实的值, 还需要经过换算."
    )

    time_from_start: float = Field(
        description="点运行到起点的时间",
    )

    # 这些暂时不实现.
    # velocities: Optional[List[float]] = Field(
    #     default=None,
    #     description="关节的目标速度（可选），顺序与 positions 一致。"
    # )
    # accelerations: Optional[List[float]] = Field(
    #     default=None,
    #     description="关节的目标加速度（可选），顺序与 positions 一致。"
    # )
    # effort: Optional[List[float]] = Field(
    #     default=None,
    #     description="关节的目标力矩/力（可选），顺序与 positions 一致。"
    # )

    def validate_kinematics(self, joints: list[Joint]) -> None:
        joint_count = len(joints)
        """验证运动学参数的完整性"""
        if self.positions and len(self.positions) != joint_count:
            raise ValueError(f"Positions length {len(self.velocities)} != joint count {joint_count}")

        # if self.velocities and len(self.velocities) != joint_count:
        #     raise ValueError(f"Velocities length {len(self.velocities)} != joint count {joint_count}")
        #
        # if self.accelerations and len(self.accelerations) != joint_count:
        #     raise ValueError(f"Accelerations length {len(self.accelerations)} != joint count {joint_count}")
        #
        # if self.effort and len(self.effort) != joint_count:
        #     raise ValueError(f"Effort length {len(self.effort)} != joint count {joint_count}")

        for i in range(joint_count):
            joint_info = joints[i]
            value = self.positions[i]
            joint_info.validate_value(value)


class Trajectory(BaseModel):
    """
    机器人带有 kv 命名的轨迹描述.
    用于换算为底层运控模块的真实轨迹.
    基础实现和 ROS2 的对齐.
    """

    loop: int = Field(
        default=1,
        description="运动轨迹的循环次数, 解析时会生成多次循环的轨迹动画.",
    )

    joint_names: list[str] = Field(default_factory=list, description="关节的名称, 与每个轨迹点的位置值一致. ")
    points: list[TrajectoryPoint] = Field(default_factory=list, description="轨迹中的点位")

    @classmethod
    def from_pose(cls, pose: Pose, duration: float = 1.0) -> Self:
        joint_names = list(pose.positions.keys())
        values = []
        for j_name in joint_names:
            values.append(pose.positions[j_name])
        trajectory = cls(
            joint_names=joint_names,
            points=[
                TrajectoryPoint(
                    positions=values,
                    time_from_start=duration,
                )
            ],
        )
        return trajectory


class Controller(BaseModel):
    """
    机器人建模中, 可独立并行运行的组件.
    """

    name: str = Field(description="组件的名称")
    description: str = Field(description="组件的描述")
    joints: dict[str, Joint] = Field(default_factory=dict, description="组件可以管理的所有关节")

    def with_joint(self, *joints: Joint) -> Self:
        for joint in joints:
            self.joints[joint.name] = joint
        return self

    def controller_description(self) -> dict:
        """
        默认的描述机制.
        """
        description = self.model_dump(exclude={"joints"})
        joint_descriptions = []
        for joint in self.joints.values():
            joint_descriptions.append(joint.joint_description())
        description["joints"] = joint_descriptions
        return description


class Transition(BaseModel):
    """
    关键帧动画的过渡片段.
    """

    time: float = Field(
        default=1.0,
        description="位姿变更的消耗时间",
    )
    positions: dict[str, float] = Field(
        default_factory=dict, description="指定运行的姿态描述. 如果 pose name 存在的话, 则不需要这个参数."
    )
    duration: float = Field(default=0.0, description="这个位姿持续的时间")


class Animation(BaseModel):
    """
    关键帧动画.
    """

    name: str = Field(description="运动动画的命名")
    description: str = Field(default="", description="运动动画的描述")
    transitions: list[Transition] = Field(default_factory=list, description="关键帧动画片段. ")
    loop: int = Field(
        default=1,
        description="动画中关键帧的循环次数, 每次会从头开始执行 Transition",
    )

    def to_trajectory(
        self,
        *,
        start_positions: Optional[dict[str, float]] = None,
        joint_names: Optional[list[str]] = None,
    ) -> Trajectory:
        """
        转化为 Trajectory 数据对象.
        """
        if start_positions is not None and len(start_positions) == 0:
            raise ValueError(f"start_positions length {len(start_positions)} shall not be 0 if not None")

        points: list[TrajectoryPoint] = []
        joint_names: list[str] | None = joint_names
        current_positions: dict[str, float] | None = None
        time_from_start = 0.0
        for transition in self.transitions:
            # 完成初始化.
            if current_positions is None:
                current_positions = transition.positions
            if joint_names is None:
                joint_names = list(transition.positions.keys())

            new_positions = current_positions.copy()
            # 更新当前目标姿态.
            new_positions.update(transition.positions)

            # 生成有序 position 数组.
            positions_values = []
            for j_name in joint_names:
                positions_values.append(current_positions[j_name])

            # 生成进入目标位姿的 point.
            if transition.time > 0.0:
                time_from_start += transition.time
                move_to_target = TrajectoryPoint(
                    time_from_start=time_from_start,
                    positions=positions_values,
                )
            else:
                # 不合法的
                raise ValueError(f"transition time cost shall not be 0. {transition}")

            points.append(move_to_target)
            # 生成持续保留在目标位姿的 point.
            if transition.duration > 0.0:
                time_from_start += transition.duration
                keep_at_target = TrajectoryPoint(
                    time_from_start=time_from_start,
                    positions=positions_values,
                )
                points.append(keep_at_target)
            # 更新当前姿态.
            current_positions = new_positions

        # 生成动画.
        return Trajectory(
            joint_names=joint_names,
            loop=self.loop,
            points=points,
        )


class PoseTransition(BaseModel):
    time: float = Field(
        default=1.0,
        description="位姿变更的消耗时间",
    )
    pose_name: str = Field(
        description="指定的位姿名称",
    )
    duration: float = Field(default=0.0, description="这个位姿持续的时间")

    def to_transition(self, pose: Pose) -> Transition:
        return Transition(
            time=self.time,
            positions=pose.positions,
            duration=self.duration,
        )


class PoseAnimation(BaseModel):
    name: str = Field(description="运动动画的命名")
    description: str = Field(default="", description="运动动画的描述")
    transitions: list[PoseTransition] = Field(default_factory=list, description="动画的过度片段")

    def append(self, pose_name: str, time: float = 1.0, duration: float = 1.0) -> None:
        transition = PoseTransition(pose_name=pose_name, time=time, duration=duration)
        self.transitions.append(transition)

    def to_animation(self, poses: dict[str, Pose]) -> Animation:
        transitions = []
        for pose_transition in self.transitions:
            pose = poses.get(pose_transition.pose_name)
            if pose is None:
                raise ValueError(f"pose {pose_transition.pose_name} not exist")
            transition = pose_transition.to_transition(pose)
            transitions.append(transition)
        return Animation(name=self.name, description=self.description, transitions=transitions)


class RobotInfo(BaseModel):
    """
    机器人的建模信息. 预计这个信息未来可以结合 urdf 等方式, 结合大模型自动生成.
    """

    name: str = Field(description="机器人的名称")
    description: str = Field(default="", description="机器人的描述")
    controllers: dict[str, Controller] = Field(default_factory=dict, description="可并行控制组件的描述")
    default_pose_name: str = Field(
        default="",
        description="默认的 pose 名称",
    )
    poses: dict[str, Pose] = Field(default_factory=dict, description="已经存储的所有位姿")
    animations: dict[str, Animation] = Field(
        default_factory=dict, description="机器人已经存储的所有的运动轨迹动画. 全身性的."
    )

    def robot_description(self) -> dict[str, Any]:
        """
        返回机器人的描述信息.
        """
        description = self.model_dump(include={"name", "description"})
        controllers = []
        for controller in self.controllers.values():
            controllers.append(controller.controller_description())
        description["controllers"] = controllers
        return description

    def poses_description(self) -> dict[str, str]:
        """
        姿态的描述
        """
        description = {}
        for pose_name, pose in self.poses.items():
            if pose.name != pose_name:
                continue
            description[pose_name] = pose.description
        return description

    def animation_description(self) -> dict[str, str]:
        """
        姿态的描述
        """
        description = {}
        for animation_name, animation in self.animations.items():
            if animation.name != animation_name:
                continue
            description[animation_name] = animation.description
        return description

    def with_controller(self, *controllers: Controller) -> Self:
        for controller in controllers:
            self.controllers[controller.name] = controller
        return self

    def default_pose(self) -> Pose:
        if self.default_pose_name:
            pose = self.poses.get(self.default_pose_name)
            if pose is not None:
                return pose
        return self._default_pose()

    def _default_pose(self) -> Pose:
        """
        return the default pose of the robot
        """
        positions = {}
        for comp in self.controllers.values():
            for name, j in comp.joints.items():
                positions[name] = j.default_value

        return Pose(
            description="default pose of the robot",
            positions=positions,
        )

    def joints(self) -> dict[str, Joint]:
        joints = {}
        for comp in self.controllers.values():
            joints.update(comp.joints)
        return joints

    def iter_joints(self) -> Iterable[Joint]:
        for comp in self.controllers.values():
            yield from comp.joints.values()

    def validate_trajectory(self, trajectory: Trajectory) -> None:
        """
        用机器人校验一个运动轨迹是否合法.
        """
        joints = self.joints()
        trajectory_joint_names = trajectory.joint_names
        selected_joints = []
        for name in trajectory_joint_names:
            if name not in joints:
                raise ValueError(f"joint name {name} not in robot joints")
            selected_joints.append(joints[name])

        for point in trajectory.points:
            point.validate_kinematics(selected_joints)
