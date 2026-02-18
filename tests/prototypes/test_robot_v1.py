import pytest

from ghoshell_moss_contrib.prototypes.ros2_robot.joint_parsers import DegreeToRadiansParser, default_parsers
from ghoshell_moss_contrib.prototypes.ros2_robot.main_channel import build_robot_main_channel
from ghoshell_moss_contrib.prototypes.ros2_robot.manager import MemoryRobotManager
from ghoshell_moss_contrib.prototypes.ros2_robot.mocks import MockRobotController
from ghoshell_moss_contrib.prototypes.ros2_robot.models import Controller, Joint, PoseAnimation, RobotInfo, Trajectory

test_robot = RobotInfo(
    name="test_robot",
    description="test robot",
).with_controller(
    Controller(
        name="arm",
        description="arm",
    ).with_joint(
        Joint(
            name="shoulder",
            origin_name="joint1",
            default_value=0.0,
            min_value=-180.0,
            max_value=180.0,
            value_parser="degrees_to_radians",
        )
    ),
)


def test_robot_info():
    assert len(test_robot.controllers) == 1
    assert test_robot.controllers["arm"].name == "arm"
    assert test_robot.controllers["arm"].joints["shoulder"].name == "shoulder"
    joint = test_robot.controllers["arm"].joints["shoulder"]
    assert joint.value_parser == "degrees_to_radians"


def test_robot_manager_baseline():
    manager = MemoryRobotManager(test_robot, {"degrees_to_radians": DegreeToRadiansParser()})
    robot = manager.robot()
    assert robot.name == test_robot.name

    default_pose = manager.get_default_pose()
    assert default_pose.positions["shoulder"] == 0.0

    test_pose = default_pose.model_copy(update={"name": "test"})
    test_pose.positions["shoulder"] = 180.0
    manager.save_pose(test_pose)

    # test pose
    test_pose = manager.get_pose("test")
    assert test_pose.name == "test"

    pose_animation = PoseAnimation(name="test_pose_animation")
    pose_animation.append(time=1.0, pose_name="test", duration=1.0)
    manager.save_pose_animation(pose_animation)

    animation = manager.get_animation("test_pose_animation")
    traj = animation.to_trajectory()
    assert len(traj.joint_names) == 1
    assert len(traj.points) == 2

    got = manager.to_raw_trajectory(traj)
    assert got.joint_names == ["joint1"]
    assert round(got.points[0].positions[0], 3) in (3.142, 3.141)


def test_robot_controller_get_position():
    robot = RobotInfo(
        name="test_robot",
        description="test robot",
    ).with_controller(
        Controller(
            name="arm",
            description="arm",
        ).with_joint(
            Joint(
                name="shoulder",
                origin_name="joint1",
                default_value=30.0,
                min_value=-180.0,
                max_value=180.0,
                value_parser=DegreeToRadiansParser.name(),
            )
        ),
    )
    manager = MemoryRobotManager(robot, default_parsers)
    pose = manager.get_default_pose()
    origin_values = pose.positions
    positions = manager.from_joint_values_to_positions(pose.positions)
    values = manager.from_joint_positions_to_values(positions)
    assert abs(origin_values["shoulder"] - values["shoulder"]) < 0.01

    _controller = MockRobotController(manager)
    _controller.update_raw_positions(positions)
    assert _controller.get_current_position_values() == values


@pytest.mark.asyncio
async def test_robot_main_channel():
    _manager = MemoryRobotManager(test_robot, {"degrees_to_radians": DegreeToRadiansParser()})
    _controller = MockRobotController(_manager)
    main_channel = build_robot_main_channel(_controller)
    pose = _manager.get_default_pose()
    traj = Trajectory.from_pose(pose)

    async with main_channel.bootstrap():
        meta = main_channel.broker.self_meta()
        # 检查下 meta 可以被正确生成.
        # assert _manager.robot().name in meta.description

        command = main_channel.broker.get_self_command("run_trajectory")
        r = await command(traj.model_dump_json())
        assert r is None
        values = _controller.get_current_position_values()
        assert values == pose.positions
