import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("jetarm_control")

    # 路径定义
    hw_yaml = os.path.join(pkg_share, "config", "jetarm_hardware.yaml")
    ctrl_yaml = os.path.join(pkg_share, "config", "controllers.yaml")  # 使用原始配置文件

    # XACRO 处理器
    robot_file = os.path.join(
        get_package_share_directory("jetarm_6dof_description"), "urdf", "jetarm_6dof_robot.urdf.xacro"
    )
    robot_file_content = ParameterValue(Command(["xacro ", robot_file]), value_type=str)

    # 1. 机器人状态发布器
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_file_content}],
        output="screen",
    )

    # 2. 控制器管理器节点 (ros2_control_node)
    cm_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_file_content},  # URDF 内容
            hw_yaml,  # 硬件配置文件
            ctrl_yaml,  # 控制器配置文件 (包含 update_rate 和控制器参数)
        ],
        output="screen",
        # 确保在控制器加载前启动，这里设置为在 cm_node 启动后才启动 spawner
        # 但由于 cm_node 比较关键，不需要特别设置 dependencies
    )

    # 3. 加载 Joint State Broadcaster 的 spawner 节点
    # 关键修改：添加 --controller-type 参数
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",  # 控制器实例名
            "--controller-type",
            "joint_state_broadcaster/JointStateBroadcaster",  # 控制器类型
        ],
        output="screen",
    )

    # 4. 加载 Joint Trajectory Controller 的 spawner 节点
    # 关键修改：添加 --controller-type 参数
    joint_trajectory_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",  # 控制器实例名
            "--controller-type",
            "joint_trajectory_controller/JointTrajectoryController",  # 控制器类型
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_pub,
            cm_node,
            # 先启动 State Broadcaster
            joint_state_spawner,
            # 延迟启动 Trajectory Controller，确保硬件接口和状态广播器已就绪
            TimerAction(
                period=3.0,
                actions=[
                    joint_trajectory_spawner,
                ],
            ),
        ]
    )
