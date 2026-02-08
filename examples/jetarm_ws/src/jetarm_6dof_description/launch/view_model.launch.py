import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue  # 确保导入这个


def generate_launch_description():
    # 获取当前包（jetarm_6dof_description）的路径
    pkg_share = get_package_share_directory("jetarm_6dof_description")

    # 定义Launch参数，允许从命令行指定URDF文件
    urdf_model_path = os.path.join(pkg_share, "urdf", "jetarm_6dof_robot.urdf.xacro")
    # 如果上面的主文件不行，可以尝试其他xacro文件，例如：
    # urdf_model_path = os.path.join(pkg_share, 'urdf', 'jetarm_6dof_description.urdf.xacro')

    # 声明一个可选的启动参数，用于在启动时指定URDF文件
    urdf_model = DeclareLaunchArgument(
        name="urdf_model", default_value=urdf_model_path, description="Absolute path to robot urdf file"
    )

    # 启动 robot_state_publisher 节点
    # 这个节点的作用是：读取URDF文件，并将机器人的关节状态转换为TF变换，并发布到 /robot_description 话题
    robot_description_content = ParameterValue(Command(["xacro ", LaunchConfiguration("urdf_model")]), value_type=str)

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                # 直接使用上面生成的字符串内容
                "robot_description": robot_description_content
            }
        ],
    )

    # 启动 joint_state_publisher 节点
    # 这个节点提供一个GUI，可以手动拖拽控制每个关节，用于测试模型联动是否正确
    joint_state_publisher_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
    )

    # 启动 RVIZ2 可视化工具
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(pkg_share, "config", "view_model.rviz")],
    )

    return LaunchDescription(
        [
            urdf_model,
            robot_state_publisher_node,
            joint_state_publisher_node,
            rviz2_node,
        ]
    )
