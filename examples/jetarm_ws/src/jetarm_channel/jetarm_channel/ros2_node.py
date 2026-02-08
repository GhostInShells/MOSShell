try:
    import rclpy
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import JointState
except ImportError:
    raise ImportError("Please run in Ros2 pkg.")

from collections.abc import Callable
from typing import Optional

from ghoshell_common.contracts import DefaultFileStorage, LoggerItf

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider
from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import RobotController
from ghoshell_moss_contrib.prototypes.ros2_robot.main_channel import build_robot_main_channel
from ghoshell_moss_contrib.prototypes.ros2_robot.manager import JointValueParser, YamlStorageRobotManager
from ghoshell_moss_contrib.prototypes.ros2_robot.models import RobotInfo

from .ros2_controller import Ros2Controller

__all__ = ["MAIN_CHANNEL_BUILDER", "Ros2RobotControllerNode", "run_node"]

MAIN_CHANNEL_BUILDER = Callable[[Channel, RobotController], Channel]


class Ros2LoggerAdapter(LoggerItf):
    def __init__(self, logger):
        self._rcutils_logger = logger

    def debug(self, msg, *args, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self._rcutils_logger.debug(msg, **kwargs)

    def info(self, msg, *args, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self._rcutils_logger.info(msg, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self._rcutils_logger.warning(msg, **kwargs)

    def error(self, msg, *args, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self._rcutils_logger.error(msg, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self.error(msg, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if len(args) > 0:
            msg = msg % args
        self._rcutils_logger.fatal(msg, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        msg = f"[{level}]: {msg}"
        self.info(msg, *args, **kwargs)


class Ros2RobotControllerNode(Node):
    def __init__(
        self,
        *,
        node_name: str,
        config_dir: str,
        robot_yaml_filename: str,
        provider: ChannelProvider,
        channel_builder: MAIN_CHANNEL_BUILDER | None = None,
        default_robot: Optional[RobotInfo] = None,
        joint_states_topic: str = "/joint_states",
        follow_joint_trajectory_server_name: str = "/joint_trajectory_controller/follow_joint_trajectory",
        joint_value_parsers: Optional[dict[str, JointValueParser]] = None,
        goal_interval: float = 0.02,  # 50Hz
    ):
        super().__init__(node_name)

        # 初始化参数
        self.declare_parameter("goal_interval", goal_interval)

        # 获取参数
        self._default_robot = default_robot
        self._joint_states_topic = joint_states_topic
        self._action_server_name = follow_joint_trajectory_server_name

        self._last_joint_callback_time = None

        logger = Ros2LoggerAdapter(self.get_logger())

        # 创建 Action 客户端
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._action_server_name,
        )

        storage = DefaultFileStorage(config_dir)
        manager = YamlStorageRobotManager(
            robot_yaml_filename,
            storage,
            logger=logger,
            default_robot=default_robot,
            parsers=joint_value_parsers,
        )
        self.manager = manager

        # 创建控制器实例
        self.controller = Ros2Controller(
            manager=manager, logger=logger, trajectory_action_client=self.action_client, goal_interval=goal_interval
        )

        # 设置关节状态订阅
        self._setup_joint_states_subscription(joint_states_topic)
        self.main_channel = build_robot_main_channel(self.controller)
        self.main_channel.build.with_binding(LoggerItf, logger)

        # 完成更多初始化步骤.
        if channel_builder is not None:
            self.main_channel = channel_builder(self.main_channel, self.controller)

        self.provider = provider

        # 启动控制器
        self.controller.start()
        # 多线程启动 channel.
        self.provider.run_in_thread(self.main_channel)

        self.get_logger().info(f"Robot {self.manager.robot().name} Controller Node initialized")

    def _setup_joint_states_subscription(self, topic_name: str):
        """设置关节状态订阅"""

        qos_profile = QoSProfile(
            depth=10, history=QoSHistoryPolicy.KEEP_LAST, reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.joint_states_subscription = self.create_subscription(
            JointState, topic_name, self._joint_states_callback, qos_profile
        )

        self.get_logger().info(f"Subscribed to joint states: {topic_name}")

    def _joint_states_callback(self, msg: JointState):
        """处理关节状态消息"""
        try:
            current_time = self.get_clock().now()
            goal_interval = self.get_parameter("goal_interval").value

            # 第一次回调时初始化
            if self._last_joint_callback_time is None:
                self._last_joint_callback_time = current_time
                # 立即处理第一次消息
            else:
                time_since_last = current_time - self._last_joint_callback_time
                desired_interval = goal_interval
                if time_since_last.nanoseconds < desired_interval * 1e9:
                    return
        except Exception as e:
            self.get_logger().fatal(e)

        try:
            # 提取关节位置
            positions = {}
            for i, joint_name in enumerate(msg.name):
                if i < len(msg.position):
                    positions[joint_name] = msg.position[i]

            # 更新控制器中的原始位置
            self.controller.update_raw_positions(positions)

        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")
        finally:
            self._last_joint_callback_time = self.get_clock().now()

    def destroy_node(self):
        """清理资源"""
        self.get_logger().info("Shutting down JetArm Controller Node")

        # 关闭服务.
        self.provider.close()
        # 关闭控制器
        self.controller.close()
        self.controller.wait_closed()

        # 调用父类清理
        super().destroy_node()


# 简单的测试主函数
def run_node(args, node: Ros2RobotControllerNode):
    try:
        # 保持节点运行
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"Node error: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()
