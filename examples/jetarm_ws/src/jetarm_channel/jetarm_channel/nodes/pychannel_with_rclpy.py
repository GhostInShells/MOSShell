import asyncio

# action_client_node.py
import rclpy
from control_msgs.action import FollowJointTrajectory
from ghoshell_common.contracts import LoggerItf
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from ghoshell_moss import PyChannel
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProvider


class JetArmChannelTestClient(Node):
    def __init__(self):
        super().__init__("jetarm_channel_test_node")

        # 创建 Action 客户端
        self.action_client = ActionClient(
            self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )

        self.get_logger().info("Action客户端已初始化")

    def send_goal(self, joint_names: list[str], positions: list[float], time_sec: float = 5.0):
        """发送轨迹目标"""

        # 等待 Action 服务器可用
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action服务器不可用")
            return None

        # 构建目标消息
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names

        # 创建轨迹点
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(time_sec)
        point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        # 发送目标
        self.get_logger().info(f"发送目标: {positions}")
        return self.action_client.send_goal_async(goal_msg)


# 使用示例
def main(args=None):
    rclpy.init(args=args)

    action_client = JetArmChannelTestClient()

    main_channel = PyChannel(name="test_channel")
    main_channel.build.with_binding(
        LoggerItf,
        action_client.get_logger,
    )

    @main_channel.build.command()
    async def reset_body():

        # 发送目标
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
        positions = [0.0, 1.0, -1.57, -1.57, 0.2, 0.0]

        future = action_client.send_goal(joint_names, positions, 3.0)
        try:
            interval = 1.0 / 50
            while not future.done():
                await asyncio.sleep(interval)
                continue

            r = future.result()
            action_client.get_logger().info(f"目标执行完成: {r}")
            return None
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()

    provider = ZMQChannelProvider(
        address="tcp://127.0.0.1:9527",
    )

    try:
        action_client.get_logger().info("初始化 channel provider")
        provider.run_in_thread(main_channel)
        action_client.get_logger().info("已经运行 pychannel, 开始 spin")
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
