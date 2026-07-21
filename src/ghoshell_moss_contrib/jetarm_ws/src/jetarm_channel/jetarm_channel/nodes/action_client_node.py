import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node


class JetArmActionClient(Node):
    def __init__(self):
        super().__init__("jetarm_action_client")

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
        from trajectory_msgs.msg import JointTrajectoryPoint

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(time_sec)
        point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        # 发送目标
        self.get_logger().info(f"发送目标: {positions}")
        return self.action_client.send_goal_async(goal_msg)

    def cancel_goal(self):
        """取消当前目标"""
        if hasattr(self, "_goal_handle") and self._goal_handle:
            self.get_logger().info("取消目标")
            return self._goal_handle.cancel_goal_async()


# 使用示例
def main(args=None):
    rclpy.init(args=args)

    action_client = JetArmActionClient()

    try:
        # 发送目标
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
        positions = [0.0, 1.0, -1.57, -1.57, 0.2, 0.0]

        future = action_client.send_goal(joint_names, positions, 3.0)

        # 等待结果
        rclpy.spin_until_future_complete(action_client, future)

        if future.result() is not None:
            action_client.get_logger().info("目标执行完成")
        else:
            action_client.get_logger().error("执行失败")

    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
