#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration

# ROS 2 消息类型
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from action_msgs.msg import GoalStatus

# 硬件 SDK
from jetarm_driver.jetarm_stm_sdk import Board
from ament_index_python.packages import get_package_share_directory

import os
import yaml
import threading
import time
from typing import List, Dict, Optional
from enum import Enum


class DriverState(Enum):
    """节点状态机"""
    IDLE = 0  # 就绪，等待指令
    ACTIVE = 1  # 正在执行轨迹
    ERROR = 2  # 发生错误


class JetArmDriverNode(Node):
    """简化版的 JetArm 驱动节点"""

    def __init__(self):
        super().__init__('jetarm_driver')

        # 状态初始化
        self._state = DriverState.IDLE
        self._current_trajectory = None
        self._trajectory_goal_handle = None
        self._trajectory_lock = threading.Lock()

        # 加载配置
        self._load_configuration()

        # 初始化硬件
        self._init_hardware()

        # 使用回调组
        self._action_callback_group = ReentrantCallbackGroup()

        # 创建 Action 服务器
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'follow_joint_trajectory',
            self._execute_trajectory_callback,
            callback_group=self._action_callback_group,
            handle_accepted_callback=self._handle_accepted_callback
        )

        # 创建服务
        self._cancel_all_service = self.create_service(
            Trigger, 'cancel_all', self._cancel_all_callback,
            callback_group=self._action_callback_group
        )

        self._home_service = self.create_service(
            Trigger, 'go_to_home', self._go_to_home_callback,
            callback_group=self._action_callback_group
        )

        # 创建状态发布器
        self._joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)

        # 状态发布定时器
        self._status_timer = self.create_timer(0.1, self._publish_joint_states)

        self.get_logger().info('🚀 JetArm驱动节点已初始化并就绪')

    def _load_configuration(self):
        """加载关节映射配置"""
        try:
            config_file = os.path.join(
                get_package_share_directory('jetarm_driver'),
                'config', 'joint_mapping.yaml'
            )
            with open(config_file, 'r') as file:
                self.joint_mapping = yaml.safe_load(file)
            self.joint_names = list(self.joint_mapping.keys())
            self.get_logger().info(f'✅ 已加载关节配置: {list(self.joint_names)}')
        except Exception as e:
            self.get_logger().error(f'❌ 加载配置失败: {e}')
            self._state = DriverState.ERROR
            raise

    def _init_hardware(self):
        """初始化硬件连接"""
        try:
            self._board = Board(device="/dev/ttyUSB0", baudrate=1000000, timeout=1)
            self._board.enable_reception(True)
            self.get_logger().info('✅ 硬件连接成功')
        except Exception as e:
            self.get_logger().error(f'❌ 硬件初始化失败: {e}')
            self._state = DriverState.ERROR

    def _handle_accepted_callback(self, goal_handle):
        """处理新接受的目标准备执行"""
        self.get_logger().info('🎯 处理新接受的目标准备执行')

        # 如果有正在执行的目标，取消它
        with self._trajectory_lock:
            if (self._trajectory_goal_handle is not None and
                    self._trajectory_goal_handle.is_active):
                self.get_logger().info('⏹️ 取消当前执行的目标')
                self._trajectory_goal_handle.abort()

        # 执行新目标
        goal_handle.execute()

    def _execute_trajectory_callback(self, goal_handle):
        """执行轨迹回调 - 修正版本"""
        self.get_logger().info('📥 收到轨迹执行请求')

        # 检查节点状态
        if self._state == DriverState.ERROR:
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "驱动器处于错误状态"
            goal_handle.abort()
            return result

        # 验证轨迹
        trajectory = goal_handle.request.trajectory
        if error_msg := self._validate_trajectory(trajectory):
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = error_msg
            goal_handle.abort()
            return result

        # 保存轨迹和目标句柄
        with self._trajectory_lock:
            self._current_trajectory = trajectory
            self._trajectory_goal_handle = goal_handle
            self._trajectory_start_time = self.get_clock().now()

        # 切换到执行状态
        self._state = DriverState.ACTIVE

        # 执行轨迹
        try:
            self._execute_trajectory_sync(trajectory, goal_handle)

            # 成功完成
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()

        except Exception as e:
            # 执行失败
            self.get_logger().error(f'❌ 轨迹执行失败: {e}')
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = str(e)
            goal_handle.abort()

        # 切换回空闲状态
        with self._trajectory_lock:
            self._current_trajectory = None
            self._trajectory_goal_handle = None
            self._state = DriverState.IDLE

        return result

    def _execute_trajectory_sync(self, trajectory, goal_handle):
        """同步执行轨迹 - 修正版本"""
        start_time = self.get_clock().now()
        prev_point_time = 0.0  # 上一个点的时间偏移

        for i, point in enumerate(trajectory.points):
            # 检查是否被取消
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self.get_logger().info('⏹️ 轨迹执行被取消')
                if goal_handle is not None:
                    goal_handle.canceled()
                return

            # 计算当前点的时间偏移（从轨迹开始到该点的持续时间）
            current_point_time = point.time_from_start.sec + point.time_from_start.nanosec / 1e9

            # 计算运动持续时间（当前点与上一个点的时间差）
            if i == 0:
                # 第一个点的持续时间就是它自己的时间偏移
                move_duration = current_point_time
            else:
                # 后续点的持续时间是当前点时间偏移减去上一个点的时间偏移
                move_duration = current_point_time - prev_point_time

            # 执行该点，传入持续时间
            self._execute_point(point, trajectory.joint_names, move_duration)

            # 发送反馈（如果有目标句柄）
            if goal_handle is not None:
                feedback = FollowJointTrajectory.Feedback()

                # 创建 actual 轨迹点（使用 JointTrajectoryPoint 类型）
                actual_point = JointTrajectoryPoint()
                actual_point.positions = point.positions
                feedback.actual = actual_point

                # 创建 desired 轨迹点（使用 JointTrajectoryPoint 类型）
                desired_point = JointTrajectoryPoint()
                desired_point.positions = point.positions
                feedback.desired = desired_point

            # 等待到下一个点的执行时间
            if i < len(trajectory.points) - 1:  # 如果不是最后一个点
                next_point_time = trajectory.points[i + 1].time_from_start.sec + \
                                  trajectory.points[i + 1].time_from_start.nanosec / 1e9
                wait_time = next_point_time - current_point_time

                # 使用 time.sleep 等待
                if wait_time > 0:
                    time.sleep(wait_time)

            # 更新上一个点的时间偏移
            prev_point_time = current_point_time

            self.get_logger().info(f'📊 执行进度: {i + 1}/{len(trajectory.points)}')

    def _execute_point(self, point, joint_names, duration: float):
        """执行单个轨迹点 - 修正版本"""
        servo_commands = []
        for i, joint_name in enumerate(joint_names):
            if joint_name not in self.joint_mapping:
                continue

            pulse = self._radians_to_pulse(point.positions[i], joint_name)
            servo_commands.append([self.joint_mapping[joint_name]['id'], pulse])

        # 执行硬件命令
        self._board.bus_servo_set_position(duration, servo_commands)

    def _publish_joint_states(self):
        """发布关节状态"""
        try:
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.name = self.joint_names

            # 读取硬件状态
            positions = []
            for joint_name in self.joint_names:
                servo_id = self.joint_mapping[joint_name]['id']
                pos_data = self._board.bus_servo_read_position(servo_id)
                if pos_data is not None:
                    positions.append(self._pulse_to_radians(pos_data[0], joint_name))
                else:
                    positions.append(0.0)

            joint_state.position = positions
            self._joint_state_pub.publish(joint_state)

        except Exception as e:
            self.get_logger().warn(f'发布关节状态失败: {e}')

    def _validate_trajectory(self, trajectory) -> Optional[str]:
        """验证轨迹有效性"""
        if not trajectory.joint_names:
            return "关节名称为空"

        for joint_name in trajectory.joint_names:
            if joint_name not in self.joint_mapping:
                return f"未知关节: {joint_name}"

        last_time_from_start = 0.0
        for point in trajectory.points:
            if point.time_from_start <= last_time_from_start:
                raise ValueError(
                    f"time from start {point.time_from_start} not greater than last one {last_time_from_start}",
                )
            last_time_from_start = point.time_from_start
            for i, joint_name in enumerate(trajectory.joint_names):
                if errmsg := self.validate_joint_position(point.positions[i], joint_name):
                    return errmsg
        return None

    def validate_joint_position(self, radians: float, joint_name: str) -> Optional[str]:
        """验证关节位置是否在安全范围内"""
        if joint_name not in self.joint_mapping:
            return f"未知关节: {joint_name}"

        config = self.joint_mapping[joint_name]
        min_rad = config['min_rad']
        max_rad = config['max_rad']

        if radians < min_rad or radians > max_rad:
            return (
                f"关节 {joint_name} 的目标位置 {radians:.2f}rad 超出安全范围 "
                f"[{min_rad:.2f}, {max_rad:.2f}]rad"
            )
        return None

    def _pulse_to_radians(self, pulse: int, joint_name: str) -> float:
        """将硬件脉宽值转换为标准弧度值"""
        config = self.joint_mapping[joint_name]
        min_pulse = config['min_pulse']
        max_pulse = config['max_pulse']
        min_rad = config['min_rad']
        max_rad = config['max_rad']

        # 确保脉冲值在有效范围内
        clamped_pulse = max(min_pulse, min(pulse, max_pulse))

        # 线性映射：脉冲值 -> 弧度值
        normalized = (clamped_pulse - min_pulse) / (max_pulse - min_pulse)
        radians = min_rad + normalized * (max_rad - min_rad)

        # 应用方向系数
        direction = config.get('direction', 1)
        return radians * direction

    def _radians_to_pulse(self, radians: float, joint_name: str) -> int:
        """将标准弧度值转换为硬件脉宽值"""
        config = self.joint_mapping[joint_name]
        min_pulse = config['min_pulse']
        max_pulse = config['max_pulse']
        min_rad = config['min_rad']
        max_rad = config['max_rad']

        # 应用方向系数
        direction = config.get('direction', 1)
        adjusted_radians = radians * direction

        # 确保弧度值在有效范围内
        clamped_radians = max(min_rad, min(adjusted_radians, max_rad))

        # 线性映射：弧度值 -> 脉冲值
        normalized = (clamped_radians - min_rad) / (max_rad - min_rad)
        pulse = min_pulse + normalized * (max_pulse - min_pulse)

        return int(round(pulse))

    def _create_joint_state_message(self, positions: List[float], joint_names: List[str]) -> JointState:
        """创建关节状态消息"""
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = joint_names
        joint_state.position = positions
        return joint_state

    def _cancel_all_callback(self, request, response):
        """取消所有命令回调"""
        self.get_logger().info('⏹️ 收到取消所有命令请求')

        # 取消当前轨迹
        with self._trajectory_lock:
            if (self._trajectory_goal_handle is not None and
                    self._trajectory_goal_handle.is_active):
                self._trajectory_goal_handle.abort()
                self._current_trajectory = None
                self._trajectory_goal_handle = None

        # 切换到空闲状态
        self._state = DriverState.IDLE

        response.success = True
        response.message = "已取消所有命令"
        return response

    def _go_to_home_callback(self, request, response):
        """回家位置回调"""
        self.get_logger().info('🏠 收到回家位置请求')

        try:
            # 创建回家轨迹
            home_trajectory = JointTrajectory()
            home_trajectory.joint_names = self.joint_names

            # 添加回家点
            point = JointTrajectoryPoint()
            point.positions = [0.0] * len(self.joint_names)  # 所有关节归零
            point.time_from_start = Duration(seconds=3.0).to_msg()
            home_trajectory.points = [point]

            # 执行回家轨迹
            self._execute_trajectory_sync(home_trajectory, None)

            response.success = True
            response.message = "已回到Home位置"
        except Exception as e:
            response.success = False
            response.message = f"回家失败: {str(e)}"

        return response

    def destroy_node(self):
        """节点销毁时的清理工作"""
        self.get_logger().info('正在关闭驱动节点...')

        # 安全停止所有舵机
        try:
            all_ids = [config['id'] for config in self.joint_mapping.values()]
            self._board.bus_servo_stop(all_ids)
        except Exception as e:
            self.get_logger().error(f'停止舵机失败: {e}')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = JetArmDriverNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        executor.spin()

    except KeyboardInterrupt:
        node.get_logger().info('节点被用户中断')
    except Exception as e:
        node.get_logger().error(f'节点运行错误: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
