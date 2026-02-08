from ghoshell_common.contracts import LoggerItf

from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import (
    MOSSRobotManager,
    RobotController,
    TrajectoryAction,
)
from ghoshell_moss_contrib.prototypes.ros2_robot.models import Trajectory

try:
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from trajectory_msgs.msg import JointTrajectoryPoint
except ImportError as e:
    raise ImportError(f"Ros2Controller requires ros2 environment rclpy to be installed.: {e}")

import logging
import queue
import threading
import time


class Ros2Controller(RobotController):
    def __init__(
        self,
        manager: MOSSRobotManager,
        trajectory_action_client: ActionClient,
        logger: LoggerItf | None = None,
        goal_interval: float = 1.0 / 50,
    ):
        self._action_client = trajectory_action_client
        self._manager = manager
        self._logger = logger or logging.getLogger(__name__)
        self._start = False
        self._close_event = threading.Event()
        self._moving_stopped = threading.Event()
        # 当前存在的轨迹运动命令状态
        self._traj_actions: list[TrajectoryAction] = []
        self._execute_queue: queue.Queue[TrajectoryAction] = queue.Queue()
        # 做 rclpy goal 的轮询周期.
        self._goal_interval = goal_interval

        # raw positions
        self._joint_positions_lock = threading.Lock()
        self._raw_joint_positions: dict[str, float] = {}
        self._loop_run_trajectory_actions_thread = threading.Thread(
            target=self._loop_run_trajectory_actions,
            daemon=True,
        )

    def _loop_run_trajectory_actions(self):
        while not self._close_event.is_set():
            try:
                action = self._execute_queue.get(block=True, timeout=0.1)
                if action.done():
                    continue
                try:
                    self._execute_trajectory_action(action)
                except Exception:
                    self._logger.exception("Execute trajectory action failed")
            except queue.Empty:
                continue
        self._close_event.set()

    def _execute_trajectory_action(self, trajectory_action: TrajectoryAction) -> None:
        self._logger.info("Executing trajectory action %s", trajectory_action.trajectory)
        # 循环检查是否 cancelled.
        if trajectory_action.done():
            return
        # 通过服务器检查判断是否可用.
        # todo: 需要想明白这个通讯是否是必要的.
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self._logger.error("Action服务器不可用")
            trajectory_action.set_exception(RuntimeError("Action 服务器不可用"))
            return

        goal = self._create_goal_from_trajectory(trajectory_action.trajectory)
        send_goal_future = self._action_client.send_goal_async(goal)
        goal_future = None
        try:
            goal_handle = None
            while not self._close_event.is_set():
                if trajectory_action.cancelled():
                    if not send_goal_future.done():
                        send_goal_future.cancel()
                        break
                if not send_goal_future.done():
                    time.sleep(self._goal_interval)
                    continue
                exp = send_goal_future.exception()
                if exp:
                    raise exp
                goal_handle = send_goal_future.result()
                break

            if not goal_handle:
                raise RuntimeError("Send goal with out future")

            goal_future = goal_handle.get_result_async()
            self._logger.info("Goal goal_future from goal handle : %s", goal_handle)

            result = None
            while not self._close_event.is_set():
                if trajectory_action.cancelled():
                    if not goal_future.done():
                        goal_future.cancel()
                        break
                if not goal_future.done():
                    time.sleep(self._goal_interval)
                    continue

                if exp := goal_future.exception():
                    self._logger.info("Goal execution failed: %s", exp)
                    trajectory_action.set_exception(exp)
                    break
                result = goal_future.result()
                break

            self._logger.info("Goal execution result %s", result)
            if result and result.status == GoalStatus.STATUS_SUCCEEDED:
                trajectory_action.set_result(None)
            else:
                exp = RuntimeError(f"Action failed: {result}")
                trajectory_action.set_exception(exp)

        except Exception as e:
            if not trajectory_action.done():
                trajectory_action.set_exception(e)
            self._logger.exception("Goal execution failed")
        finally:
            if not trajectory_action.done():
                trajectory_action.cancel()
            if send_goal_future and not send_goal_future.done():
                send_goal_future.cancel()
            if goal_future and not goal_future.done():
                goal_future.cancel()

    def _create_goal_from_trajectory(self, trajectory: Trajectory) -> FollowJointTrajectory.Goal:
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = trajectory.joint_names
        points = []
        for traj_point in trajectory.points:
            point = JointTrajectoryPoint()
            point.positions = traj_point.positions
            sec = int(traj_point.time_from_start)
            nanosec = int((traj_point.time_from_start - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            points.append(point)
        goal_msg.trajectory.points = points
        return goal_msg

    def close(self) -> None:
        if not self._close_event.is_set():
            self.stop_movement()
            self._close_event.set()

    def closed(self) -> bool:
        return self._close_event.is_set()

    def wait_closed(self) -> None:
        if not self._start:
            return
        self._close_event.wait()

    def start(self) -> None:
        if self._start:
            return
        self._start = True
        self._loop_run_trajectory_actions_thread.start()

    def manager(self) -> MOSSRobotManager:
        return self._manager

    def add_trajectory_actions(self, *actions: TrajectoryAction) -> None:
        for action in actions:
            # 运行时更新 action 的运动轨迹.
            action.trajectory = self.manager().to_raw_trajectory(action.trajectory)
            self._traj_actions.append(action)
            # 插入新的运动轨迹命令.
            self._execute_queue.put(action)

    def stop_movement(self) -> None:
        # 将所有运行中的轨迹命令设置取消.
        if len(self._traj_actions) > 0:
            traj_actions = self._traj_actions.copy()
            self._traj_actions.clear()
            for traj_action in traj_actions:
                # 取消命令.
                if not traj_action.done():
                    traj_action.cancel()

    def wait_for_available(self, timeout: float | None = None) -> None:
        self._action_client.wait_for_server(timeout_sec=timeout)

    def get_raw_positions(self) -> dict[str, float]:
        with self._joint_positions_lock:
            return self._raw_joint_positions.copy()

    def update_raw_positions(self, positions: dict[str, float]) -> None:
        with self._joint_positions_lock:
            self._raw_joint_positions = positions
