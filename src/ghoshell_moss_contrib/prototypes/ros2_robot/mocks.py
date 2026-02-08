from typing import Optional

from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import MOSSRobotManager, RobotController, TrajectoryAction


class MockRobotController(RobotController):
    def __init__(self, manager: MOSSRobotManager):
        self._manager = manager
        self._raw_positions: Optional[dict[str, float]] = None

    def close(self) -> None:
        pass

    def start(self) -> None:
        pass

    def closed(self) -> bool:
        pass

    def wait_closed(self) -> None:
        pass

    def manager(self) -> MOSSRobotManager:
        return self._manager

    def get_raw_positions(self) -> dict[str, float]:
        if self._raw_positions is None:
            default_positions = self._manager.get_default_pose().positions
            return self._manager.from_joint_values_to_positions(default_positions)
        return self._raw_positions

    def update_raw_positions(self, positions: dict[str, float]) -> None:
        self._raw_positions = positions

    def stop_movement(self) -> None:
        pass

    def wait_for_available(self, timeout: float | None = None) -> None:
        pass

    def add_trajectory_actions(self, *actions: TrajectoryAction) -> None:
        for action in actions:
            if len(action.trajectory.points) > 0:
                last = action.trajectory.points[-1]
                positions = {}

                for i, joint_name in enumerate(action.trajectory.joint_names):
                    positions[joint_name] = last.positions[i]
                positions = self.manager().from_joint_values_to_positions(positions)
                self.update_raw_positions(positions)
            action.set_result(None)
