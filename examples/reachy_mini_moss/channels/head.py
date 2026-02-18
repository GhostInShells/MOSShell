import logging

import numpy as np
from numpy import typing as npt
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import time_trajectory, linear_pose_interpolation

from examples.reachy_mini_moss.state import ReachyMiniState
from examples.reachy_mini_moss.vision.head_tracker import HeadTracker
from examples.reachy_mini_moss.vision.yolo.model import stringify_positions
from ghoshell_moss import PyChannel, Message, Text
from reachy_mini.utils import create_head_pose

class HeadMove(Move):
    def __init__(
            self,
            start_pose: npt.NDArray[np.float64],
            target_pose: npt.NDArray[np.float64],
            duration: float = 0.5,
    ):
        self.start_pose = start_pose
        self.target_pose = target_pose
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        interp_time = time_trajectory(t / self.duration)
        interp_head_pose = linear_pose_interpolation(
            self.start_pose, self.target_pose, interp_time
        )
        return interp_head_pose, None, None


class Head:
    def __init__(self, mini: ReachyMini, state: ReachyMiniState, logger: logging.Logger):
        self.mini = mini
        self._state = state
        self.logger = logger

        self._head_tracker = HeadTracker(mini)

    async def move(
            self,
            x: float = 0,
            y: float = 0,
            z: float = 0,
            roll: float = 0,
            pitch: float = 0,
            yaw: float = 0,
            duration: float = 0.5,
    ):
        """Move to a pose in 6D space (position and orientation).

        Args:
            x (float): X coordinate of the position. range: [-1.5cm, +2.5cm]
            y (float): Y coordinate of the position. range: [-4cm, +4cm]
            z (float): Z coordinate of the position. range: [-4cm, +2.5cm]
            roll (float): Roll angle. range(degree): [-40, +40]
            pitch (float): Pitch angle. range(degree): [-40, +40]
            yaw (float): Yaw angle. range(degree): [-60, +60]
            duration (float): Duration in seconds.
        """
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(x, y, z, roll, pitch, yaw),
            duration=duration
        ))

    async def reset(self, duration: float = 0.5):
        """
        Reset the head, watching forward
        """
        self._state.tracking.clear()
        await self.mini.async_play_move(move=HeadMove(
            self.mini.get_current_head_pose(),
            create_head_pose(),
            duration=duration,
        ))

    async def start_tracking_face(self, tracking_id: int=-1):
        """
        Keep gazing at the user.
        """
        self._state.tracking.set()
        self._head_tracker.set_tracking_id(tracking_id)

    async def stop_tracking_face(self):
        self._state.tracking.clear()
        self._head_tracker.set_tracking_id(-1)

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini_head__")
        if self._state.tracking.is_set() and self._head_tracker.face_tracking_positions:
            msg.with_content(
                Text(text=f"You are keep looking user with head tracking"),
                Text(text=f"Head tracking information is {stringify_positions(self._head_tracker.face_tracking_positions)}"),
                Text(text=f"Current tracking id is {self._head_tracker.current_tracking_id}")
            )

        msg.with_content(
            Text(text=f"Current head pose is {self.mini.get_current_head_pose()}")
        )

        return [msg]

    async def on_policy_run(self):
        self.logger.info(f"Running on-policy run, waken is {self._state.waken.is_set()}")
        if self._state.waken.is_set() and self._state.tracking.is_set():
            self.logger.info("self._head_tracker.enabled.set()")
            self._head_tracker.enabled.set()

    async def on_policy_pause(self):
        self.logger.info("Running on-policy pause")
        self._head_tracker.enabled.clear()

    def as_channel(self) -> PyChannel:
        head = PyChannel(name="head", block=True)

        head.build.with_context_messages(self.context_messages)
        head.build.on_policy_run(self.on_policy_run)
        head.build.on_policy_pause(self.on_policy_pause)

        head.build.command()(self.move)
        head.build.command()(self.reset)
        head.build.command()(self.start_tracking_face)
        head.build.command()(self.stop_tracking_face)

        return head

    async def bootstrap(self):
        await self._head_tracker.start()

    async def aclose(self):
        await self._head_tracker.stop()