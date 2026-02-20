import asyncio
import numpy as np
import numpy.typing as npt
from reachy_mini.motion.move import Move

from reachy_mini.utils.interpolation import (
    time_trajectory,
)


class ReachyMiniState:
    def __init__(self):
        self.waken = asyncio.Event()
        self.tracking = asyncio.Event()
        self.twisting = asyncio.Event()
        self.start_body_yaw = 0.0


class BodyYawMove(Move):

    def __init__(
        self,
        start_body_yaw: float,
        target_body_yaw: float | None,
        duration: float,
    ):
        self.start_body_yaw = start_body_yaw
        self.target_body_yaw = (
            target_body_yaw if target_body_yaw is not None else start_body_yaw
        )

        self._duration = duration

    @property
    def duration(self) -> float:
        """Duration of the goto in seconds."""
        return self._duration

    def evaluate(
        self, t: float
    ) -> tuple[
        npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, float | None
    ]:
        """Evaluate the goto at time t."""
        interp_time = time_trajectory(t / self.duration)

        interp_body_yaw_joint = (
            self.start_body_yaw
            + (self.target_body_yaw - self.start_body_yaw) * interp_time
        )

        return None, None, np.deg2rad(interp_body_yaw_joint)
