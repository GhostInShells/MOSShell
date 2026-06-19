from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return self.vx, self.vy, self.vyaw

    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.vyaw == 0.0


@dataclass(slots=True)
class BaseState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    height: float = 1.4
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sim_time: float = 0.0
    status: str = "idle"
    backend: str = ""
    note: str = ""


@dataclass(slots=True)
class ControllerSnapshot:
    running: bool
    backend: str
    command: VelocityCommand = field(default_factory=VelocityCommand)
    base_state: BaseState = field(default_factory=BaseState)
    observation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_error: str = ""

    def summary(self) -> str:
        cmd = self.command
        base = self.base_state
        moving = "yes" if not cmd.is_zero() else "no"
        parts = [
            f"backend={self.backend}",
            f"running={self.running}",
            f"moving={moving}",
            f"cmd=({cmd.vx:.2f}, {cmd.vy:.2f}, {cmd.vyaw:.2f})",
            f"pose=({base.x:.2f}, {base.y:.2f}, yaw={base.yaw:.2f})",
            f"status={base.status}",
        ]
        if self.last_error:
            parts.append(f"error={self.last_error}")
        if base.note:
            parts.append(f"note={base.note}")
        return " | ".join(parts)


@dataclass(slots=True)
class ControllerHealth:
    ready_for_motion: bool = False
    fallen: bool = False
    phase: str = "booting"
    reason: str = ""
    base_height: float = 0.0


class VelocityRobotController(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def closed(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stand(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_pose(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_base_state(self) -> BaseState:
        raise NotImplementedError

    @abstractmethod
    def get_snapshot(self) -> ControllerSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_health(self) -> ControllerHealth:
        raise NotImplementedError
