from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class PolicyRunner(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZeroPolicy(PolicyRunner):
    def __init__(self, num_actions: int):
        self._num_actions = num_actions

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self._num_actions, dtype=np.float32)


class DemoHumanoidPolicy(PolicyRunner):
    """Open-loop demo policy for M0 pipeline bring-up.

    It is not a learned locomotion policy. The goal is only to make the
    Gymnasium humanoid visibly move when a velocity command is set, so the
    CTML -> channel -> controller -> MuJoCo chain can be verified locally.
    """

    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._phase = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        cmd = np.asarray(obs[6:9] if obs.size >= 9 else [0.0, 0.0, 0.0], dtype=np.float32)
        vx, vy, vyaw = float(cmd[0]), float(cmd[1]), float(cmd[2])
        mag = abs(vx) + abs(vy) + abs(vyaw)
        if mag < 1e-3:
            self._phase = 0.0
            return np.zeros(self._num_actions, dtype=np.float32)

        self._phase += 0.18 + min(mag, 1.0) * 0.12
        s = float(np.sin(self._phase))
        c = float(np.cos(self._phase))
        amp = min(0.45, 0.18 + abs(vx) * 0.35 + abs(vy) * 0.15 + abs(vyaw) * 0.15)

        action = np.zeros(self._num_actions, dtype=np.float32)
        if self._num_actions >= 17:
            action[0] = np.clip(vyaw * 0.35, -0.5, 0.5)   # abdomen_z
            action[1] = np.clip(vy * 0.25, -0.4, 0.4)     # abdomen_y
            action[2] = np.clip(-abs(vx) * 0.1, -0.3, 0.0)  # abdomen_x

            action[3] = amp * s                           # right_hip_x
            action[4] = np.clip(vyaw * 0.35, -0.5, 0.5)   # right_hip_z
            action[5] = np.clip(vx * 0.25, -0.4, 0.4)     # right_hip_y
            action[6] = -amp * 0.8 * c - 0.08             # right_knee

            action[7] = -amp * s                          # left_hip_x
            action[8] = np.clip(-vyaw * 0.35, -0.5, 0.5)  # left_hip_z
            action[9] = np.clip(vx * 0.25, -0.4, 0.4)     # left_hip_y
            action[10] = amp * 0.8 * c - 0.08             # left_knee

            arm_swing = amp * 0.55
            action[11] = -arm_swing * s
            action[12] = arm_swing * 0.35
            action[13] = -arm_swing * 0.5 * c
            action[14] = arm_swing * s
            action[15] = -arm_swing * 0.35
            action[16] = arm_swing * 0.5 * c
            return np.clip(action, -1.0, 1.0)

        idx = np.arange(self._num_actions, dtype=np.float32)
        action = 0.3 * np.sin(self._phase + idx * 0.7)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


class TorchScriptPolicy(PolicyRunner):
    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"TorchScript policy not found: {self._path}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for TorchScriptPolicy") from exc

        self._torch = torch
        self._module = torch.jit.load(str(self._path), map_location="cpu")
        self._module.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with self._torch.no_grad():
            action = self._module(tensor)
        if isinstance(action, (tuple, list)):
            action = action[0]
        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)


class SB3Policy(PolicyRunner):
    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"SB3 policy not found: {self._path}")

        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for SB3Policy") from exc

        self._model = PPO.load(str(self._path))

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _state = self._model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return np.asarray(action, dtype=np.float32)
