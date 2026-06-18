from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from control.interface import BaseState, VelocityCommand


@dataclass(slots=True)
class PolicySpec:
    kind: str = "zero"
    path: str = ""


@dataclass(slots=True)
class SimConfig:
    backend: str
    num_obs: int
    num_actions: int
    control_hz: float = 50.0
    sim_dt: float = 0.02
    decimation: int = 1
    render: bool = True
    env_id: str = "Humanoid-v4"
    model_path: str = ""
    seed: int = 0
    action_scale: float = 1.0
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 1.0
    command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    kps: list[float] = field(default_factory=list)
    kds: list[float] = field(default_factory=list)
    cmd_init: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_linear_speed: float = 0.6
    max_yaw_speed: float = 0.8
    default_angles: list[float] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    note: str = ""
    policy: PolicySpec = field(default_factory=PolicySpec)


def load_sim_config(path: str | Path) -> SimConfig:
    file = Path(path)
    data = yaml.safe_load(file.read_text()) or {}

    policy_data = data.get("policy") or {}
    command_scale = tuple(float(x) for x in (data.get("command_scale") or [1.0, 1.0, 1.0]))
    if len(command_scale) != 3:
        raise ValueError("command_scale must contain exactly 3 values")
    cmd_init = tuple(float(x) for x in (data.get("cmd_init") or [0.0, 0.0, 0.0]))
    if len(cmd_init) != 3:
        raise ValueError("cmd_init must contain exactly 3 values")

    return SimConfig(
        backend=data["backend"],
        num_obs=int(data["num_obs"]),
        num_actions=int(data["num_actions"]),
        control_hz=float(data.get("control_hz", 50.0)),
        sim_dt=float(data.get("sim_dt", 0.02)),
        decimation=int(data.get("decimation", 1)),
        render=bool(data.get("render", True)),
        env_id=str(data.get("env_id", "Humanoid-v4")),
        model_path=str(data.get("model_path", "")),
        seed=int(data.get("seed", 0)),
        action_scale=float(data.get("action_scale", 1.0)),
        ang_vel_scale=float(data.get("ang_vel_scale", 1.0)),
        dof_pos_scale=float(data.get("dof_pos_scale", 1.0)),
        dof_vel_scale=float(data.get("dof_vel_scale", 1.0)),
        command_scale=(command_scale[0], command_scale[1], command_scale[2]),
        kps=[float(x) for x in (data.get("kps") or [])],
        kds=[float(x) for x in (data.get("kds") or [])],
        cmd_init=(cmd_init[0], cmd_init[1], cmd_init[2]),
        max_linear_speed=float(data.get("max_linear_speed", 0.6)),
        max_yaw_speed=float(data.get("max_yaw_speed", 0.8)),
        default_angles=list(data.get("default_angles") or []),
        joint_names=list(data.get("joint_names") or []),
        note=str(data.get("note", "")),
        policy=PolicySpec(
            kind=str(policy_data.get("kind", "zero")),
            path=str(policy_data.get("path", "")),
        ),
    )


def clamp_command(cmd: VelocityCommand, cfg: SimConfig) -> VelocityCommand:
    max_v = abs(cfg.max_linear_speed)
    max_w = abs(cfg.max_yaw_speed)
    return VelocityCommand(
        vx=max(-max_v, min(max_v, cmd.vx)),
        vy=max(-max_v, min(max_v, cmd.vy)),
        vyaw=max(-max_w, min(max_w, cmd.vyaw)),
    )


def build_observation(
    base_state: BaseState,
    cmd: VelocityCommand,
    last_action: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    cmd = clamp_command(cmd, cfg)
    values = [
        *base_state.lin_vel,
        *base_state.ang_vel,
        cmd.vx * cfg.command_scale[0],
        cmd.vy * cfg.command_scale[1],
        cmd.vyaw * cfg.command_scale[2],
        base_state.height,
        base_state.yaw,
    ]
    values.extend(np.asarray(last_action, dtype=np.float32).tolist())
    obs = np.asarray(values, dtype=np.float32)
    if obs.size >= cfg.num_obs:
        return obs[: cfg.num_obs]
    padded = np.zeros(cfg.num_obs, dtype=np.float32)
    padded[: obs.size] = obs
    return padded


def action_to_ctrl(action: np.ndarray, cfg: SimConfig) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size >= cfg.num_actions:
        clipped = action[: cfg.num_actions]
    else:
        clipped = np.zeros(cfg.num_actions, dtype=np.float32)
        clipped[: action.size] = action
    return np.clip(clipped * cfg.action_scale, -1.0, 1.0)


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float32)
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


def build_g1_mujoco_observation(
    qpos: np.ndarray,
    qvel: np.ndarray,
    cmd: VelocityCommand,
    last_action: np.ndarray,
    step_count: int,
    cfg: SimConfig,
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    last_action = np.asarray(last_action, dtype=np.float32)
    default_angles = np.asarray(cfg.default_angles, dtype=np.float32)
    cmd = clamp_command(cmd, cfg)

    qj = (qpos[7 : 7 + cfg.num_actions] - default_angles) * cfg.dof_pos_scale
    dqj = qvel[6 : 6 + cfg.num_actions] * cfg.dof_vel_scale
    quat = qpos[3:7]
    omega = qvel[3:6] * cfg.ang_vel_scale
    gravity_orientation = get_gravity_orientation(quat)

    cmd_arr = np.array([cmd.vx, cmd.vy, cmd.vyaw], dtype=np.float32) * np.asarray(cfg.command_scale, dtype=np.float32)
    period = 0.8
    count = step_count * cfg.sim_dt
    phase = (count % period) / period
    clock = np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float32)

    obs = np.zeros(cfg.num_obs, dtype=np.float32)
    obs[:3] = omega
    obs[3:6] = gravity_orientation
    obs[6:9] = cmd_arr
    obs[9 : 9 + cfg.num_actions] = qj
    obs[9 + cfg.num_actions : 9 + 2 * cfg.num_actions] = dqj
    obs[9 + 2 * cfg.num_actions : 9 + 3 * cfg.num_actions] = last_action[: cfg.num_actions]
    obs[9 + 3 * cfg.num_actions : 9 + 3 * cfg.num_actions + 2] = clock
    return obs


def action_to_target_dof_pos(action: np.ndarray, cfg: SimConfig) -> np.ndarray:
    default_angles = np.asarray(cfg.default_angles, dtype=np.float32)
    scaled = action_to_ctrl(action, cfg)
    return scaled + default_angles


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def snapshot_to_dict(base_state: BaseState) -> dict[str, Any]:
    return {
        "x": round(base_state.x, 3),
        "y": round(base_state.y, 3),
        "yaw": round(base_state.yaw, 3),
        "height": round(base_state.height, 3),
        "lin_vel": [round(v, 3) for v in base_state.lin_vel],
        "ang_vel": [round(v, 3) for v in base_state.ang_vel],
        "sim_time": round(base_state.sim_time, 3),
        "status": base_state.status,
        "backend": base_state.backend,
        "note": base_state.note,
    }
