from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from control.interface import BaseState, ControllerHealth, ControllerSnapshot, VelocityCommand, VelocityRobotController
from control.obs import (
    SimConfig,
    action_to_target_dof_pos,
    build_g1_mujoco_observation,
    build_observation,
    clamp_command,
    pd_control,
)
from control.policy import PolicyRunner


class MujocoVelocityController(VelocityRobotController):
    """Velocity controller with two backends:

    - `gym_humanoid_v4`: M0 pipeline validation using Gymnasium Humanoid-v4.
    - `mujoco_g1`: direct MuJoCo model path reserved for G1 M1 integration.
    """

    def __init__(self, cfg: SimConfig, policy: PolicyRunner):
        self._cfg = cfg
        self._policy = policy
        self._logger = logging.getLogger(__name__)
        self._cmd = VelocityCommand()
        self._cmd_lock = threading.Lock()
        self._state = BaseState(backend=cfg.backend, note=cfg.note)
        self._state_lock = threading.Lock()
        self._observation = np.zeros(cfg.num_obs, dtype=np.float32)
        self._obs_lock = threading.Lock()
        self._last_action = np.zeros(cfg.num_actions, dtype=np.float32)
        self._health = ControllerHealth(phase="booting")
        self._health_lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._reset_requested = threading.Event()
        self._reset_requested.set()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._last_error = ""

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._running = False

    def closed(self) -> bool:
        return self._stop_event.is_set()

    def set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        with self._cmd_lock:
            self._cmd = clamp_command(VelocityCommand(vx=vx, vy=vy, vyaw=vyaw), self._cfg)

    def stop(self) -> None:
        self.set_velocity_command(0.0, 0.0, 0.0)

    def stand(self) -> None:
        self.stop()
        health = self.get_health()
        if health.fallen or health.phase in ("booting", "resetting"):
            self._reset_requested.set()

    def reset_pose(self) -> None:
        self.stop()
        self._reset_requested.set()

    def get_observation(self) -> np.ndarray:
        with self._obs_lock:
            return self._observation.copy()

    def get_base_state(self) -> BaseState:
        with self._state_lock:
            state = self._state
            return BaseState(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                height=state.height,
                lin_vel=state.lin_vel,
                ang_vel=state.ang_vel,
                sim_time=state.sim_time,
                status=state.status,
                backend=state.backend,
                note=state.note,
            )

    def get_snapshot(self) -> ControllerSnapshot:
        with self._cmd_lock:
            cmd = VelocityCommand(vx=self._cmd.vx, vy=self._cmd.vy, vyaw=self._cmd.vyaw)
        return ControllerSnapshot(
            running=self._running and not self._stop_event.is_set(),
            backend=self._cfg.backend,
            command=cmd,
            base_state=self.get_base_state(),
            observation=self.get_observation(),
            last_error=self._last_error,
        )

    def get_health(self) -> ControllerHealth:
        with self._health_lock:
            return ControllerHealth(
                ready_for_motion=self._health.ready_for_motion,
                fallen=self._health.fallen,
                phase=self._health.phase,
                reason=self._health.reason,
                base_height=self._health.base_height,
            )

    def _control_loop(self) -> None:
        try:
            if self._cfg.backend == "gym_humanoid_v4":
                self._run_gym_humanoid()
            elif self._cfg.backend == "mujoco_g1":
                self._run_direct_mujoco()
            else:
                raise ValueError(f"Unsupported backend: {self._cfg.backend}")
        except Exception as exc:
            self._last_error = str(exc)
            self._logger.exception("g1_sim control loop failed")
            with self._state_lock:
                self._state.status = "error"
                self._state.note = str(exc)
            self._set_health(ControllerHealth(
                ready_for_motion=False,
                fallen=False,
                phase="error",
                reason=str(exc),
                base_height=self._state.height,
            ))
        finally:
            self._running = False

    def _run_gym_humanoid(self) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError("gymnasium[mujoco] is required for backend gym_humanoid_v4") from exc

        render_mode = "human" if self._cfg.render else None
        env = gym.make(self._cfg.env_id, render_mode=render_mode)
        try:
            obs, _info = env.reset(seed=self._cfg.seed)
            action_space = env.action_space
            action_size = int(np.prod(action_space.shape))
            if action_size != self._cfg.num_actions:
                self._logger.warning(
                    "config num_actions=%s but env action size=%s; env size wins",
                    self._cfg.num_actions,
                    action_size,
                )
                self._last_action = np.zeros(action_size, dtype=np.float32)

            dt = 1.0 / max(self._cfg.control_hz, 1.0)
            sim_time = 0.0
            while not self._stop_event.is_set():
                loop_started = time.perf_counter()
                with self._cmd_lock:
                    cmd = VelocityCommand(vx=self._cmd.vx, vy=self._cmd.vy, vyaw=self._cmd.vyaw)

                base_state = self._integrate_base_state(dt, sim_time, cmd)
                policy_obs = build_observation(base_state, cmd, self._last_action, self._cfg)
                action = self._policy.act(policy_obs)
                env_action = np.asarray(action, dtype=np.float32).reshape(-1)
                if env_action.size != action_size:
                    resized = np.zeros(action_size, dtype=np.float32)
                    resized[: min(action_size, env_action.size)] = env_action[: min(action_size, env_action.size)]
                    env_action = resized
                env_action = np.clip(env_action, action_space.low, action_space.high)

                obs, _reward, terminated, truncated, _info = env.step(env_action)
                self._last_action = env_action
                sim_time += dt

                with self._obs_lock:
                    self._observation = policy_obs
                with self._state_lock:
                    self._state = base_state
                    self._state.status = "moving" if not cmd.is_zero() else "idle"
                    self._state.note = "M0 pipeline backend: Gymnasium Humanoid-v4"
                self._set_health(ControllerHealth(
                    ready_for_motion=True,
                    fallen=False,
                    phase="moving" if not cmd.is_zero() else "standing",
                    reason="",
                    base_height=base_state.height,
                ))

                if self._cfg.render:
                    env.render()

                if terminated or truncated:
                    obs, _info = env.reset()
                    self._last_action = np.zeros_like(self._last_action)

                elapsed = time.perf_counter() - loop_started
                sleep_for = dt - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            env.close()

    def _run_direct_mujoco(self) -> None:
        model_path = Path(self._cfg.model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent.parent / model_path
        if not model_path.exists():
            raise FileNotFoundError(
                "G1 MuJoCo model is missing. "
                f"Expected model at {model_path}. "
                "Copy assets from unitree_rl_gym or mujoco_menagerie first."
            )

        try:
            import mujoco
            import mujoco.viewer as mj_viewer
        except ImportError as exc:
            raise RuntimeError("mujoco is required for backend mujoco_g1") from exc

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        model.opt.timestep = self._cfg.sim_dt
        viewer = None
        camera_body_id = -1
        camera_lookat = np.zeros(3, dtype=np.float64)
        camera_height_offset = 0.12
        camera_locked_lookat_z = 0.0
        camera_follow_alpha = 0.06
        camera_deadzone_xy = np.array([0.32, 0.24], dtype=np.float64)
        if self._cfg.render:
            try:
                viewer = mj_viewer.launch_passive(model, data)
            except Exception:
                self._logger.exception("launch_passive viewer failed; continuing headless")
                viewer = None
        try:
            camera_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        except Exception:
            camera_body_id = 1 if model.nbody > 1 else -1
        if camera_body_id >= 0:
            camera_lookat[:] = np.asarray(data.xpos[camera_body_id], dtype=np.float64)
            camera_locked_lookat_z = float(camera_lookat[2] + camera_height_offset)
            camera_lookat[2] = camera_locked_lookat_z
        if viewer is not None:
            try:
                # Keep the robot centered with a stable three-quarter follow view.
                viewer.cam.distance = 2.4
                viewer.cam.azimuth = 140.0
                viewer.cam.elevation = -18.0
                if camera_body_id >= 0:
                    viewer.cam.lookat[:] = camera_lookat
            except Exception:
                self._logger.exception("failed to configure viewer follow camera")

        default_angles = np.asarray(self._cfg.default_angles, dtype=np.float32)
        if default_angles.size:
            qpos_tail = default_angles.astype(np.float64)
            count = min(qpos_tail.size, max(0, data.qpos.size - 7))
            if count > 0:
                data.qpos[7:7 + count] = qpos_tail[:count]
                mujoco.mj_forward(model, data)

        kps = np.asarray(self._cfg.kps, dtype=np.float32)
        kds = np.asarray(self._cfg.kds, dtype=np.float32)
        stand_kps = kps * 1.5
        stand_kds = kds * 1.5
        if not default_angles.size:
            raise ValueError("M1 requires non-empty default_angles")
        if kps.size != self._cfg.num_actions or kds.size != self._cfg.num_actions:
            raise ValueError("M1 requires kps/kds length to equal num_actions")

        target_dof_pos = default_angles.copy()
        step_counter = 0
        dt = self._cfg.sim_dt
        control_decimation = max(self._cfg.decimation, 1)
        stand_height = float(model.body_pos[1][2]) if model.nbody > 1 else 0.793
        ready_after_reset_secs = 1.0
        motion_ready_at = 0.0
        prev_phase = ""
        prev_fallen = None
        while not self._stop_event.is_set():
            loop_started = time.perf_counter()
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                if data.qpos.size >= 7:
                    data.qpos[:] = 0.0
                    data.qpos[2] = stand_height
                    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                if data.qvel.size:
                    data.qvel[:] = 0.0
                if data.ctrl.size:
                    data.ctrl[:] = 0.0
                count = min(default_angles.size, max(0, data.qpos.size - 7))
                if count > 0:
                    data.qpos[7:7 + count] = default_angles[:count].astype(np.float64)
                mujoco.mj_forward(model, data)
                target_dof_pos = default_angles.copy()
                self._last_action = np.zeros(self._cfg.num_actions, dtype=np.float32)
                step_counter = 0
                motion_ready_at = float(data.time) + ready_after_reset_secs

            with self._cmd_lock:
                requested_cmd = VelocityCommand(vx=self._cmd.vx, vy=self._cmd.vy, vyaw=self._cmd.vyaw)

            base_height = float(data.qpos[2]) if data.qpos.size > 2 else 0.0
            fallen = base_height < stand_height * 0.55
            in_reset_window = float(data.time) < motion_ready_at
            if fallen:
                effective_cmd = VelocityCommand()
                phase = "fallen"
                reason = "robot has fallen; call reset() or stand() to recover"
                self.stop()
            elif in_reset_window:
                effective_cmd = VelocityCommand()
                phase = "resetting"
                reason = "resetting to stand pose"
            else:
                effective_cmd = requested_cmd
                phase = "moving" if not effective_cmd.is_zero() else "standing"
                reason = ""

            active_kps = kps
            active_kds = kds
            policy_obs = self.get_observation()
            if step_counter % control_decimation == 0:
                policy_obs = build_g1_mujoco_observation(
                    np.asarray(data.qpos, dtype=np.float32),
                    np.asarray(data.qvel, dtype=np.float32),
                    effective_cmd,
                    self._last_action,
                    step_counter,
                    self._cfg,
                )
                if phase != "fallen":
                    action = np.asarray(self._policy.act(policy_obs), dtype=np.float32).reshape(-1)
                    if action.size != self._cfg.num_actions:
                        resized = np.zeros(self._cfg.num_actions, dtype=np.float32)
                        resized[: min(action.size, self._cfg.num_actions)] = action[: min(action.size, self._cfg.num_actions)]
                        action = resized
                    self._last_action = action
                    target_dof_pos = action_to_target_dof_pos(action, self._cfg)
                else:
                    self._last_action = np.zeros(self._cfg.num_actions, dtype=np.float32)
                    target_dof_pos = default_angles.copy()
            if phase == "fallen":
                active_kps = stand_kps
                active_kds = stand_kds

            tau = pd_control(
                target_dof_pos,
                np.asarray(data.qpos[7 : 7 + self._cfg.num_actions], dtype=np.float32),
                active_kps,
                np.zeros_like(active_kds),
                np.asarray(data.qvel[6 : 6 + self._cfg.num_actions], dtype=np.float32),
                active_kds,
            )
            if data.ctrl.size > 0:
                limit = min(tau.size, data.ctrl.size)
                data.ctrl[:limit] = tau[:limit]

            mujoco.mj_step(model, data)
            step_counter += 1
            if viewer is not None and viewer.is_running():
                if camera_body_id >= 0:
                    target_lookat = np.asarray(data.xpos[camera_body_id], dtype=np.float64).copy()
                    target_lookat[2] = camera_locked_lookat_z
                    desired_lookat = camera_lookat.copy()
                    delta_xy = target_lookat[:2] - camera_lookat[:2]
                    for axis in range(2):
                        if abs(delta_xy[axis]) > camera_deadzone_xy[axis]:
                            desired_lookat[axis] = target_lookat[axis] - np.sign(delta_xy[axis]) * camera_deadzone_xy[axis]
                    camera_lookat[:] = (1.0 - camera_follow_alpha) * camera_lookat + camera_follow_alpha * desired_lookat
                    camera_lookat[2] = camera_locked_lookat_z
                    viewer.cam.lookat[:] = camera_lookat
                viewer.sync()

            base_state = BaseState(
                x=float(data.qpos[0]) if data.qpos.size > 0 else 0.0,
                y=float(data.qpos[1]) if data.qpos.size > 1 else 0.0,
                yaw=float(data.qpos[6]) if data.qpos.size > 6 else 0.0,
                height=float(data.qpos[2]) if data.qpos.size > 2 else 0.0,
                lin_vel=(
                    float(data.qvel[0]) if data.qvel.size > 0 else 0.0,
                    float(data.qvel[1]) if data.qvel.size > 1 else 0.0,
                    float(data.qvel[2]) if data.qvel.size > 2 else 0.0,
                ),
                ang_vel=(
                    float(data.qvel[3]) if data.qvel.size > 3 else 0.0,
                    float(data.qvel[4]) if data.qvel.size > 4 else 0.0,
                    float(data.qvel[5]) if data.qvel.size > 5 else 0.0,
                ),
                sim_time=float(data.time),
                status=phase,
                backend=self._cfg.backend,
                note="M1 backend: unitree_rl_gym-style MuJoCo deploy loop" if not reason else f"M1 backend: {reason}",
            )

            with self._obs_lock:
                self._observation = policy_obs
            with self._state_lock:
                self._state = base_state
            self._set_health(ControllerHealth(
                ready_for_motion=phase in ("standing", "moving"),
                fallen=phase == "fallen",
                phase=phase,
                reason=reason,
                base_height=base_state.height,
            ))
            if phase != prev_phase or fallen != prev_fallen:
                prev_phase = phase
                prev_fallen = fallen

            elapsed = time.perf_counter() - loop_started
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        if viewer is not None:
            viewer.close()

    def _integrate_base_state(self, dt: float, sim_time: float, cmd: VelocityCommand) -> BaseState:
        with self._state_lock:
            prev = self._state
        yaw = prev.yaw + cmd.vyaw * dt
        x = prev.x + cmd.vx * dt
        y = prev.y + cmd.vy * dt
        return BaseState(
            x=x,
            y=y,
            yaw=yaw,
            height=prev.height,
            lin_vel=(cmd.vx, cmd.vy, 0.0),
            ang_vel=(0.0, 0.0, cmd.vyaw),
            sim_time=sim_time + dt,
            status="moving" if not cmd.is_zero() else "idle",
            backend=self._cfg.backend,
            note=prev.note,
        )

    def _set_health(self, health: ControllerHealth) -> None:
        with self._health_lock:
            self._health = health
