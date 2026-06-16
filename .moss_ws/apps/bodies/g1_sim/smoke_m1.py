from __future__ import annotations

import numpy as np

from control.interface import VelocityCommand
from control.obs import action_to_target_dof_pos, build_g1_mujoco_observation, load_sim_config, pd_control


def main() -> None:
    cfg = load_sim_config("config/g1.yaml")
    qpos = np.zeros(7 + cfg.num_actions, dtype=np.float32)
    qvel = np.zeros(6 + cfg.num_actions, dtype=np.float32)
    qpos[2] = 0.76
    qpos[3] = 1.0
    qpos[7 : 7 + cfg.num_actions] = np.array(cfg.default_angles, dtype=np.float32)

    cmd = VelocityCommand(0.4, 0.0, 0.1)
    last_action = np.zeros(cfg.num_actions, dtype=np.float32)
    obs = build_g1_mujoco_observation(qpos, qvel, cmd, last_action, step_count=10, cfg=cfg)

    action = np.linspace(-0.2, 0.2, cfg.num_actions, dtype=np.float32)
    target = action_to_target_dof_pos(action, cfg)
    tau = pd_control(
        target,
        qpos[7 : 7 + cfg.num_actions],
        np.array(cfg.kps, dtype=np.float32),
        np.zeros(cfg.num_actions, dtype=np.float32),
        qvel[6 : 6 + cfg.num_actions],
        np.array(cfg.kds, dtype=np.float32),
    )

    print("OBS_SHAPE", obs.shape)
    print("OBS_HEAD", np.round(obs[:12], 4).tolist())
    print("TARGET_HEAD", np.round(target[:6], 4).tolist())
    print("TAU_HEAD", np.round(tau[:6], 4).tolist())


if __name__ == "__main__":
    main()
