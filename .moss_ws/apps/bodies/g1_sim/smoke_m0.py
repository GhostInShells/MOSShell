from __future__ import annotations

import asyncio

from ghoshell_moss.core.blueprint.channel_builder import test_channel

from control.mujoco_controller import MujocoVelocityController
from control.obs import load_sim_config
from control.policy import DemoHumanoidPolicy
from g1_sim_channel import build_g1_sim_channel


async def main() -> None:
    cfg = load_sim_config("config/humanoid_v4.yaml")
    cfg.render = False

    controller = MujocoVelocityController(cfg, DemoHumanoidPolicy(cfg.num_actions))
    controller.start()
    channel = build_g1_sim_channel(controller)
    try:
        await asyncio.sleep(1.0)
        print("WARMUP_SUMMARY", controller.get_snapshot().summary())

        tasks = await test_channel(
            channel,
            ctml='<bodies_g1_sim:move vx="0.35" vy="0.0" vyaw="0.0" duration="0" />',
            timeout=8.0,
        )
        print("TASK_COUNT", len(tasks))
        for idx, task in enumerate(tasks, start=1):
            try:
                result = await task
            except Exception as exc:
                result = f"ERR: {exc!r}"
            print(f"RESULT_{idx}", result)

        await asyncio.sleep(1.5)
        moving_snap = controller.get_snapshot()
        print("MOVING_SUMMARY", moving_snap.summary())
        print("MOVING_SIM_TIME", round(moving_snap.base_state.sim_time, 3))
        print("MOVING_OBS_DIM", moving_snap.observation.size)

        tasks = await test_channel(
            channel,
            ctml="""
<bodies_g1_sim:state />
<bodies_g1_sim:stop />
""",
            timeout=8.0,
        )
        print("TASK_COUNT_2", len(tasks))
        for idx, task in enumerate(tasks, start=1):
            try:
                result = await task
            except Exception as exc:
                result = f"ERR: {exc!r}"
            print(f"RESULT2_{idx}", result)

        await asyncio.sleep(0.3)
        final_snap = controller.get_snapshot()
        print("FINAL_SUMMARY", final_snap.summary())
        print("FINAL_SIM_TIME", round(final_snap.base_state.sim_time, 3))
        print("FINAL_OBS_DIM", final_snap.observation.size)
    finally:
        controller.close()


if __name__ == "__main__":
    asyncio.run(main())
