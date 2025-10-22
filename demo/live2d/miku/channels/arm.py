from ghoshell_moss.channels.py_channel import PyChannel
import time
import asyncio
import live2d.v3 as live2d


left_arm_chan = PyChannel(name='left_arm')
right_arm_chan = PyChannel(name='right_arm')


@left_arm_chan.build.command()
async def up(duration: float = 1.5):
    """
    抬起左手

    :param duration:  执行时间
    """
    model = left_arm_chan.client.container.force_fetch(live2d.LAppModel)
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = 8 * progress
        print(f"progress: {progress}, angle: {angle}")
        model.SetParameterValue('PARAM_ARM_L_01', angle)
        await asyncio.sleep(0.016)

@left_arm_chan.build.command()
async def down(duration: float = 1.5):
    """
    放下左手
    """
    model = left_arm_chan.client.container.force_fetch(live2d.LAppModel)
    start_angle = model.GetParameterValue('PARAM_ARM_L_01')
    if start_angle <= 0:
        return
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = start_angle - 8 * progress
        model.SetParameterValue('PARAM_ARM_L_01', angle)
        await asyncio.sleep(0.016)


@right_arm_chan.build.command()
async def up(duration: float = 1.5):
    """抬手"""
    model = right_arm_chan.client.container.force_fetch(live2d.LAppModel)
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = 8 * progress
        model.SetParameterValue('PARAM_ARM_R_01', angle)
        await asyncio.sleep(0.016)

@right_arm_chan.build.command()
async def down(duration: float = 1.5):
    """
    放下左手
    """
    model = right_arm_chan.client.container.force_fetch(live2d.LAppModel)
    start_angle = model.GetParameterValue('PARAM_ARM_R_01')
    if start_angle <= 0:
        return
    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        angle = start_angle - 8 * progress
        model.SetParameterValue('PARAM_ARM_R_01', angle)
        await asyncio.sleep(0.016)