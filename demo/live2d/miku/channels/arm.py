from ghoshell_moss.channels.py_channel import PyChannel
import time
import asyncio
import live2d.v3 as live2d


left_arm_chan = PyChannel(name='left_arm')
right_arm_chan = PyChannel(name='right_arm')


@left_arm_chan.build.command()
async def move(duration: float = 1.5, angle: float = 10.0):
    """
    移动左臂到指定角度

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度约为身体夹角45度，0.0为最小角度
    """
    model = left_arm_chan.client.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index('PARAM_ARM_L_01')
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        print(f"progress: {progress}, target: {target}")
        model.SetParameterValue('PARAM_ARM_L_01', target)
        await asyncio.sleep(0.016)


@right_arm_chan.build.command()
async def move(duration: float = 1.5, angle: float = 10.0):
    """
    移动右臂到指定角度

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度约为身体夹角45度，0.0为最小角度
    """
    model = right_arm_chan.client.container.force_fetch(live2d.LAppModel)
    index = model.GetParamIds().index('PARAM_ARM_R_01')
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        print(f"progress: {progress}, target: {target}")
        model.SetParameterValue('PARAM_ARM_R_01', target)
        await asyncio.sleep(0.016)
