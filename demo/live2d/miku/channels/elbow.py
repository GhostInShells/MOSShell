from ghoshell_moss.channels.py_channel import PyChannel
import time
import asyncio
import live2d.v3 as live2d


left_elbow_chan = PyChannel(name='left_elbow')
right_elbow_chan = PyChannel(name='right_elbow')


@left_elbow_chan.build.command()
async def move(duration: float = 1.5, angle: float = 0.0):
    """
    移动左臂到指定角度，负角度为下推到身体背后

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度，-30.0为最小角度
    """
    model = left_elbow_chan.client.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index('Param4')
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        print(f"progress: {progress}, target: {target}")
        model.SetParameterValue('Param4', target)
        await asyncio.sleep(0.016)

@right_elbow_chan.build.command()
async def move(duration: float = 1.5, angle: float = 0.0):
    """
    移动右臂到指定角度，负角度为下推到身体背后

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度，-30.0为最小角度
    """
    model = right_elbow_chan.client.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index('Param5')
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        print(f"progress: {progress}, target: {target}")
        model.SetParameterValue('Param5', target)
        await asyncio.sleep(0.016)