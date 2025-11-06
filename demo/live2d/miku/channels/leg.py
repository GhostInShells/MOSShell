from ghoshell_moss.channels.py_channel import PyChannel
import time
import asyncio
import live2d.v3 as live2d


left_leg_chan = PyChannel(name='left_leg')
right_leg_chan = PyChannel(name='right_leg')


@left_leg_chan.build.command()
async def move(duration: float = 1.5, angle: float = 10.0):
    """
    移动左腿到指定角度

    :param duration:  执行时间，时间越短角度变化越快
    :param angle:  身体垂直角度，10.0为最大角度约为身体垂直夹角30度，-10.0为最小角度
    """

    PARAM = "PARAM_LEG_L_Z"

    model = left_leg_chan.client.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index(PARAM)
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue(PARAM, target)
        await asyncio.sleep(0.016)
    
    value = model.GetParameterValue(model.GetParamIds().index(PARAM))


@right_leg_chan.build.command()
async def move(duration: float = 1.5, angle: float = -10.0):
    """
    移动右腿到指定角度

    :param duration:  执行时间，时间越短角度变化越快
    :param angle:  身体垂直角度，-10.0为最大角度约为身体垂直夹角30度，0.0为最小角度
    """
    PARAM = "PARAM_LEG_R_Z"
    model = right_leg_chan.client.container.force_fetch(live2d.LAppModel)
    index = model.GetParamIds().index(PARAM)    
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue(PARAM, target)
        await asyncio.sleep(0.016)
