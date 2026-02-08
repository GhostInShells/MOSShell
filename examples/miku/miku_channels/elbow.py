import asyncio
import time

import live2d.v3 as live2d

from ghoshell_moss.core.py_channel import PyChannel

left_elbow_chan = PyChannel(name="left_elbow")
right_elbow_chan = PyChannel(name="right_elbow")


@left_elbow_chan.build.command()
async def move(duration: float = 1.5, angle: float = 0.0):
    """
    以大臂为参考系，移动左小臂到指定夹角

    :param duration:  执行时间
    :param angle:  与大臂形成的夹角
                    - 正值：小臂向外伸展，最大角度为10.0
                    - 负值：小臂向内弯折，最小角度为-30.0
    """
    model = left_elbow_chan.broker.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index("Param4")
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue("Param4", target)
        await asyncio.sleep(0.016)


@right_elbow_chan.build.command()
async def right_move(duration: float = 1.5, angle: float = 0.0):
    """
    以大臂为参考系，移动右小臂到指定夹角

    :param duration:  执行时间
    :param angle:  与大臂形成的夹角
                    - 正值：小臂向外伸展，最大角度为10.0
                    - 负值：小臂向内弯折，最小角度为-30.0
    """
    model = right_elbow_chan.broker.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index("Param5")
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue("Param5", target)
        await asyncio.sleep(0.016)
