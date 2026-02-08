import asyncio
import time

import live2d.v3 as live2d

from ghoshell_moss.core.py_channel import PyChannel

from .motions import open_close

eye_chan = PyChannel(name="eye")


@eye_chan.build.command()
async def gaze(x: float = 0.0, y: float = 0.0, duration: float = 1.5, speed: float = 1.0):
    """
     gaze 眼球

    :param x: 眼球 X 坐标，-1 到 1 之间的浮点数，默认 0.0
    :param y: 眼球 Y 坐标，-1 到 1 之间的浮点数，默认 0.0
    :param duration:  执行时间，保持眼球在目标位置时间
    :param speed:  眼球移动速度，正常速度默认1.0
    """
    PARAM_BALL_X = "PARAM_EYE_BALL_X"
    PARAM_BALL_Y = "PARAM_EYE_BALL_Y"
    model = eye_chan.broker.container.force_fetch(live2d.LAppModel)

    # 获取当前眼球位置
    x_index = model.GetParamIds().index(PARAM_BALL_X)
    y_index = model.GetParamIds().index(PARAM_BALL_Y)
    current_x = model.GetParameterValue(x_index)
    current_y = model.GetParameterValue(y_index)

    # 第一阶段：移动到目标位置（根据speed参数控制速度）
    move_duration = (
        abs(x - current_x) / speed if abs(x - current_x) > abs(y - current_y) else abs(y - current_y) / speed
    )
    move_duration = max(move_duration, 0.1)  # 确保至少有0.1秒的移动时间

    start_time = time.time()
    while time.time() - start_time < move_duration:
        progress = (time.time() - start_time) / move_duration
        target_x = current_x + (x - current_x) * progress
        target_y = current_y + (y - current_y) * progress
        model.SetParameterValue(PARAM_BALL_X, target_x)
        model.SetParameterValue(PARAM_BALL_Y, target_y)
        await asyncio.sleep(0.016)

    # 确保精确到达目标位置
    model.SetParameterValue(PARAM_BALL_X, x)
    model.SetParameterValue(PARAM_BALL_Y, y)

    # 第二阶段：在目标位置停留指定时间
    await asyncio.sleep(duration)

    # 第三阶段：移回原点
    origin_x, origin_y = 0.0, 0.0
    back_duration = abs(x - origin_x) / speed if abs(x - origin_x) > abs(y - origin_y) else abs(y - origin_y) / speed
    back_duration = max(back_duration, 0.1)  # 确保至少有0.1秒的移动时间

    start_time = time.time()
    while time.time() - start_time < back_duration:
        progress = (time.time() - start_time) / back_duration
        target_x = x + (origin_x - x) * progress
        target_y = y + (origin_y - y) * progress
        model.SetParameterValue(PARAM_BALL_X, target_x)
        model.SetParameterValue(PARAM_BALL_Y, target_y)
        await asyncio.sleep(0.016)

    # 确保精确回到原点
    model.SetParameterValue(PARAM_BALL_X, origin_x)
    model.SetParameterValue(PARAM_BALL_Y, origin_y)


eye_left_chan = eye_chan.new_child(name="eye_left")


@eye_left_chan.build.command()
async def blink(duration: float = 1.5, speed: float = 1.0, max_open: float = 1.0, min_open: float = 0.0):
    """
    眨左眼

    :param duration: 动画持续时间，单位秒
    :param speed: 开合速度参数，值越大速度越快，默认1.0
    :param max_open: 最大睁开程度，0 到 1 之间的浮点数，默认为1.0
    :param min_open: 最小睁开程度，0 到 1 之间的浮点数，默认为0.0
    """
    PARAM = "PARAM_EYE_L_OPEN"
    model = eye_left_chan.broker.container.force_fetch(live2d.LAppModel)

    # 调用通用动画函数
    final_value = await open_close(
        model=model,
        param_name=PARAM,
        duration=duration,
        speed=speed,
        max_value=max_open,
        min_value=min_open,
        initial_direction="close",  # 眨眼从闭合开始
    )

    # 确保最终状态是完全睁开
    model.SetParameterValue(PARAM, 1.0)


eye_right_chan = eye_chan.new_child(name="eye_right")


@eye_right_chan.build.command()
async def right_blink(
    duration: float = 1.5,
    speed: float = 1.0,
    max_open: float = 1.0,
    min_open: float = 0.0,
):
    """
    眨右眼

    :param duration: 动画持续时间，单位秒
    :param speed: 开合速度参数，值越大速度越快，默认1.0
    :param max_open: 最大睁开程度，0 到 1 之间的浮点数，默认为1.0
    :param min_open: 最小睁开程度，0 到 1 之间的浮点数，默认为0.0
    """
    PARAM = "PARAM_EYE_R_OPEN"
    model = eye_right_chan.broker.container.force_fetch(live2d.LAppModel)

    # 调用通用动画函数
    final_value = await open_close(
        model=model,
        param_name=PARAM,
        duration=duration,
        speed=speed,
        max_value=max_open,
        min_value=min_open,
        initial_direction="close",  # 眨眼从闭合开始
    )

    # 确保最终状态是完全睁开
    model.SetParameterValue(PARAM, 1.0)
