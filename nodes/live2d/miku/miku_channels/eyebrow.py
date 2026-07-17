import asyncio
import time

import live2d.v3 as live2d

from ghoshell_moss.core.py_channel import PyChannel

eyebrow_left_chan = PyChannel(name="eyebrow_left")
eyebrow_right_chan = PyChannel(name="eyebrow_right")


async def _smooth_move_eyebrow(
    model: live2d.LAppModel, param_x, param_y, param_angle, target_x, target_y, target_angle, speed
):
    """
    通用的眉毛平滑移动函数

    :param model: Live2D模型实例
    :param param_x: X参数名
    :param param_y: Y参数名
    :param param_angle: 角度参数名
    :param target_x: 目标X位置
    :param target_y: 目标Y位置
    :param target_angle: 目标角度
    :param speed: 移动速度，默认1.0为正常速度
    """
    # 获取当前眉毛位置和角度
    x_index = model.GetParamIds().index(param_x)
    y_index = model.GetParamIds().index(param_y)
    angle_index = model.GetParamIds().index(param_angle)

    current_x = model.GetParameterValue(x_index)
    current_y = model.GetParameterValue(y_index)
    current_angle = model.GetParameterValue(angle_index)

    # 计算移动持续时间，基于最大的距离和速度
    max_distance = max(abs(target_x - current_x), abs(target_y - current_y), abs(target_angle - current_angle))
    move_duration = max_distance / speed if speed > 0 else 0.1
    move_duration = max(move_duration, 0.1)  # 确保至少有0.1秒的移动时间

    # 平滑移动到目标位置
    start_time = time.time()
    while time.time() - start_time < move_duration:
        progress = (time.time() - start_time) / move_duration

        # 线性插值计算当前帧的位置和角度
        current_frame_x = current_x + (target_x - current_x) * progress
        current_frame_y = current_y + (target_y - current_y) * progress
        current_frame_angle = current_angle + (target_angle - current_angle) * progress

        # 更新模型参数
        model.SetParameterValue(param_x, current_frame_x)
        model.SetParameterValue(param_y, current_frame_y)
        model.SetParameterValue(param_angle, current_frame_angle)

        await asyncio.sleep(0.016)  # 约60FPS

    # 确保精确到达目标位置
    model.SetParameterValue(param_x, target_x)
    model.SetParameterValue(param_y, target_y)
    model.SetParameterValue(param_angle, target_angle)


@eyebrow_left_chan.build.command()
async def move(x: float = 0.0, y: float = 0.0, angle: float = 0.0, speed: float = 1.0):
    """
    平滑移动左眉

    :param x: 水平位置参数，-1 到 1 之间的浮点数，默认0.0，-1 表示最右，1 表示最左
    :param y: 垂直位置参数，-1 到 1 之间的浮点数，默认0.0，-1 表示最上，1 表示最下
    :param angle: 角度参数，-1 到 1 之间的浮点数，默认0.0，-1 表示左转最大角度，1 表示右转最大角度
    :param speed: 移动速度，正常速度默认1.0
    """
    PARAM_BROW_Y = "PARAM_BROW_L_Y"  # MAX 1.0 MIN -1.0
    PARAM_BROW_X = "PARAM_BROW_L_X"  # MAX 1.0 MIN -1.0
    PARAM_BROW_ANGLE = "PARAM_BROW_L_ANGLE"  # MAX 1.0 MIN -1.0

    model = eyebrow_left_chan.broker.container.force_fetch(live2d.LAppModel)

    # 调用通用平滑移动函数
    await _smooth_move_eyebrow(
        model=model,
        param_x=PARAM_BROW_X,
        param_y=PARAM_BROW_Y,
        param_angle=PARAM_BROW_ANGLE,
        target_x=x,
        target_y=y,
        target_angle=angle,
        speed=speed,
    )


@eyebrow_right_chan.build.command()
async def right_move(x: float = 0.0, y: float = 0.0, angle: float = 0.0, speed: float = 1.0):
    """
    平滑移动右眉

    :param x: 水平位置参数，-1 到 1 之间的浮点数，默认0.0，-1 表示最左，1 表示最右
    :param y: 垂直位置参数，-1 到 1 之间的浮点数，默认0.0，-1 表示最上，1 表示最下
    :param angle: 角度参数，-1 到 1 之间的浮点数，默认0.0，-1 表示右转最大角度，1 表示左转最大角度
    :param speed: 移动速度，正常速度默认1.0
    """
    PARAM_BROW_Y = "PARAM_BROW_R_Y"  # MAX 1.0 MIN -1.0
    PARAM_BROW_X = "PARAM_BROW_R_X"  # MAX 1.0 MIN -1.0
    PARAM_BROW_ANGLE = "PARAM_BROW_R_ANGLE"  # MAX 1.0 MIN -1.0

    model = eyebrow_right_chan.broker.container.force_fetch(live2d.LAppModel)

    # 调用通用平滑移动函数
    await _smooth_move_eyebrow(
        model=model,
        param_x=PARAM_BROW_X,
        param_y=PARAM_BROW_Y,
        param_angle=PARAM_BROW_ANGLE,
        target_x=x,
        target_y=y,
        target_angle=angle,
        speed=speed,
    )
