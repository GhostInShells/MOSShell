import asyncio
import time

import live2d.v3 as live2d


async def open_close(
    model: live2d.LAppModel,
    param_name: str,
    duration=1.5,
    speed=1.0,
    max_value=1.0,
    min_value=0.0,
    initial_direction="close",
):
    """
    通用参数动画函数，用于处理各种参数的开合动画

    :param model: Live2D模型实例
    :param param_name: 要控制的参数名称
    :param duration: 动画持续时间，单位秒
    :param speed: 动画速度参数，值越大速度越快
    :param max_value: 参数的最大值
    :param min_value: 参数的最小值
    :param initial_direction: 初始方向，"close"(闭合) 或 "open"(打开)
    """
    # 参数验证
    if max_value > 1.0 or max_value < 0.0:
        max_value = 1.0
    if min_value < 0.0 or min_value > max_value:
        min_value = 0.0
    if speed <= 0:
        speed = 1.0

    # 动画帧间隔
    FRAME_INTERVAL = 0.016
    MOVE_INTERVAL = 0.02 * speed

    # 保存当前参数状态
    param_index = model.GetParamIds().index(param_name)
    original_value = model.GetParameterValue(param_index)

    # 控制开合方向的状态变量
    if initial_direction == "close":
        delta = -MOVE_INTERVAL  # 开始时是闭合方向
        current_value = max_value
    else:
        delta = MOVE_INTERVAL  # 开始时是打开方向
        current_value = min_value

    start_time = time.time()
    time_up = False

    try:
        # 动画循环
        while True:
            current_time = time.time()

            # 检查是否已经到达设定时间
            if not time_up and current_time - start_time >= duration:
                time_up = True

            # 如果时间已到，向最终状态移动
            if time_up:
                # 根据初始方向决定最终状态
                if initial_direction == "close":
                    # 闭合动画，最终状态是打开
                    if current_value < max_value:
                        new_value = current_value + MOVE_INTERVAL
                        if new_value > max_value:
                            new_value = max_value
                    else:
                        break
                else:
                    # 打开动画，最终状态是闭合
                    if current_value > min_value:
                        new_value = current_value - MOVE_INTERVAL
                        if new_value < min_value:
                            new_value = min_value
                    else:
                        break
            else:
                # 正常动画阶段，计算新位置
                new_value = current_value + delta

                # 检查是否到达端点，到达则改变方向
                if new_value <= min_value:
                    new_value = min_value
                    delta = -delta  # 改变方向为打开
                elif new_value >= max_value:
                    new_value = max_value
                    delta = -delta  # 改变方向为闭合

            # 设置参数值
            model.SetParameterValue(param_name, new_value)
            current_value = new_value

            # 等待一帧的时间
            await asyncio.sleep(FRAME_INTERVAL)

        # 确保最终状态正确
        if initial_direction == "close":
            model.SetParameterValue(param_name, max_value)
            final_value = max_value
        else:
            model.SetParameterValue(param_name, min_value)
            final_value = min_value

    except Exception as e:
        # 如果发生异常，恢复原始状态
        model.SetParameterValue(param_name, original_value)
        raise e
