import asyncio
import time

import live2d.v3 as live2d

from ghoshell_moss.core.py_channel import PyChannel

left_arm_chan = PyChannel(name="left_arm")
right_arm_chan = PyChannel(name="right_arm")


@left_arm_chan.build.command()
async def move(duration: float = 1.5, angle: float = 10.0):
    """
    移动左臂到指定角度

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度约为身体夹角45度，0.0为最小角度
    """
    model = left_arm_chan.broker.container.force_fetch(live2d.LAppModel)

    index = model.GetParamIds().index("PARAM_ARM_L_01")
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue("PARAM_ARM_L_01", target)
        await asyncio.sleep(0.016)


@right_arm_chan.build.command()
async def right_move(duration: float = 1.5, angle: float = 10.0):
    """
    移动右臂到指定角度

    :param duration:  执行时间
    :param angle:  提升角度，10.0为最大角度约为身体夹角45度，0.0为最小角度
    """
    model = right_arm_chan.broker.container.force_fetch(live2d.LAppModel)
    index = model.GetParamIds().index("PARAM_ARM_R_01")
    current_angle = model.GetParameterValue(index)

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        target = current_angle + (angle - current_angle) * progress
        model.SetParameterValue("PARAM_ARM_R_01", target)
        await asyncio.sleep(0.016)


@left_arm_chan.build.command()
async def akimbo(duration: float = 0.5):
    """
    执行左手臂叉腰动作

    :param duration:  执行时间
    """
    model = left_arm_chan.broker.container.force_fetch(live2d.LAppModel)

    # 获取左臂和左肘部的参数索引
    arm_index = model.GetParamIds().index("PARAM_ARM_L_01")
    elbow_index = model.GetParamIds().index("Param4")

    # 获取当前角度
    current_arm_angle = model.GetParameterValue(arm_index)
    current_elbow_angle = model.GetParameterValue(elbow_index)

    # 目标角度：手臂提升到中等角度，肘部向内弯折
    target_arm_angle = 4.0
    target_elbow_angle = -30.0

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration

        # 计算插值后的角度
        arm_target = current_arm_angle + (target_arm_angle - current_arm_angle) * progress
        elbow_target = current_elbow_angle + (target_elbow_angle - current_elbow_angle) * progress

        # 设置参数值
        model.SetParameterValue("PARAM_ARM_L_01", arm_target)
        model.SetParameterValue("Param4", elbow_target)

        await asyncio.sleep(0.016)


@right_arm_chan.build.command()
async def right_akimbo(duration: float = 0.5):
    """
    执行右手臂叉腰动作

    :param duration:  执行时间
    """
    model = right_arm_chan.broker.container.force_fetch(live2d.LAppModel)

    # 获取右臂和右肘部的参数索引
    arm_index = model.GetParamIds().index("PARAM_ARM_R_01")
    elbow_index = model.GetParamIds().index("Param5")

    # 获取当前角度
    current_arm_angle = model.GetParameterValue(arm_index)
    current_elbow_angle = model.GetParameterValue(elbow_index)

    # 目标角度：手臂提升到中等角度，肘部向内弯折
    target_arm_angle = 4.0
    target_elbow_angle = -30.0

    start_time = time.time()
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration

        # 计算插值后的角度
        arm_target = current_arm_angle + (target_arm_angle - current_arm_angle) * progress
        elbow_target = current_elbow_angle + (target_elbow_angle - current_elbow_angle) * progress

        # 设置参数值
        model.SetParameterValue("PARAM_ARM_R_01", arm_target)
        model.SetParameterValue("Param5", elbow_target)

        await asyncio.sleep(0.016)
