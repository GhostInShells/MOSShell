import live2d.v3 as live2d

from ghoshell_moss.core.py_channel import PyChannel

from .motions import open_close

mouth_chan = PyChannel(name="mouth")


@mouth_chan.build.command()
async def open(size: float = 1.0):
    """
    open mouth

    @param size: 嘴巴打开大小，0 到 1 之间的浮点数
    """
    # 参数验证
    if size > 1.0:
        size = 1.0
    elif size < 0.0:
        size = 0.0

    model = mouth_chan.broker.container.force_fetch(live2d.LAppModel)
    model.SetParameterValue("ParamMouthOpenY", size)


@mouth_chan.build.command()
async def speak(duration: float = 5.0, speed: float = 1.0, max_open: float = 0.7, min_open: float = 0.0):
    """
    说话的嘴部动作

    @param duration: 动画持续时间，单位秒
    @param speed: 开合速度参数，值越大速度越快，默认1.0
    @param max_open: 最大张开程度，0 到 1 之间的浮点数，默认为0.7
    @param min_open: 最小张开程度，0 到 1 之间的浮点数，默认为0.0
    """
    PARAM = "ParamMouthOpenY"
    # 特殊处理嘴部动作，说话通常从张开开始
    model = mouth_chan.broker.container.force_fetch(live2d.LAppModel)

    # 调用通用动画函数，注意初始方向设置为打开
    final_value = await open_close(
        model=model,
        param_name=PARAM,
        duration=duration,
        speed=speed,
        max_value=max_open,
        min_value=min_open,
        initial_direction="open",  # 说话从打开开始
    )
    # 确保最终状态是完全闭合
    model.SetParameterValue(PARAM, 0.0)
