from ghoshell_moss.channels.py_channel import PyChannel
import live2d.v3 as live2d
import asyncio
import time

necktie_chan = PyChannel(name='necktie')


@necktie_chan.build.command()
async def flutter(duration: float = 5.0):
    """
    领带飘动
    """
    LEFT = -1.0
    RIGHT = 1.0
    SPEED = 0.01
    PARAM_NECKTIE = "PARAM_NECKTIE"


    model = necktie_chan.client.container.force_fetch(live2d.LAppModel)
    
    # 设置初始位置为原点
    current_value = 0.0
    model.SetParameterValue(PARAM_NECKTIE, current_value)
    await asyncio.sleep(SPEED)
    
    start_time = time.time()
    # 控制飘动方向的状态变量，使用增量而不是布尔值
    # 正值表示向右，负值表示向左
    delta = 0.02  # 每帧移动的步长
    
    # 跟踪是否已达到duration时间
    time_up = False
    
    while True:
        current_time = time.time()
        
        # 检查是否已经到达设定时间
        if not time_up and current_time - start_time >= duration:
            time_up = True
        
        # 如果时间已到，需要回到原点
        if time_up:
            # 如果已经接近原点，直接设置为0并结束
            if abs(current_value) < delta:
                model.SetParameterValue(PARAM_NECKTIE, 0.0)
                break
            # 否则向原点移动
            if current_value > 0:
                delta = -abs(delta)  # 向左移动
            else:
                delta = abs(delta)   # 向右移动
        
        # 计算新位置
        new_value = current_value + delta
        
        # 检查是否到达端点，到达则改变方向
        if new_value <= LEFT:
            new_value = LEFT
            delta = -delta  # 改变方向向右
        elif new_value >= RIGHT:
            new_value = RIGHT
            delta = -delta  # 改变方向向左
        
        # 设置领带位置（一帧只设置一个位置）
        model.SetParameterValue(PARAM_NECKTIE, new_value)
        current_value = new_value
        
        # 等待一帧的时间
        await asyncio.sleep(SPEED)
