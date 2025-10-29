from ghoshell_moss.channels.py_channel import PyChannel
import live2d.v3 as live2d
import asyncio
import time

body_chan = PyChannel(name='body')

policy_pause_event = asyncio.Event()

@body_chan.build.on_policy_run
async def on_policy_run():
    policy_pause_event.clear()
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    while not policy_pause_event.is_set():
        # 等待 其他 Motions 完成
        while not model.IsMotionFinished():
            await asyncio.sleep(0.1)
        # Policy的Priority设置为1（较低），是为了确保其他Motion可打断Policy Motion
        model.StartMotion("Happy", 1, 1)

@body_chan.build.on_policy_pause
async def on_policy_pause():
    policy_pause_event.set()

async def start_motion(model: live2d.LAppModel, motion_name: str, no: int, duration: float):
    # model.StartMotion(motion_name, no, 4)
    start_time = time.time()

    while time.time() - start_time < duration:
        model.StartMotion(motion_name, no, 4)
        while not model.IsMotionFinished():
            # print(f"{motion_name} motion is running")
            await asyncio.sleep(0.1)
    model.ResetParameters()


@body_chan.build.command()
async def angry(no: int = 0, duration: float = 5.0):
    """
    angry motion, two motions can be use 0 and 1

    :param no:  angry motion number
    :param duration:  angry motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Angry", no, duration)


@body_chan.build.command()
async def happy(no: int = 0, duration: float = 5.0):
    """
    happy motion, three motions can be use 0, 1 and 2

    :param no:  happy motion number
    :param duration:  happy motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Happy", no, duration)


@body_chan.build.command()
async def love(duration: float = 5.0):
    """
    love motion

    :param duration:  love motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Love", 0, duration)


@body_chan.build.command()
async def gentle_torso_twist(duration: float = 5.0):
    """
    轻旋上身

    :param duration:  gentle torso twist duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "TurnHead", 0, duration)


@body_chan.build.command()
async def sad(duration: float = 5.0):
    """
    sad motion

    :param duration:  sad motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Sad", 0, duration)


@body_chan.build.command()
async def nod_head(no: int = 0, duration: float = 5.0):
    """
    nod head motion, two motions can be use 0 and 1

    :param no:  nod head motion number
    :param duration:  nod head motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "NodHead", no, duration)


@body_chan.build.command()
async def walk(duration: float = 5.0):
    """
    walk motion

    :param duration:  walk motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Walk", 0, duration)


@body_chan.build.command()
async def sleep( duration: float = 5.0):
    """
    sleep motion

    :param duration:  sleep motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Sleep", 0, duration)


@body_chan.build.command()
async def activate_body(no: int = 0, duration: float = 5.0):
    """
    activate body motion, two motions can be use 0 and 1

    :param no:  activate body motion number
    :param duration:  activate body motion duration
    """
    model = body_chan.client.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "ActivateBody", no, duration)
