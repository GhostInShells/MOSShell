from ghoshell_moss.channels.py_channel import PyChannel
import live2d.v3 as live2d
import asyncio

motion_chan = PyChannel(name='motion')


@motion_chan.build.command()
async def angry(no: int = 0):
    """
    angry motion, two motions can be use 0 and 1

    :param no:  angry motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Angry", no, 3)
    while not model.IsMotionFinished():
        print("angry motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def happy(no: int = 0):
    """
    happy motion, three motions can be use 0, 1 and 2

    :param no:  happy motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Happy", no, 4)
    while not model.IsMotionFinished():
        print("happy motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def love(no: int = 0):
    """
    love motion, one motion can be use 0

    :param no:  love motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Love", no, 3)
    while not model.IsMotionFinished():
        print("love motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def turn_head(no: int = 0):
    """
    turn head motion, one motion can be use 0

    :param no:  turn head motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("TurnHead", no, 3)
    while not model.IsMotionFinished():
        print("turn head motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def sad(no: int = 0):
    """
    sad motion, one motion can be use 0

    :param no:  sad motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Sad", no, 3)
    while not model.IsMotionFinished():
        print("sad motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def nod_head(no: int = 0):
    """
    nod head motion, two motions can be use 0 and 1

    :param no:  nod head motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("NodHead", no, 3)
    while not model.IsMotionFinished():
        print("nod head motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def walk(no: int = 0):
    """
    walk motion, one motion can be use 0

    :param no:  walk motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Walk", no, 3)
    while not model.IsMotionFinished():
        print("walk motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def sleep(no: int = 0):
    """
    sleep motion, one motion can be use 0

    :param no:  sleep motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("Sleep", no, 3)
    while not model.IsMotionFinished():
        print("sleep motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()


@motion_chan.build.command()
async def activate_body(no: int = 0):
    """
    activate body motion, two motions can be use 0 and 1

    :param no:  activate body motion number
    """
    model = motion_chan.client.container.force_fetch(live2d.LAppModel)
    model.StartMotion("ActivateBody", no, 3)
    while not model.IsMotionFinished():
        print("activate body motion is running")
        await asyncio.sleep(0.1)
    model.ResetParameters()
