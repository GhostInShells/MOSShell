import asyncio
import time

import live2d.v3 as live2d
from pydantic import Field

from ghoshell_moss.core.concepts.states import StateBaseModel
from ghoshell_moss.core.py_channel import PyChannel

body_chan = PyChannel(
    name="body",
    description="Live2d body of image MIKU",
    blocking=True,
)

policy_pause_event = asyncio.Event()


@body_chan.build.idle
async def on_policy_run():
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    policy_pause_event.clear()
    while not policy_pause_event.is_set() and body_chan.is_running():
        # 等待 其他 Motions 完成
        while not model.IsMotionFinished():
            await asyncio.sleep(0.1)
        if not body_chan.is_running():
            break
        model.ResetExpressions()  # 防止表情重叠
        model.ResetExpression()
        # Policy的Priority设置为1（较低），是为了确保其他Motion可打断Policy Motion
        state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
        model.StartMotion(state_model.policy, 0, 1)


@body_chan.build.state_model()
class BodyPolicyStateModel(StateBaseModel):
    state_name = "body"
    state_desc = "body state model"

    policy: str = Field(default="Happy", description="body policy")


# 因为description是sync函数，待修改为async函数，所以这里先mock一下本地数据
mock_policy = "Happy"


@body_chan.build.command()
async def set_default_policy(policy: str = "Happy"):
    """
    设置一个新的默认body policy，当执行完其他动作时，会自动执行默认policy

    :param policy:  body policy, default is Happy, choices are Happy, Angry, Love, Sad
    """
    state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
    state_model.policy = policy
    global mock_policy
    mock_policy = policy
    await body_chan.broker.states.save(state_model)


@body_chan.build.description()
def description() -> str:
    """获取当前body policy"""
    return f"当前body policy是{mock_policy}"


async def start_motion(model: live2d.LAppModel, motion_name: str, no: int, duration: float):
    start_time = time.time()

    while time.time() - start_time < duration:
        model.StartMotion(motion_name, no, 4)
        while not model.IsMotionFinished():
            await asyncio.sleep(0.1)
    model.ResetParameters()


@body_chan.build.command()
async def angry(no: int = 0, duration: float = 5.0):
    """
    angry motion, two motions can be use 0 and 1

    :param no:  angry motion number
    :param duration:  angry motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Angry", no, duration)


@body_chan.build.command()
async def happy(no: int = 0, duration: float = 5.0):
    """
    happy motion, three motions can be use 0, 1 and 2

    :param no:  happy motion number
    :param duration:  happy motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Happy", no, duration)


@body_chan.build.command()
async def love(duration: float = 5.0):
    """
    love motion

    :param duration:  love motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Love", 0, duration)


@body_chan.build.command()
async def gentle_torso_twist(duration: float = 5.0):
    """
    轻旋上身

    :param duration:  gentle torso twist duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "TurnHead", 0, duration)


@body_chan.build.command()
async def sad(duration: float = 5.0):
    """
    sad motion

    :param duration:  sad motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Sad", 0, duration)


@body_chan.build.command()
async def nod_head(no: int = 0, duration: float = 5.0):
    """
    nod head motion, two motions can be use 0 and 1

    :param no:  nod head motion number
    :param duration:  nod head motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "NodHead", no, duration)


@body_chan.build.command()
async def walk(duration: float = 5.0):
    """
    walk motion

    :param duration:  walk motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Walk", 0, duration)


@body_chan.build.command()
async def sleep(duration: float = 5.0):
    """
    sleep motion

    :param duration:  sleep motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "Sleep", 0, duration)


@body_chan.build.command()
async def activate_body(no: int = 0, duration: float = 5.0):
    """
    activate body motion, two motions can be use 0 and 1

    :param no:  activate body motion number
    :param duration:  activate body motion duration
    """
    model = body_chan.broker.container.force_fetch(live2d.LAppModel)
    await start_motion(model, "ActivateBody", no, duration)
