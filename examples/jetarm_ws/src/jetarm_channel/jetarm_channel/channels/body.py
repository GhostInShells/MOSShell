import asyncio

from pydantic import Field

from ghoshell_moss.core.concepts.states import StateBaseModel
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss_contrib.prototypes.ros2_robot.main_channel import reset_pose, run_trajectory

body_chan = PyChannel(name="body")

policy_pause_event = asyncio.Event()


@body_chan.build.on_idle
async def on_policy_run():
    policy_pause_event.clear()
    while not policy_pause_event.is_set():
        state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
        if state_model.policy == "breathing":
            await _breathing()
        elif state_model.policy == "waving":
            await _waving()
        elif state_model.policy == "thinking":
            await _thinking()
            await asyncio.sleep(0.5)
        elif state_model.policy == "reset_pose":
            await reset_pose()
            break
        else:
            break


class BodyPolicyStateModel(StateBaseModel):
    state_name = "body"
    state_desc = "body state model"

    policy: str = Field(default="breathing", description="body policy")


body_chan.build.state_model(BodyPolicyStateModel)

mock_policy = "breathing"


@body_chan.build.command()
async def set_default_policy(policy: str = "breathing"):
    """
    设置一个新的默认body policy

    :param policy:  body policy, default is breathing, choices are breathing, waving, thinking and reset_pose
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


async def _waving():
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "wrist_roll"],
        "points": [
            {"positions": [30, -58, -107, 45, 0, 0], "time_from_start": 0.0},
            {"positions": [30, -60, -98, 54, 0, 0], "time_from_start": 0.2},
            {"positions": [30, -56, -86, 59, 0, 0], "time_from_start": 0.4},
            {"positions": [30, -48, -75, 59, 0, 0], "time_from_start": 0.6},
            {"positions": [30, -39, -70, 54, 0, 0], "time_from_start": 0.8},
            {"positions": [30, -32, -73, 45, 0, 0], "time_from_start": 1.0},
            {"positions": [30, -30, -82, 36, 0, 0], "time_from_start": 1.2},
            {"positions": [30, -34, -94, 31, 0, 0], "time_from_start": 1.4},
            {"positions": [30, -42, -105, 31, 0, 0], "time_from_start": 1.6},
            {"positions": [30, -51, -110, 36, 0, 0], "time_from_start": 1.8},
            {"positions": [30, -58, -107, 45, 0, 0], "time_from_start": 2.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def waving():
    """
    波浪wave
    """
    state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
    if state_model.policy == "waving":
        return
    await _waving()


@body_chan.build.command()
async def curious_looking():
    """
    好奇张望
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_roll", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
        "points": [
            {"positions": [30.0, 0.0, -45.0, -90.0, 45.0, 0.0], "time_from_start": 0.0},
            {"positions": [38.0, 40.0, -25.0, -75.0, 52.0, 30.0], "time_from_start": 1.0},
            {"positions": [30.0, 0.0, -45.0, -90.0, 45.0, 0.0], "time_from_start": 1.5},
            {"positions": [38.0, -35.0, -65.0, -105.0, 32.0, -25.0], "time_from_start": 2.0},
            {"positions": [30.0, 0.0, -45.0, -90.0, 45.0, 0.0], "time_from_start": 3.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def greeting():
    """
    打招呼
    """
    text = """
    {
      "joint_names": ["gripper", "wrist_roll", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
      "points": [
        {"positions": [30, 0, -45, -90, 45, 0], "time_from_start": 0.0},
        {"positions": [40, 25, -35, -70, 35, 25], "time_from_start": 0.3},
        {"positions": [45, 40, -25, -60, 25, 40], "time_from_start": 0.6},
        {"positions": [40, 25, -35, -70, 35, 25], "time_from_start": 0.9},
        {"positions": [30, 0, -45, -90, 45, 0], "time_from_start": 1.2},
        {"positions": [40, -25, -35, -70, 35, -25], "time_from_start": 1.5},
        {"positions": [45, -40, -25, -60, 25, -40], "time_from_start": 1.8},
        {"positions": [40, -25, -35, -70, 35, -25], "time_from_start": 2.1},
        {"positions": [30, 0, -45, -90, 45, 0], "time_from_start": 2.4}
      ],
      "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def nodding_confirmation():
    """
    点头确认
    """
    text = """
    {
      "joint_names": ["wrist_pitch", "elbow_pitch", "shoulder_pitch"],
      "points": [
        {"positions": [-45, -90, 45], "time_from_start": 0.0},
        {"positions": [-55, -95, 40], "time_from_start": 0.2},
        {"positions": [-65, -100, 35], "time_from_start": 0.4},
        {"positions": [-55, -95, 40], "time_from_start": 0.6},
        {"positions": [-45, -90, 45], "time_from_start": 0.8},
        {"positions": [-55, -95, 40], "time_from_start": 1.0},
        {"positions": [-65, -100, 35], "time_from_start": 1.2},
        {"positions": [-55, -95, 40], "time_from_start": 1.4},
        {"positions": [-45, -90, 45], "time_from_start": 1.6},
        {"positions": [-55, -95, 40], "time_from_start": 1.8},
        {"positions": [-65, -100, 35], "time_from_start": 2.0},
        {"positions": [-55, -95, 40], "time_from_start": 2.2},
        {"positions": [-45, -90, 45], "time_from_start": 2.4}
      ],
      "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def shaking_refusal():
    """
    摇头否定
    """
    text = """
    {
        "joint_names": ["wrist_roll", "shoulder_roll", "elbow_pitch", "shoulder_pitch"],
        "points": [
            {"positions": [0, 0, -90, 45], "time_from_start": 0.0},
            {"positions": [10, 7, -88, 44], "time_from_start": 0.05},
            {"positions": [35, 25, -85, 43], "time_from_start": 0.15},
            {"positions": [25, 18, -87, 44], "time_from_start": 0.25},
            {"positions": [-35, -25, -85, 43], "time_from_start": 0.4},
            {"positions": [-25, -18, -87, 44], "time_from_start": 0.5},
            {"positions": [35, 25, -85, 43], "time_from_start": 0.65},
            {"positions": [25, 18, -87, 44], "time_from_start": 0.75},
            {"positions": [-35, -25, -85, 43], "time_from_start": 0.9},
            {"positions": [-25, -18, -87, 44], "time_from_start": 1.0},
            {"positions": [35, 25, -85, 43], "time_from_start": 1.15},
            {"positions": [25, 18, -87, 44], "time_from_start": 1.25},
            {"positions": [0, 0, -90, 45], "time_from_start": 1.4}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def surprised():
    """
    惊讶
    """
    text = """
    {
      "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
      "points": [
        {"positions": [30, -45, -90, 45, 0], "time_from_start": 0.0},
        {"positions": [15, -35, -95, 47, 3], "time_from_start": 0.1},
        {"positions": [0, -25, -100, 49, -3], "time_from_start": 0.2},
        {"positions": [-10, -20, -105, 51, 5], "time_from_start": 0.3},
        {"positions": [-5, -22, -103, 50, -5], "time_from_start": 0.4},
        {"positions": [-10, -20, -105, 51, 3], "time_from_start": 0.5},
        {"positions": [5, -28, -98, 48, 0], "time_from_start": 0.7},
        {"positions": [15, -33, -94, 47, 0], "time_from_start": 0.9},
        {"positions": [25, -38, -92, 46, 0], "time_from_start": 1.2},
        {"positions": [30, -45, -90, 45, 0], "time_from_start": 1.5}
      ],
      "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def happy_swing():
    """
    快乐摇摆
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_roll", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
        "points": [
            {"positions": [30, 0, -45, -90, 45, 0], "time_from_start": 0.0},
            {"positions": [20, 15, -40, -85, 47, 10], "time_from_start": 0.2},
            {"positions": [25, -15, -35, -80, 49, -10], "time_from_start": 0.4},
            {"positions": [20, 20, -40, -85, 47, 12], "time_from_start": 0.6},
            {"positions": [25, -20, -35, -80, 49, -12], "time_from_start": 0.8},
            {"positions": [20, 25, -40, -85, 47, 15], "time_from_start": 1.0},
            {"positions": [25, -25, -35, -80, 49, -15], "time_from_start": 1.2},
            {"positions": [20, 15, -40, -85, 47, 10], "time_from_start": 1.4},
            {"positions": [25, -15, -35, -80, 49, -10], "time_from_start": 1.6},
            {"positions": [30, 0, -45, -90, 45, 0], "time_from_start": 1.8}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def sad_bowing():
    """
    悲伤低头
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
        "points": [
            {"positions": [30, -45, -90, 45, 0], "time_from_start": 0.0},
            {"positions": [20, -70, -100, 40, -5], "time_from_start": 0.5},
            {"positions": [15, -90, -110, 35, -8], "time_from_start": 1.0},
            {"positions": [10, -110, -120, 30, -10], "time_from_start": 1.5},
            {"positions": [8, -120, -125, 28, -12], "time_from_start": 2.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def proud_show():
    """
    得意炫耀
    """
    text = """
{
"joint_names": ["wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "gripper", "wrist_roll"],
"points": [
{"positions": [-45.0, -90.0, 45.0, 0.0, 30.0, 0.0], "time_from_start": 0.0},
{"positions": [-20.0, -75.0, 55.0, 8.0, 15.0, 5.0], "time_from_start": 0.25},
{"positions": [5.0, -63.0, 60.0, 25.0, -15.0, 20.0], "time_from_start": 0.5},
{"positions": [5.0, -63.0, 60.0, 25.0, 5.0, 20.0], "time_from_start": 0.75},
{"positions": [5.0, -63.0, 60.0, 25.0, -15.0, 20.0], "time_from_start": 1.0},
{"positions": [5.0, -63.0, 60.0, 25.0, 5.0, 20.0], "time_from_start": 1.25},
{"positions": [5.0, -63.0, 60.0, 25.0, -15.0, 20.0], "time_from_start": 1.5},
{"positions": [5.0, -63.0, 60.0, 25.0, 5.0, 20.0], "time_from_start": 1.75},
{"positions": [0.0, -68.0, 57.0, 18.0, -5.0, 15.0], "time_from_start": 2.0},
{"positions": [-10.0, -72.0, 53.0, 12.0, 0.0, 10.0], "time_from_start": 2.25},
{"positions": [-20.0, -78.0, 50.0, 8.0, 10.0, 5.0], "time_from_start": 2.5},
{"positions": [-30.0, -83.0, 48.0, 3.0, 20.0, 2.0], "time_from_start": 2.75},
{"positions": [-45.0, -90.0, 45.0, 0.0, 30.0, 0.0], "time_from_start": 3.0}
],
"loop": 1
}
    """
    await run_trajectory(text)


@body_chan.build.command()
async def confused_tilting():
    """
    疑惑歪头
    """
    text = """
    {
        "joint_names": ["wrist_roll", "shoulder_roll", "wrist_pitch", "elbow_pitch", "gripper"],
        "points": [
            {"positions": [0, 0, -45, -90, 30], "time_from_start": 0.0},
            {"positions": [20, 15, -40, -88, -10], "time_from_start": 0.3},
            {"positions": [25, 18, -38, -86, 30], "time_from_start": 0.6},
            {"positions": [25, 18, -38, -86, -10], "time_from_start": 0.9},
            {"positions": [25, 18, -38, -86, 30], "time_from_start": 1.1},
            {"positions": [25, 18, -38, -86, -10], "time_from_start": 1.3},
            {"positions": [25, 18, -38, -86, 30], "time_from_start": 1.5},
            {"positions": [15, 12, -40, -87, 15], "time_from_start": 1.8},
            {"positions": [25, 18, -38, -86, -10], "time_from_start": 2.1},
            {"positions": [18, 15, -39, -86, 25], "time_from_start": 2.4},
            {"positions": [25, 18, -38, -86, -10], "time_from_start": 2.7},
            {"positions": [0, 0, -42, -88, 20], "time_from_start": 3.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def alert_defending():
    """
    身体前伏地面，警惕防守
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "wrist_roll"],
        "points": [
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 0.0},
            {"positions": [-10, -30, -100, 20, 0, 0], "time_from_start": 0.2},
            {"positions": [-25, -15, -110, 0, 0, 0], "time_from_start": 0.4},
            {"positions": [-35, -5, -118, -15, 0, 0], "time_from_start": 0.6},
            {"positions": [-45, 0, -125, -25, 0, 0], "time_from_start": 0.8},
            {"positions": [-50, 0, -128, -35, 0, 0], "time_from_start": 1.0},
            {"positions": [-55, 0, -130, -40, 0, 0], "time_from_start": 1.2},
            {"positions": [-60, 0, -132, -45, 0, 0], "time_from_start": 1.4},
            {"positions": [-60, 0, -135, -50, 0, 0], "time_from_start": 1.6},
            {"positions": [-60, 0, -135, -55, 0, 0], "time_from_start": 1.8},
            {"positions": [-60, 0, -135, -55, 5, 3], "time_from_start": 2.0},
            {"positions": [-60, 0, -135, -55, -5, -3], "time_from_start": 2.1},
            {"positions": [-60, 0, -135, -55, 8, 5], "time_from_start": 2.2},
            {"positions": [-60, 0, -135, -55, -8, -5], "time_from_start": 2.3},
            {"positions": [-60, 0, -135, -55, 0, 0], "time_from_start": 2.4},
            {"positions": [-60, 0, -135, -55, 0, 0], "time_from_start": 2.6},
            {"positions": [-55, 0, -132, -50, 0, 0], "time_from_start": 2.8},
            {"positions": [-45, -5, -125, -40, 0, 0], "time_from_start": 3.0},
            {"positions": [-30, -15, -115, -20, 0, 0], "time_from_start": 3.2},
            {"positions": [-15, -25, -105, 0, 0, 0], "time_from_start": 3.4},
            {"positions": [0, -35, -98, 15, 0, 0], "time_from_start": 3.6},
            {"positions": [15, -40, -92, 30, 0, 0], "time_from_start": 3.8},
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 4.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def friendly_inviting():
    """
    友好邀请
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "wrist_roll"],
        "points": [
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 0.0},
            {"positions": [20, -60, -100, 35, -20, -15], "time_from_start": 0.3},
            {"positions": [15, -55, -95, 40, -15, -10], "time_from_start": 0.6},
            {"positions": [20, -60, -100, 35, -20, -15], "time_from_start": 0.9},
            {"positions": [15, -55, -95, 40, -15, -10], "time_from_start": 1.2},
            {"positions": [20, -60, -100, 35, -20, -15], "time_from_start": 1.5},
            {"positions": [15, -55, -95, 40, -15, -10], "time_from_start": 2.0},
            {"positions": [20, -60, -100, 35, -20, -15], "time_from_start": 2.3},
            {"positions": [25, -50, -92, 42, -10, -5], "time_from_start": 2.6},
            {"positions": [28, -47, -91, 44, -5, -2], "time_from_start": 2.9},
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 3.2}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


async def _thinking():
    text = """
    {
    "joint_names": ["gripper", "wrist_pitch", "wrist_roll", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
    "points": [
    {"positions": [30, -45, 0, -90, 45, 0], "time_from_start": 0.0},
    {"positions": [25, -50, 10, -95, 40, 5], "time_from_start": 0.5},
    {"positions": [20, -55, 15, -100, 38, 10], "time_from_start": 1.0},
    {"positions": [18, -55, 18, -100, 38, 12], "time_from_start": 1.5},
    {"positions": [20, -53, 15, -98, 40, 10], "time_from_start": 2.0},
    {"positions": [25, -48, 8, -93, 42, 5], "time_from_start": 2.5},
    {"positions": [30, -45, 0, -90, 45, 0], "time_from_start": 3.0}
    ],
    "loop": 1
    }
        """
    await run_trajectory(text)


@body_chan.build.command()
async def thinking():
    """
    思考
    """
    state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
    if state_model.policy == "thinking":
        return
    await _thinking()


@body_chan.build.command()
async def sleepy_yawning():
    """
    困倦打哈欠
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "wrist_roll", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
        "points": [
            {"positions": [30, -45, 0, -90, 45, 0], "time_from_start": 0.0},
            {"positions": [10, -40, 0, -85, 48, 0], "time_from_start": 1.0},
            {"positions": [-40, -20, 0, -75, 55, 0], "time_from_start": 2.0},
            {"positions": [-40, -20, 0, -75, 55, 0], "time_from_start": 2.5},
            {"positions": [10, -40, 0, -85, 48, 0], "time_from_start": 3.0},
            {"positions": [30, -45, 0, -90, 45, 0], "time_from_start": 4.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def draw_circle():
    """
    画一个圆
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "wrist_roll", "elbow_pitch", "shoulder_pitch", "shoulder_roll"],
        "points": [
            {"positions": [30, -55, 0, -70, 52, 0], "time_from_start": 0.0},
            {"positions": [30, -52, -21, -76, 55, 21], "time_from_start": 0.5},
            {"positions": [30, -45, -30, -90, 52, 30], "time_from_start": 1.0},
            {"positions": [30, -38, -21, -104, 45, 21], "time_from_start": 1.5},
            {"positions": [30, -35, 0, -110, 38, 0], "time_from_start": 2.0},
            {"positions": [30, -38, 21, -104, 35, -21], "time_from_start": 2.5},
            {"positions": [30, -45, 30, -90, 38, -30], "time_from_start": 3.0},
            {"positions": [30, -52, 21, -76, 45, -21], "time_from_start": 3.5},
            {"positions": [30, -55, 0, -70, 52, 0], "time_from_start": 4.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def snake_slithering():
    """
    蛇形爬行
    """
    text = """
    {
        "joint_names": ["shoulder_roll", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "gripper"],
        "points": [
            {"positions": [0, 30, -110, -40, 0, 25], "time_from_start": 0.0},
            {"positions": [20, 35, -105, -38, 10, 20], "time_from_start": 0.4},
            {"positions": [30, 40, -95, -35, 20, 15], "time_from_start": 0.8},
            {"positions": [20, 45, -85, -32, 10, 20], "time_from_start": 1.2},
            {"positions": [0, 50, -80, -30, 0, 25], "time_from_start": 1.6},
            {"positions": [-20, 45, -85, -32, -10, 20], "time_from_start": 2.0},
            {"positions": [-30, 40, -95, -35, -20, 15], "time_from_start": 2.4},
            {"positions": [-20, 35, -105, -38, -10, 20], "time_from_start": 2.8},
            {"positions": [0, 30, -110, -40, 0, 25], "time_from_start": 3.2},
            {"positions": [20, 25, -115, -42, 10, 20], "time_from_start": 3.6},
            {"positions": [30, 20, -120, -45, 20, 15], "time_from_start": 4.0},
            {"positions": [20, 25, -115, -42, 10, 20], "time_from_start": 4.4},
            {"positions": [0, 30, -110, -40, 0, 25], "time_from_start": 4.8}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


async def _breathing():
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "wrist_roll"],
        "points": [
            {"positions": [30.0, -45.0, -90.0, 45.0, 0.0, 0.0], "time_from_start": 0.0},
            {"positions": [20.0, -43.0, -85.0, 48.0, 0.0, 0.0], "time_from_start": 0.75},
            {"positions": [15.0, -40.0, -80.0, 52.0, 0.0, 0.0], "time_from_start": 1.5},
            {"positions": [15.5, -40.2, -80.5, 51.8, 0.0, 0.0], "time_from_start": 1.6},
            {"positions": [20.0, -43.0, -85.0, 48.0, 0.0, 0.0], "time_from_start": 2.25},
            {"positions": [30.0, -45.0, -90.0, 45.0, 0.0, 0.0], "time_from_start": 3.0},
            {"positions": [32.0, -47.0, -95.0, 42.0, 0.0, 0.0], "time_from_start": 3.75},
            {"positions": [35.0, -50.0, -100.0, 38.0, 0.0, 0.0], "time_from_start": 4.5}, 
            {"positions": [34.8, -49.8, -99.5, 38.2, 0.0, 0.0], "time_from_start": 4.6},
            {"positions": [32.0, -47.0, -95.0, 42.0, 0.0, 0.0], "time_from_start": 5.25},
            {"positions": [30.0, -45.0, -90.0, 45.0, 0.0, 0.0], "time_from_start": 6.0}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)


@body_chan.build.command()
async def breathing():
    """
    呼吸（一次）
    """
    state_model = body_chan.broker.states.get_model(BodyPolicyStateModel)
    if state_model.policy == "breathing":
        return
    await _breathing()


@body_chan.build.command()
async def stretch():
    """
    伸懒腰
    """
    text = """
    {
        "joint_names": ["gripper", "wrist_pitch", "elbow_pitch", "shoulder_pitch", "shoulder_roll", "wrist_roll"],
        "points": [
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 0.0},
            {"positions": [10, -35, -80, 50, 0, 0], "time_from_start": 0.5},
            {"positions": [-10, -25, -60, 55, 0, 0], "time_from_start": 1.0},
            {"positions": [-30, -15, -40, 58, 0, 0], "time_from_start": 1.5},
            {"positions": [-40, -10, -30, 60, 0, 0], "time_from_start": 2.0},
            {"positions": [-40, -10, -30, 60, 0, 0], "time_from_start": 2.5},
            {"positions": [-30, -15, -40, 58, 0, 0], "time_from_start": 3.0},
            {"positions": [-10, -25, -60, 55, 0, 0], "time_from_start": 3.5},
            {"positions": [10, -35, -80, 50, 0, 0], "time_from_start": 4.0},
            {"positions": [30, -45, -90, 45, 0, 0], "time_from_start": 4.5}
        ],
        "loop": 1
    }
    """
    await run_trajectory(text)
