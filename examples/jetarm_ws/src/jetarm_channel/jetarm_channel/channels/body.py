from typing import Optional, Literal
from typing_extensions import Self

from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, INSTANCE

from ghoshell_moss import PyChannel, ChannelRuntime, Builder, Channel, Message, Text
from ghoshell_moss.core.concepts.command import CommandTaskResult
from ghoshell_moss_contrib.prototypes.ros2_robot.main_channel import run_trajectory, reset_pose
import asyncio


class JetArmChannel(Channel):
    def __init__(self, name: str, description: str) -> None:
        self._id = uuid()
        self._name = name
        self._description = description

        self._runtime: Optional[ChannelRuntime] = None
        self._container_instances = {}

        self._idle_move: Literal["breathing", "waving", "thinking"] = "breathing"

    def name(self) -> str:
        return self._name

    def id(self) -> str:
        return self._id

    def description(self) -> str:
        return self._description

    def bootstrap(self, container: Optional[IoCContainer] = None) -> "ChannelRuntime":
        if self._runtime and self._runtime.is_available():
            return self._runtime

        chan = PyChannel(name=self._name, description=self._description, blocking=True)

        for contract, instance in self._container_instances.items():
            chan.build.with_binding(contract, instance)

        chan.build.idle(self.on_idle)
        chan.build.context_messages(self.context_messages)

        chan.build.command()(self.set_idle_move)
        chan.build.command(doc=self.motion_doc)(self.motion)

        self._runtime = chan.bootstrap(container=container)
        return self._runtime

    def motion_doc(self):
        return f"""
        机械臂运动控制，可用name有{', '.join(MOTIONS.keys())}
        """

    async def motion(self, name: str):
        move = MOTIONS.get(name)
        if not move:
            return CommandTaskResult(
                observe=False,
                messages=[Message.new(role="user", name="__jetarm__").with_content(
                    Text(text=f"未找到运动配置{name}")
                )]
            )
        move_str = move.get("text")
        await run_trajectory(move_str)

    async def on_idle(self):
        try:
            while True:
                await self.motion(self._idle_move)
        except asyncio.CancelledError:
            pass

    async def context_messages(self):
        message = Message.new(role="user", name="__jetarm__")
        message.with_content(
            Text(text=f"空闲时运动为{self._idle_move}")
        )
        return [message]

    async def set_idle_move(self, move: str):
        """
        设置空闲时运动

        :param move: 空闲时运动，choices are breathing, waving, thinking
        """
        self._idle_move = move

    async def reset(self):
        """
        重置机器人到初始位置
        """
        await reset_pose()

    def with_binding(self, contract: type[INSTANCE], binding: INSTANCE) -> Self:
        self._container_instances[contract] = binding
        return self


# 统一管理所有动作的配置字典（所有轨迹、描述全部在这里）
MOTIONS = {
    "waving": {
        "text": """
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
        """,
        "desc": "波浪wave"
    },
    "curious_looking": {
        "text": """
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
        """,
        "desc": "好奇张望"
    },
    "greeting": {
        "text": """
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
        """,
        "desc": "打招呼"
    },
    "nodding_confirmation": {
        "text": """
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
        """,
        "desc": "点头确认"
    },
    "shaking_refusal": {
        "text": """
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
        """,
        "desc": "摇头否定"
    },
    "surprised": {
        "text": """
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
        """,
        "desc": "惊讶"
    },
    "happy_swing": {
        "text": """
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
        """,
        "desc": "快乐摇摆"
    },
    "sad_bowing": {
        "text": """
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
        """,
        "desc": "悲伤低头"
    },
    "proud_show": {
        "text": """
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
        """,
        "desc": "得意炫耀"
    },
    "confused_tilting": {
        "text": """
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
        """,
        "desc": "疑惑歪头"
    },
    "alert_defending": {
        "text": """
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
        """,
        "desc": "身体前伏地面，警惕防守"
    },
    "friendly_inviting": {
        "text": """
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
        """,
        "desc": "友好邀请"
    },
    "thinking": {
        "text": """
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
        """,
        "desc": "思考"
    },
    "sleepy_yawning": {
        "text": """
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
        """,
        "desc": "困倦打哈欠"
    },
    "draw_circle": {
        "text": """
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
        """,
        "desc": "画一个圆"
    },
    "snake_slithering": {
        "text": """
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
        """,
        "desc": "蛇形爬行"
    },
    "breathing": {
        "text": """
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
        """,
        "desc": "呼吸（一次）"
    },
    "stretch": {
        "text": """
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
        """,
        "desc": "伸懒腰"
    }
}
