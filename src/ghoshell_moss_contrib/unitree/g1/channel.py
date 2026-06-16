"""
G1 Body Channel — Unitree G1 人形机器人身体控制。

三个子 channel:
  - loco:  全身运动控制 (LocoClient)
  - arm:   手臂预设动作 (G1ArmActionClient)
  - audio: 音频灯光 (AudioClient)

依赖注入: build_g1_channel() 接受可选的 client 实例。
  传 None → 命令返回 "no client" (安全降级，不崩溃)。
  传 mock → 本地无痛测试。
  传真实 client → PC2 实机运行。

Code as Prompt: 每个命令的 Python 函数签名 = 模型看到的接口。
"""

from __future__ import annotations

from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel,
    MutableChannel,
)
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel

from ghoshell_moss_contrib.unitree.g1._bootstrap import bootstrap
bootstrap()

# ── arm action name → id 映射 (来自 SDK g1_arm_action_client.py) ──────────

_ARM_ACTION_MAP: dict[str, int] = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}

# ── loco channel ──────────────────────────────────────────────────────────

def _build_loco_channel(client=None) -> MutableChannel:
    """构建 G1 运动控制子 channel。"""

    chan = new_channel(name="loco", description="G1 locomotion — FSM/velocity/stand height")

    @chan.build.command(always_observe=False)
    async def damp() -> str:
        """急停: 设置 FSM 为 damp 模式。总是可用。"""
        if client is None:
            return "no client"
        code = client.Damp()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def start() -> str:
        """启动运动控制: 设置 FSM 为 start 模式 (500)。"""
        if client is None:
            return "no client"
        code = client.Start()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def sit() -> str:
        """坐下: 设置 FSM 为 sit 模式 (3)。"""
        if client is None:
            return "no client"
        code = client.Sit()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def squat_to_stand_up() -> str:
        """从蹲姿站起: FSM 706。"""
        if client is None:
            return "no client"
        code = client.Squat2StandUp()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def stand_up_to_squat() -> str:
        """从站姿蹲下: FSM 706。"""
        if client is None:
            return "no client"
        code = client.StandUp2Squat()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def lie_to_stand_up() -> str:
        """从躺姿站起: FSM 702。"""
        if client is None:
            return "no client"
        code = client.Lie2StandUp()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def stop_move() -> str:
        """停止移动: 速度归零。"""
        if client is None:
            return "no client"
        code = client.StopMove()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def high_stand() -> str:
        """高位站立: 设置最大站立高度。"""
        if client is None:
            return "no client"
        code = client.HighStand()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def low_stand() -> str:
        """低位站立: 设置最小站立高度。"""
        if client is None:
            return "no client"
        code = client.LowStand()
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def move(
        vx: float,
        vy: float,
        vyaw: float,
        continuous: bool = False,
    ) -> str:
        """移动控制。

        vx:  前后速度 (m/s), 正值前进
        vy:  横向速度 (m/s), 正值左移
        vyaw: 旋转速度 (rad/s), 正值左转
        continuous: True=持续移动, False=1s 定时
        """
        if client is None:
            return "no client"
        code = client.Move(vx, vy, vyaw, continous_move=continuous)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def balance_stand(balance_mode: int) -> str:
        """设置平衡模式。

        balance_mode: 0=normal, 1=balance
        """
        if client is None:
            return "no client"
        code = client.BalanceStand(balance_mode)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def wave_hand(turn: bool = False) -> str:
        """挥手动作。turn=True 时转身挥手。"""
        if client is None:
            return "no client"
        code = client.WaveHand(turn_flag=turn)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def shake_hand() -> str:
        """握手动作 (自动切换阶段)。"""
        if client is None:
            return "no client"
        code = client.ShakeHand()
        return "ok" if code == 0 else f"error:{code}"

    # -- 底层 API: SetVelocity (用于自定义速度曲线) --
    @chan.build.command(always_observe=False)
    async def set_velocity(
        vx: float,
        vy: float,
        omega: float,
        duration: float = 1.0,
    ) -> str:
        """底层速度控制: 直接设置 vx, vy, omega 和持续时间。

        上层通常用 move() 即可。此命令用于需要精确控制 duration 的场景。
        """
        if client is None:
            return "no client"
        code = client.SetVelocity(vx, vy, omega, duration)
        return "ok" if code == 0 else f"error:{code}"

    return chan


# ── arm channel ───────────────────────────────────────────────────────────

def _build_arm_channel(client=None) -> MutableChannel:
    """构建 G1 手臂控制子 channel。"""

    chan = new_channel(name="arm", description="G1 arm — preset action execution")

    @chan.build.command(always_observe=True)
    async def list_actions() -> str:
        """获取可用手臂动作列表。"""
        if client is None:
            return "no client"
        code, data = client.GetActionList()
        if code == 0:
            if isinstance(data, list):
                names = [item.get("name", str(item)) for item in data]
                return f"ok: {', '.join(names)}"
            return f"ok: {data}"
        return f"error:{code}"

    @chan.build.command(always_observe=False)
    async def execute_action(action_name: str) -> str:
        """执行手臂预设动作。

        action_name: 动作名称，如 "wave hand", "clap", "hug", "hands up" 等。
        用 list_actions 查看完整列表。
        """
        if client is None:
            return "no client"
        action_id = _ARM_ACTION_MAP.get(action_name)
        if action_id is None:
            return f"unknown action: {action_name}"
        code = client.ExecuteAction(action_id)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def release_arm() -> str:
        """释放手臂: 停止当前手臂动作，释放手臂控制。"""
        if client is None:
            return "no client"
        code = client.ExecuteAction(99)
        return "ok" if code == 0 else f"error:{code}"

    return chan


# ── audio channel ─────────────────────────────────────────────────────────

def _build_audio_channel(client=None) -> MutableChannel:
    """构建 G1 音频灯光子 channel。"""

    chan = new_channel(name="audio", description="G1 audio — TTS/LED/volume control")

    @chan.build.command(always_observe=False)
    async def say(text: str, speaker_id: int = 0) -> str:
        """TTS 语音合成并播放。

        text: 要说的文本
        speaker_id: 音色 ID (0=默认)
        """
        if client is None:
            return "no client"
        code = client.TtsMaker(text, speaker_id)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=True)
    async def get_volume() -> str:
        """获取当前音量 (0-100)。"""
        if client is None:
            return "no client"
        code, vol = client.GetVolume()
        if code != 0:
            return f"error:{code}"
        # GetVolume 返回 (code, dict) — dict 形如 {"volume": N}
        if isinstance(vol, dict):
            return f"ok: {vol.get('volume', vol)}"
        return f"ok: {vol}"

    @chan.build.command(always_observe=False)
    async def set_volume(volume: int) -> str:
        """设置音量。volume: 0-100。"""
        if client is None:
            return "no client"
        code = client.SetVolume(volume)
        return "ok" if code == 0 else f"error:{code}"

    @chan.build.command(always_observe=False)
    async def led_control(r: int, g: int, b: int) -> str:
        """控制 G1 机身 RGB LED。

        r, g, b: 0-255 颜色分量。
        """
        if client is None:
            return "no client"
        code = client.LedControl(r, g, b)
        return "ok" if code == 0 else f"error:{code}"

    return chan


# ── main channel builder ──────────────────────────────────────────────────

def build_g1_channel(
    loco_client=None,
    arm_client=None,
    audio_client=None,
) -> PrimeChannel:
    """构建完整的 G1 身体控制 channel 树。

    所有 SDK client 参数可选:
    - 传 None → 命令返回 "no client" (安全降级)
    - 传 mock → 本地测试
    - 传真实 client (LocoClient, G1ArmActionClient, AudioClient) → PC2 实机

    返回的 channel 可直接用于:
      - chan.bootstrap() → 本地测试
      - Matrix.provide_channel(chan) → 生产运行
    """
    main = new_channel(
        name="bodies_g1",
        description="Unitree G1 人形机器人身体控制: 运动/手臂/音频",
    )

    loco_chan = _build_loco_channel(loco_client)
    arm_chan = _build_arm_channel(arm_client)
    audio_chan = _build_audio_channel(audio_client)

    main.import_channels(loco_chan, arm_chan, audio_chan)

    main.build.instruction(
        "G1 body channel — loco/arm/audio 三组命令。"
        "当前开发阶段: channel 本地测试中 (阶段 E 前置)。"
        "实机验证前先在本地通过 mock SDK 验证命令注册、参数传递、安全约束。"
    )

    return main


# ── production entry — used by main.py ────────────────────────────────────

def build_g1_channel_production():
    """生产入口: 从 SDK 创建真实 client 并构建 channel。

    仅在 PC2 上可用 (需要 cyclonedds + unitree_sdk2_python)。
    macOS 上 import 会失败 — 这是预期行为。
    """
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()

    arm = G1ArmActionClient()
    arm.SetTimeout(10.0)
    arm.Init()

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    return build_g1_channel(loco_client=loco, arm_client=arm, audio_client=audio)
