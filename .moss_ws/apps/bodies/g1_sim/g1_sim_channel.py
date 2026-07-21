from __future__ import annotations

import asyncio
import json

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.channel import ChannelCtx

from control.interface import VelocityRobotController
from control.obs import snapshot_to_dict


async def _run_for_duration(
    controller: VelocityRobotController,
    vx: float,
    vy: float,
    vyaw: float,
    duration: float,
) -> str:
    health = controller.get_health()
    if not health.ready_for_motion:
        reason = health.reason or "robot is not ready for motion"
        return f"blocked: {reason}"
    controller.set_velocity_command(vx, vy, vyaw)
    if duration > 0:
        try:
            await asyncio.sleep(duration)
        finally:
            controller.stand()
        return f"ok: cmd=({vx:.2f}, {vy:.2f}, {vyaw:.2f}) for {duration:.2f}s"
    return f"ok: cmd=({vx:.2f}, {vy:.2f}, {vyaw:.2f}) continuous"


async def _wait_until_ready(
    controller: VelocityRobotController,
    timeout: float,
    poll: float,
) -> str:
    timeout = max(0.0, float(timeout))
    poll = max(0.02, float(poll))
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        health = controller.get_health()
        if health.ready_for_motion:
            return f"ok: ready (phase={health.phase}, height={health.base_height:.3f})"
        if asyncio.get_running_loop().time() >= deadline:
            reason = health.reason or "robot is still not ready for motion"
            return f"blocked: timeout waiting ready; {reason}"
        await asyncio.sleep(poll)


def build_g1_sim_channel(controller: VelocityRobotController):
    channel = new_channel(
        name="bodies_g1_sim",
        description="G1 pure software locomotion demo. Supports stand-first high-level commands like prepare/recover/forward/backward/left/right/walk/turn/move/stop/state/health.",
    )
    channel.build.with_binding(VelocityRobotController, controller)

    @channel.build.command(always_observe=False)
    async def walk(vx: float = 0.4, duration: float = 2.0) -> str:
        """让机器人向前/后行走。vx>0 前进, vx<0 后退。duration<=0 表示持续行走。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, vx, 0.0, 0.0, duration)

    @channel.build.command(always_observe=False)
    async def forward(speed: float = 0.4, duration: float = 2.0) -> str:
        """更自然的前进别名。speed 会自动取正值。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, abs(speed), 0.0, 0.0, duration)

    @channel.build.command(always_observe=False)
    async def go_forward(speed: float = 0.4) -> str:
        """持续前进，直到收到 stop()/end_showcase() 或新的移动命令。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, abs(speed), 0.0, 0.0, 0.0)

    @channel.build.command(always_observe=False)
    async def backward(speed: float = 0.25, duration: float = 1.5) -> str:
        """更自然的后退别名。speed 会自动取正值后转成负向前进速度。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, -abs(speed), 0.0, 0.0, duration)

    @channel.build.command(always_observe=False)
    async def go_backward(speed: float = 0.25) -> str:
        """持续后退，直到收到 stop()/end_showcase() 或新的移动命令。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, -abs(speed), 0.0, 0.0, 0.0)

    @channel.build.command(always_observe=False)
    async def turn(vyaw: float = 0.5, duration: float = 1.5) -> str:
        """原地转向。vyaw>0 左转, vyaw<0 右转。duration<=0 表示持续转向。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, 0.0, 0.0, vyaw, duration)

    @channel.build.command(always_observe=False)
    async def left(speed: float = 0.5, duration: float = 1.5) -> str:
        """更自然的左转别名。speed 会自动取正值。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, 0.0, 0.0, abs(speed), duration)

    @channel.build.command(always_observe=False)
    async def keep_left(speed: float = 0.5) -> str:
        """持续左转，直到收到 stop()/end_showcase() 或新的移动命令。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, 0.0, 0.0, abs(speed), 0.0)

    @channel.build.command(always_observe=False)
    async def right(speed: float = 0.5, duration: float = 1.5) -> str:
        """更自然的右转别名。speed 会自动取正值后转成负 yaw。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, 0.0, 0.0, -abs(speed), duration)

    @channel.build.command(always_observe=False)
    async def keep_right(speed: float = 0.5) -> str:
        """持续右转，直到收到 stop()/end_showcase() 或新的移动命令。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, 0.0, 0.0, -abs(speed), 0.0)

    @channel.build.command(always_observe=False)
    async def move(vx: float, vy: float, vyaw: float, duration: float = 1.0) -> str:
        """底层速度控制：同时设定前后/横移/转向速度。duration<=0 表示持续移动。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        return await _run_for_duration(c, vx, vy, vyaw, duration)

    @channel.build.command(always_observe=False)
    async def stop() -> str:
        """停止当前动作，并优先回到站稳状态。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        c.stand()
        return "ok: stand requested"

    @channel.build.command(always_observe=False)
    async def end_showcase() -> str:
        """结束展示/巡逻/表演，立即停下并回到站稳。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        c.stand()
        return "ok: showcase ended; stand requested"

    @channel.build.command(always_observe=False)
    async def stand() -> str:
        """进入站稳优先模式。若已跌倒，会触发 reset-to-stand。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        before = c.get_health()
        c.stand()
        if before.fallen or before.phase in ("booting", "resetting"):
            return "ok: reset-to-stand requested"
        return "ok: stand requested"

    @channel.build.command(always_observe=False)
    async def recover() -> str:
        """更自然的恢复站稳语义。等价于 stand()。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        before = c.get_health()
        c.stand()
        if before.fallen or before.phase in ("booting", "resetting"):
            return "ok: reset-to-stand requested"
        return "ok: stand requested"

    @channel.build.command(always_observe=False)
    async def prepare(timeout: float = 3.0, poll: float = 0.05) -> str:
        """进入 stand-first 恢复流程，并等待到可运动状态再返回。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        c.stand()
        return await _wait_until_ready(c, timeout, poll)

    @channel.build.command(always_observe=False)
    async def reset() -> str:
        """强制重置到标准站姿，并在短暂稳定窗口后允许运动。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        c.reset_pose()
        return "ok: reset requested"

    @channel.build.command(always_observe=True)
    async def health() -> str:
        """返回当前是否可运动、是否跌倒、所处阶段与原因。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        health_info = c.get_health()
        data = {
            "ready_for_motion": health_info.ready_for_motion,
            "fallen": health_info.fallen,
            "phase": health_info.phase,
            "reason": health_info.reason,
            "base_height": round(health_info.base_height, 3),
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @channel.build.command(always_observe=True)
    async def state() -> str:
        """返回机器人当前状态（速度指令、位姿、控制后端、错误信息）。"""
        c = ChannelCtx.get_contract(VelocityRobotController)
        snapshot = c.get_snapshot()
        health_info = c.get_health()
        data = {
            "summary": snapshot.summary(),
            "command": snapshot.command.as_tuple(),
            "base_state": snapshot_to_dict(snapshot.base_state),
            "health": {
                "ready_for_motion": health_info.ready_for_motion,
                "fallen": health_info.fallen,
                "phase": health_info.phase,
                "reason": health_info.reason,
                "base_height": round(health_info.base_height, 3),
            },
            "observation_dim": int(snapshot.observation.size),
            "last_error": snapshot.last_error,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @channel.build.context_messages
    async def context() -> list[str]:
        snapshot = controller.get_snapshot()
        health_info = controller.get_health()
        return [
            f"[bodies/g1_sim] {snapshot.summary()}",
            f"Health: phase={health_info.phase}, ready={health_info.ready_for_motion}, fallen={health_info.fallen}",
            "Commands: prepare(timeout, poll), recover(), reset(), stand(), forward(speed, duration), backward(speed, duration), left(speed, duration), right(speed, duration), go_forward(speed), go_backward(speed), keep_left(speed), keep_right(speed), walk(vx, duration), turn(vyaw, duration), move(vx, vy, vyaw, duration), stop(), end_showcase(), state(), health()",
            "Natural mapping: '站稳/恢复/别动' -> prepare()/recover()/stand(); '往前走/后退' -> forward()/backward(); '左转/右转' -> left()/right(); '一直往前走/持续前进' -> go_forward(); '一直后退' -> go_backward(); '一直左转/一直右转' -> keep_left()/keep_right(); '停下/结束展示/结束表演' -> stop()/end_showcase().",
            "Rule: stand-first. If fallen or resetting, call prepare()/recover()/reset()/stand() before motion commands. If you need to move immediately after a reset, prefer prepare().",
        ]

    channel.build.instruction(
        "G1 纯仿真身体控制。优先使用 stand-first 高层语义：prepare/recover/reset/stand/forward/backward/left/right/stop/state/health。"
        "对自然语言优先做高层映射：'站稳、恢复、别动、重新站好、准备好再走' -> prepare()/recover()/stand()；"
        "'往前走、向前走' -> forward()；'后退' -> backward()；'左转' -> left()；'右转' -> right()。"
        "如果用户说'一直往前走、持续前进、继续往前走直到我说停'，优先使用 go_forward()；"
        "'一直后退、持续后退' -> go_backward()；'一直左转、持续左转' -> keep_left()；'一直右转、持续右转' -> keep_right()。"
        "如果用户说'停下、停止、别动、结束展示、结束表演、结束巡逻'，优先使用 stop() 或 end_showcase()。"
        "只有在用户明确要求同时控制前后/横移/转向时，才使用 move(vx, vy, vyaw, duration)。"
        "不要臆造蹲下、起立、跳跃、手势等命令。"
        "机器人跌倒或仍在 reset 阶段时，不要直接发移动命令；应先 prepare()/recover()/reset()/stand()。"
        "若用户只给方向意图而没给参数，短时演示可用 forward(0.4, 2.0)、backward(0.25, 1.5)、left(0.5, 1.5)、right(0.5, 1.5)；语音指挥中的持续动作优先使用 go_forward()/go_backward()/keep_left()/keep_right()。"
        "往前走用 walk(vx>0)，后退用 walk(vx<0)，左转用 turn(vyaw>0)，右转用 turn(vyaw<0)。"
        "建议 vx 不超过 0.6 m/s，vyaw 不超过 0.8 rad/s。"
    )
    return channel
