"""Reachy Mini body node.

The daemon at `REACHY_DAEMON` (default http://127.0.0.1:8000) owns the hardware; this node
is one thin command surface over its REST API — head pose, antennas, wake / sleep, state.

Channel: reachy
CTML:    <reachy:look pitch="20"/>

Angles in this channel are **degrees**; the wire is radians.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.error
import urllib.request

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

DAEMON = os.getenv("REACHY_DAEMON", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("REACHY_TIMEOUT", "10"))


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(DAEMON + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read().decode("utf-8", "replace")
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body[:400]}


async def _call(method: str, path: str, payload: dict | None = None) -> dict:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _request, method, path, payload)
    except urllib.error.URLError as e:
        raise RuntimeError(f"reachy daemon unreachable at {DAEMON}: {e}") from e


def _rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


async def _goto(
    *,
    yaw: float | None = None,
    pitch: float | None = None,
    roll: float | None = None,
    left: float | None = None,
    right: float | None = None,
    body_yaw: float | None = None,
    duration: float = 0.8,
    interpolation: str = "minjerk",
) -> dict:
    """One goto with only the parts that were asked for."""
    payload: dict = {"duration": float(duration), "interpolation": interpolation}
    if yaw is not None or pitch is not None or roll is not None:
        payload["head_pose"] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": _rad(roll or 0.0),
            "pitch": _rad(pitch or 0.0),
            "yaw": _rad(yaw or 0.0),
        }
    if left is not None or right is not None:
        payload["antennas"] = [_rad(left or 0.0), _rad(right or 0.0)]
    if body_yaw is not None:
        payload["body_yaw"] = _rad(body_yaw)
    return await _call("POST", "/api/move/goto", payload)


async def _await_move(ack: dict, duration: float = 0.0, timeout: float = 15.0) -> None:
    """Poll until the move we just asked for is no longer running.

    The daemon acks the request, not the motion: without this, a command returns while
    the body is still travelling, and anything sequenced after it races the body.
    """
    uuid = ack.get("uuid")
    if not uuid:
        return
    # The daemon registers the move a beat after the ack, so a check made too early sees
    # an empty list and believes the body is already there. Waiting out the motion's own
    # length first closes that hole, and makes the guarantee "at least as long as asked".
    if duration > 0:
        await asyncio.sleep(duration)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await asyncio.sleep(0.05)
        running = await _call("GET", "/api/move/running")
        if not isinstance(running, list) or uuid not in running:
            return


async def main(matrix: Matrix) -> None:
    channel = new_channel(
        name="reachy",
        description=(
            "Reachy Mini body — the robot on the desk. Head pose, antennas, wake / sleep, "
            "and a live state read. Angles are degrees."
        ),
    )

    @channel.build.command(always_observe=True)
    async def status() -> str:
        """Read the body: motor mode, head pose, body yaw, antennas (all radians, as the daemon reports)."""
        state = await _call("GET", "/api/state/full")
        motors = await _call("GET", "/api/motors/status")
        return json.dumps(
            {"motors": motors.get("mode"), "state": state}, ensure_ascii=False
        )

    @channel.build.command()
    async def look(
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        duration: float = 0.8,
        wait: bool = True,
    ) -> str:
        """Turn the head. yaw = left / right, pitch = up / down, roll = tilt. Degrees.

        :param yaw: degrees, positive turns one way
        :param pitch: degrees, positive looks up
        :param roll: degrees, tilt
        :param duration: seconds the motion takes
        :param wait: block until the head has actually arrived
        """
        ack = await _goto(yaw=yaw, pitch=pitch, roll=roll, duration=duration)
        if wait:
            await _await_move(ack, duration)
        return f"look yaw={yaw} pitch={pitch} roll={roll} ({duration}s)"

    @channel.build.command()
    async def center(duration: float = 0.6) -> str:
        """Bring the head back to neutral."""
        await _goto(yaw=0.0, pitch=0.0, roll=0.0, duration=duration)
        return "centered"

    @channel.build.command()
    async def antennas(
        left: float = 0.0, right: float = 0.0, duration: float = 0.6, wait: bool = True
    ) -> str:
        """Move the two antennas. Degrees; positive lifts.

        :param left: left antenna, degrees
        :param right: right antenna, degrees
        :param duration: seconds the motion takes
        :param wait: block until the antennas have actually arrived
        """
        ack = await _goto(left=left, right=right, duration=duration)
        if wait:
            await _await_move(ack, duration)
        return f"antennas left={left} right={right} ({duration}s)"

    @channel.build.command(always_observe=True)
    async def nod(times: int = 2, depth: float = 12.0) -> str:
        """Nod the head — waits until the nod is finished, so the answer means it happened.

        :param times: how many nods
        :param depth: how far down, degrees
        """
        step = max(0.25, 0.5)
        for _ in range(max(1, times)):
            await _await_move(await _goto(pitch=-depth, duration=step), step)
            await _await_move(await _goto(pitch=0.0, duration=step), step)
        return f"nodded {times}x{int(depth)}deg"

    @channel.build.command(always_observe=True)
    async def wiggle(times: int = 2) -> str:
        """Shake the antennas side to side — the cheap "I'm here" gesture."""
        for _ in range(max(1, times)):
            await _await_move(await _goto(left=28.0, right=-12.0, duration=0.28), 0.28)
            await _await_move(await _goto(left=-12.0, right=28.0, duration=0.28), 0.28)
        await _goto(left=0.0, right=0.0, duration=0.3)
        return f"wiggled {times}x"

    @channel.build.command(always_observe=True)
    async def dance(times: int = 2, tempo: float = 0.3) -> str:
        """Sway the head and the antennas side to side — a short, silly dance.

        :param times: how many left-right cycles
        :param tempo: seconds per half beat
        """
        for _ in range(max(1, times)):
            await _await_move(
                await _goto(yaw=-24.0, roll=-14.0, left=36.0, right=32.0, duration=tempo),
                tempo,
            )
            await _await_move(
                await _goto(yaw=24.0, roll=14.0, left=-10.0, right=-10.0, duration=tempo),
                tempo,
            )
        await _goto(yaw=0.0, pitch=0.0, roll=0.0, left=0.0, right=0.0, duration=0.5)
        return f"danced {times}x"

    @channel.build.command(always_observe=True)
    async def wake(up: bool = True) -> str:
        """Wake the robot up, or let it go back to sleep.

        :param up: True = wake up, False = go to sleep
        """
        path = "/api/move/play/wake_up" if up else "/api/move/play/goto_sleep"
        await _call("POST", path, {})
        return "awake" if up else "asleep"

    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
