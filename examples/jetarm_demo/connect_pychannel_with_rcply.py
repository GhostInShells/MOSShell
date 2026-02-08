import asyncio

from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProxy

trajectory = """
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

JETARM_ADDRESS = "tcp://192.168.1.15:9527"
"""这个 ip 地址需要根据 jetarm 在局域网内的实际地址进行修改. """


async def main():
    """
    测试 jetarm 的脚本, 通过 zmq proxy 调用 zmq provider 的方式, 与 jetarm channel 进行通讯.
    然后测试脚本可以运行.
    """
    chan = ZMQChannelProxy(
        name="jetarm",
        address=JETARM_ADDRESS,
    )

    async with chan.bootstrap() as broker:
        await broker.refresh_meta()
        meta = broker.meta()
        print(meta.model_dump_json(indent=2))
        cmd = broker.get_command("run_trajectory")

        print("+++++++", cmd.meta())

        r = await cmd(trajectory)
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
