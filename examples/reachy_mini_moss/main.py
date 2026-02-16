import asyncio
import logging

import cv2
from PIL import Image
from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE

from examples.reachy_mini_moss.reachy_mini_dances_library import DanceMove
from examples.reachy_mini_moss.reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from examples.reachy_mini_moss.vision.head_tracker import HeadTracker
from ghoshell_moss import PyChannel, Message, Base64Image, Text

logger = logging.getLogger('reachy_mini_moss')
logger.setLevel(logging.INFO)


class ReachyMiniMoss:
    def __init__(self, mini: ReachyMini):
        self.mini = mini

        self._head_tracker = HeadTracker(mini)
        self._waken = asyncio.Event()

        self._bootstrapped = asyncio.Event()

        self._is_keep_looking_user = False

    async def wake_up(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        self._waken.set()

    async def goto_sleep(self):
        self._is_keep_looking_user = False
        self.mini.goto_sleep()
        self.mini.disable_motors()
        self._waken.clear()

    async def dance(self, name: str):
        await self.mini.async_play_move(DanceMove(name))

    async def keep_looking_user(self, yes: bool=True):
        self._is_keep_looking_user = yes

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")

        # mini vision
        frame = self.mini.media.get_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            img_pil = Image.fromarray(frame_rgb)
            # img_pil.save("temp.png")
            msg.with_content(
                Text(text="This image is what you see")
            ).with_content(
                Base64Image.from_pil_image(img_pil)
            )

        msg.with_content(
            Text(text=f"Current head pose is {self.mini.get_current_head_pose()}")
        )

        if self._is_keep_looking_user:
            msg.with_content(
                Text(text=f"You are keep looking user")
            )

        return [msg]

    async def on_policy_run(self):
        logger.info(f"Running on-policy run, waken is {self._waken.is_set()}")
        # await self._waken.wait()
        # double check
        if self._waken.is_set() and self._is_keep_looking_user:
            logger.info("self._head_tracker.enabled.set()")
            self._head_tracker.enabled.set()

    async def on_policy_pause(self):
        logger.info("Running on-policy pause")
        self._head_tracker.enabled.clear()

    def as_channel(self) -> PyChannel:
        logger.info("as channel")
        assert self._bootstrapped.is_set()

        chan = PyChannel(name="reachy_mini", description=f"sleep head pose is {SLEEP_HEAD_POSE}", block=True)

        chan.build.with_context_messages(self.context_messages)

        chan.build.on_policy_run(self.on_policy_run)
        chan.build.on_policy_pause(self.on_policy_pause)

        chan.build.command(doc=f"Dance name can be chosen in {AVAILABLE_MOVES.keys()}")(self.dance)
        chan.build.command()(self.wake_up)
        chan.build.command()(self.goto_sleep)
        chan.build.command()(self.keep_looking_user)

        return chan

    async def bootstrap(self):
        await self._head_tracker.start()
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self._head_tracker.stop()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


async def run_agent(container, speech, root_dir):
    from ghoshell_moss import new_shell
    from ghoshell_moss_contrib.agent import SimpleAgent, ModelConf
    from ghoshell_moss.transports.zmq_channel import ZMQChannelHub
    from ghoshell_moss.transports.zmq_channel.zmq_hub import ZMQHubConfig, ZMQProxyConfig

    # hub channel
    zmq_hub = ZMQChannelHub(
        config=ZMQHubConfig(
            name="hub",
            description="可以启动指定的子通道并运行.",
            # todo: 当前版本全部基于约定来做. 快速验证.
            root_dir=root_dir,
            proxies={
                "slide": ZMQProxyConfig(
                    script="slide_app.py",
                    description="可以打开你的slide studio gui，通过这个通道你可以呈现并讲述一个slide主题",
                ),
            },
        ),
    )

    with ReachyMini() as _mini:
        async with ReachyMiniMoss(_mini) as moss:
            shell = new_shell(container=container, speech=speech)
            shell.main_channel.import_channels(
                moss.as_channel(),
                # zmq_hub.as_channel()
            )
            agent = SimpleAgent(
                instruction="你叫小灵，拥有一个reachy mini的实体机器人，你的性格热情可爱，并且给你配置了TTS功能，你说的话都可以被用户听到哦",
                shell=shell,
                speech=speech,
                model=ModelConf(
                    kwargs={
                        "thinking": {
                            "type": "disabled",
                        },
                    },
                ),
                container=container,
            )

            await agent.run()


def main():
    import pathlib
    ws_dir = pathlib.Path(__file__).parent.parent.joinpath(".workspace")
    current_dir = pathlib.Path(__file__).parent
    root_dir = str(current_dir.parent.joinpath("moss_zmq_channels").absolute())

    from ghoshell_moss_contrib.example_ws import get_example_speech, workspace_container

    with workspace_container(ws_dir) as container:
        speech = get_example_speech(container, default_speaker="saturn_zh_female_keainvsheng_tob")
        asyncio.run(run_agent(container, speech, root_dir))

if __name__ == "__main__":
    main()


