import asyncio
import logging
import os

import cv2
from PIL import Image
from ghoshell_common.contracts import LoggerItf
from ghoshell_container import Container, get_container
from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE

from examples.moss_agent import load_instructions
from examples.reachy_mini_moss.audio.player import ReachyMiniStreamPlayer
from examples.reachy_mini_moss.channels.antennas import Antennas
from examples.reachy_mini_moss.channels.head import Head
from examples.reachy_mini_moss.reachy_mini_dances_library import DanceMove
from examples.reachy_mini_moss.reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
from examples.reachy_mini_moss.state import ReachyMiniState, BodyYawMove
from ghoshell_moss import PyChannel, Message, Base64Image, Text, Speech

logger = logging.getLogger('reachy_mini_moss')
logger.setLevel(logging.INFO)


class ReachyMiniMoss:
    def __init__(self, mini: ReachyMini):
        self.mini = mini
        self.mini.set_target_body_yaw(0.0)
        self._state = ReachyMiniState()
        self._state.twisting.set()

        self._head = Head(mini, self._state, logger)
        self._antennas = Antennas(mini, self._state, logger)

        self._bootstrapped = asyncio.Event()
        self._enable_twisting = asyncio.Event()

    async def wake_up(self):
        self.mini.enable_motors()
        self.mini.wake_up()
        self._state.waken.set()
        await asyncio.sleep(1)

    async def goto_sleep(self):
        self._state.tracking.clear()
        self.mini.goto_sleep()
        self.mini.disable_motors()
        self._state.waken.clear()

    async def dance(self, name: str):
        await self.mini.async_play_move(DanceMove(name))

    async def context_messages(self):
        msg = Message.new(role="user", name="__reachy_mini__")

        appearance_img = Image.open("appearance.png")
        structure_img = Image.open("structure.png")
        msg.with_content(
            Text(text="These two images shows your appearance and structure"),
            Base64Image.from_pil_image(appearance_img),
            Base64Image.from_pil_image(structure_img),
        )

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

        if self._state.twisting.is_set():
            msg.with_content(
                Text(text="You are twisting on idle.")
            )

        return [msg]

    async def start_twisting(self):
        self._state.twisting.set()

    async def stop_twisting(self):
        self.mini.set_target_body_yaw(0.0)
        self._state.twisting.clear()
        self._enable_twisting.clear()

    async def on_policy_run(self):
        if not self._state.waken.is_set():
            return
        if not self._state.twisting.is_set():
            return

        self._enable_twisting.set()
        while self._enable_twisting.is_set():
            await self.mini.async_play_move(
                BodyYawMove(self._state.start_body_yaw, 10, 1.5),
            )
            self._state.start_body_yaw = 10
            await self.mini.async_play_move(
                BodyYawMove(self._state.start_body_yaw, -10, 1.5),
            )
            self._state.start_body_yaw = -10

    async def on_policy_pause(self):
        self._enable_twisting.clear()

    def as_channel(self) -> PyChannel:
        logger.info("as channel")
        assert self._bootstrapped.is_set()

        body = PyChannel(name="reachy_mini", description=f"sleep head pose is {SLEEP_HEAD_POSE}", block=True)

        body.build.with_context_messages(self.context_messages)
        dance_docstrings = []
        for name, move in AVAILABLE_MOVES.items():
            func, params, meta = move
            dance_docstrings.append(f"name: {name} description: {meta.get("description", "")} subcycles per beat: {params.get('subcycles_per_beat', 1.0)}")

        body.build.command(doc=f"Dance can be chosen in \n{"\n".join(dance_docstrings)}")(self.dance)
        body.build.command()(self.wake_up)
        body.build.command()(self.goto_sleep)

        body.build.command()(self.start_twisting)
        body.build.command()(self.stop_twisting)
        body.build.on_policy_run(self.on_policy_run)
        body.build.on_policy_pause(self.on_policy_pause)

        body.import_channels(
            self._head.as_channel(),
            self._antennas.as_channel(),
        )

        return body

    async def bootstrap(self):
        await self._head.bootstrap()
        self._bootstrapped.set()

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def aclose(self):
        await self._head.aclose()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


async def run_agent(container, root_dir):
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
            speech = get_speech(_mini, container, default_speaker="zh_female_santongyongns_saturn_bigtts")
            shell = new_shell(container=container, speech=speech)
            shell.main_channel.import_channels(
                moss.as_channel(),
                # zmq_hub.as_channel()
            )
            instructions = load_instructions(
                container,
                ["persona.md"],
                "reachy_mini_instructions",
            )
            agent = SimpleAgent(
                instruction=instructions,
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


def get_speech(
    mini: ReachyMini,
    container: Container | None = None,
    default_speaker: str | None = None,
) -> Speech:
    from ghoshell_moss.speech import TTSSpeech
    from ghoshell_moss.speech.mock import MockSpeech
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf

    container = container or get_container()
    use_voice = os.environ.get("USE_VOICE_SPEECH", "no") == "yes"
    if not use_voice:
        return MockSpeech()
    app_key = os.environ.get("VOLCENGINE_STREAM_TTS_APP")
    app_token = os.environ.get("VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    resource_id = os.environ.get("VOLCENGINE_STREAM_TTS_RESOURCE_ID", "seed-tts-2.0")
    if not app_key or not app_token:
        raise NotImplementedError(
            "Env $VOLCENGINE_STREAM_TTS_APP or $VOLCENGINE_STREAM_TTS_ACCESS_TOKEN not configured."
            "Maybe examples/.env not set, or you need to set $USE_VOICE_SPEECH `no`"
        )
    tts_conf = VolcengineTTSConf(
        app_key=app_key,
        access_token=app_token,
        resource_id=resource_id,
        sample_rate=mini.media.get_output_audio_samplerate(),
    )
    if default_speaker:
        tts_conf.default_speaker = default_speaker
    return TTSSpeech(player=ReachyMiniStreamPlayer(mini), tts=VolcengineTTS(conf=tts_conf), logger=container.get(LoggerItf))


def main():
    import pathlib
    ws_dir = pathlib.Path(__file__).parent.parent.joinpath(".workspace")
    current_dir = pathlib.Path(__file__).parent
    root_dir = str(current_dir.parent.joinpath("moss_zmq_channels").absolute())

    from ghoshell_moss_contrib.example_ws import workspace_container

    with workspace_container(ws_dir) as container:
        asyncio.run(run_agent(container, root_dir))

if __name__ == "__main__":
    main()


