import json
import logging
from typing import Any

from ghoshell_common.contracts import Workspace, WorkspaceConfigs, YamlConfig
from ghoshell_container import IoCContainer
from pydantic import BaseModel, Field
from python_mpv_jsonipc import MPV

from ghoshell_moss import PyChannel

"""
# 介绍

基于 mpv 实现的视频播放 channel. 目前是基线测试. 未来的目标是让 AI 陪伴人类看本地视频.  

预计 Beta 实现的功能: 

1. 基于 mpv 操作一个视频的播放, 关闭等. 
2. 通过播放时间轴, 让模型知道播放到什么位置. 通过抽帧, 让模型看到最近几帧的画面. 
3. 通过字幕提取, 让模型可以看到整个视频的上下文, 而且知道当前播放的上下文位置. 
4. 模型可以对单个视频标记一个数据结构, 用来存储指定片段的时间位置/播放速度/描述. 
5. 模型可以随时播放某个已经存储过的片段, 而且知道这个片段的讯息和如何描述. 
"""

# todo: 收敛到通过 ioc 容器获取 logger
logger = logging.getLogger(__name__)

# 提供给大模型看到的参数
READABLE_PROPERTIES = [
    "duration",
    "eof_reached",
    "playlist",
]
WRITABLE_PROPERTIES = [
    "loop",
    "speed",
    "volume",
    "pause",
]
EXCLUDE_COMMANDS = [
    "play",
    "stop",
    "pause",
    "load_file",
]

mpv_chan = PyChannel(name="mpv_player")


class VideoInfo(BaseModel):
    filename: str = Field(default="", description="视频文件的路径")
    description: str = Field(default="", description="视频文件的描述")


class VideoConfig(YamlConfig):
    relative_path = "video.yaml"

    video_list: list[VideoInfo] = Field(default_factory=list, description="视频播放列表")

    @classmethod
    def load(cls, container: IoCContainer):
        ws = container.force_fetch(Workspace)
        configs = WorkspaceConfigs(ws.configs())
        return configs.get_or_create(VideoConfig())

    def to_str(self):
        return "\n".join(
            [f"{i + 1}. filename:{v.filename} description:{v.description}" for i, v in enumerate(self.video_list)]
        )


def create_property_setter_getter(prop_name):
    """创建独立的setter函数"""

    async def set_prop(value):
        mpv = mpv_chan.broker.container.force_fetch(MPV)
        setattr(mpv, prop_name, value)
        logger.info("Set %s to %s", prop_name, value)

    async def get_prop():
        mpv = mpv_chan.broker.container.force_fetch(MPV)
        value = getattr(mpv, prop_name)
        logger.info("Get %s = %s", prop_name, value)
        return value

    # 设置函数名（可选，便于调试）
    set_prop.__name__ = f"set_{prop_name}"
    get_prop.__name__ = f"get_{prop_name}"
    return set_prop, get_prop


def create_command_executor(command_name: str, command_args: list[dict[str, Any]]):
    async def command_executor(text__: str):
        mpv = mpv_chan.broker.container.force_fetch(MPV)
        command = getattr(mpv, command_name)
        kwargs = json.loads(text__)
        args = []
        for c_arg in command_args:
            c_value = kwargs.get(c_arg["name"])
            if c_value is None:
                continue
            args.append(c_value)
        command(*args)

    command_executor.__name__ = command_name
    return command_executor


@mpv_chan.build.with_description()
def description():
    video_config = VideoConfig.load(mpv_chan.broker.container)

    states = ""
    mpv = mpv_chan.broker.container.force_fetch(MPV)
    for prop in READABLE_PROPERTIES + WRITABLE_PROPERTIES:
        states += f"{prop}: {getattr(mpv, prop)}\n"

    desc = f"""
video file list:
{video_config.to_str()}

mpv player states:
{states}
"""
    return desc


@mpv_chan.build.command()
def play_ws_video(filename: str):
    """
    play a video file which is under local workspace
    """
    ws = mpv_chan.broker.container.force_fetch(Workspace)
    mpv = mpv_chan.broker.container.force_fetch(MPV)
    path = ws.assets().sub_storage("video").abspath() + "/" + filename
    mpv.play(path)
    mpv.wait_for_property("duration")


@mpv_chan.build.command()
def play_ws_audio(filename: str):
    """
    play a audio file which is under local workspace
    """
    ws = mpv_chan.broker.container.force_fetch(Workspace)
    mpv = mpv_chan.broker.container.force_fetch(MPV)
    path = ws.assets().sub_storage("audio").abspath() + "/" + filename
    mpv.play(path)
    mpv.wait_for_property("duration")


@mpv_chan.build.command()
def stop():
    """
    stop current playing video or audio
    """
    mpv = mpv_chan.broker.container.force_fetch(MPV)
    mpv.stop()


def build_mpv_chan(container: IoCContainer):
    mpv = MPV()
    container.set(MPV, mpv)

    # build mpv property setter to channel
    for prop in WRITABLE_PROPERTIES:
        if prop not in mpv.properties:
            logger.warning("Property %s is not defined.", prop)
            continue

        setter, getter = create_property_setter_getter(prop)
        mpv_chan.build.command(doc=f"""set mpv property {prop}=value""")(setter)
        # using description instead of get property
        # mpv_chan.build.command(doc=f"""get mpv property {prop}""")(getter)

    # build mpv command to channel
    for mpv_cmd in mpv.command_list:
        mpv_cmd_name = mpv_cmd["name"].replace("-", "_")
        if mpv_cmd_name in ["play", "stop", "pause", "load_file"]:
            continue
        if mpv_cmd_name in mpv.properties:
            mpv_cmd_name = f"{mpv_cmd_name}_cmd"

        func = create_command_executor(mpv_cmd_name, mpv_cmd["args"])

        mpv_cmd_args = mpv_cmd["args"]
        mpv_chan.build.command(
            doc=(
                f"{mpv_cmd_name} is a mpv command.\n"
                f":params text__: 用 json 序列化的字典类型结构, 其参数定义是{mpv_cmd_args}"
            )
        )(func)

    return mpv_chan
