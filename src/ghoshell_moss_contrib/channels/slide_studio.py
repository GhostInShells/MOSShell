import logging
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import frontmatter
from PIL import Image
from ghoshell_common.contracts import YamlConfig, WorkspaceConfigs, LoggerItf, Storage, FileStorage
from ghoshell_common.helpers import timestamp_ms
from ghoshell_container import IoCContainer, Container
from pydantic import Field, PrivateAttr

from ghoshell_moss import PyChannel, CommandError, CommandErrorCode
from ghoshell_moss.message import Message, Base64Image, Text
from ghoshell_moss_contrib.gui.image_viewer import SimpleImageViewer


class SlideAssetInfo(YamlConfig):
    relative_path = ".meta.yaml"

    name: str = Field(default="", description="Slide的标识名")  # 面向人和大模型的文本描述
    description: str = Field(default="", description="Slide的描述")  # 面向人和大模型的文本描述

    # system unmodified fields
    created_at: Optional[float] = Field(default_factory=timestamp_ms, description="创建时间")
    updated_at: Optional[float] = Field(default_factory=timestamp_ms, description="更新时间")
    origin_filetype: str = Field(default="", description="Slide的原始文件类型")  # ppt/pptx/pdf/...
    origin_filepath: str = Field(default="", description="Slide的原始文件地址")

    # extension unmarshalled fields
    _dirname: str = PrivateAttr()

    @property
    def origin_filename(self):
        return f".origin.{self.origin_filetype}"

    @property
    def dirname(self):
        return self._dirname


class SlideFrame:
    def __init__(self):
        self.filename: Optional[str] = None
        self.image: Optional[Image.Image] = None  # 页图片展示
        self.title: Optional[str] = None  # 页标题
        self.outline: Optional[str] = None  # 页提纲
        self.content: Optional[str] = None  # 页文本内容
        self.finished: bool = False

    def finish(self):
        self.finished = True


class SlideAssets:
    def __init__(self, storage: FileStorage):
        self._storage = storage
        self._slides: List[SlideAssetInfo] = []
        self.refresh()

    def refresh(self) -> List[SlideAssetInfo]:
        """
        同步最新素材库信息
        """

        slides = []

        with os.scandir(self._storage.abspath()) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name not in [".", ".."]:
                    slides.append(entry.name)

        res = []
        for slide_dirname in slides:
            slide_storage: FileStorage | Storage = self._storage.sub_storage(slide_dirname)
            configs = WorkspaceConfigs(slide_storage)
            slide_config = configs.get_or_create(SlideAssetInfo())
            slide_config._dirname = slide_dirname
            if not slide_config.name:
                slide_config.name = slide_dirname
            res.append(slide_config)
        self._slides = res
        return self._slides

    def list(self) -> List[SlideAssetInfo]:
        return self._slides

    def get(self, name: str) -> Optional[SlideAssetInfo]:
        for slide in self._slides:
            if slide.name == name:
                return slide
        return None

    def load_frames(self, name: str) -> Tuple[SlideAssetInfo, List[SlideFrame]]:
        info = self.get(name)
        if not info:
            raise CommandError(CommandErrorCode.NOT_AVAILABLE, message=f"slide {name} not found")

        slide_storage: FileStorage | Storage = self._storage.sub_storage(info.dirname)
        filenames = slide_storage.dir("", False)

        frame_map: defaultdict[str, SlideFrame] = defaultdict(SlideFrame)
        for filename in filenames:
            if filename.startswith("."):
                continue

            file_abs_path = slide_storage.abspath() + "/" + filename
            image_filename = filename.rstrip(".md")
            frame = frame_map[image_filename]
            frame.filename = image_filename

            if filename.endswith(".md"):
                post = frontmatter.load(file_abs_path)
                frame.title = post.metadata.get("title") or ""
                frame.outline = post.metadata.get("outline") or ""
                frame.content = post.content.strip()
            else:
                frame.image = Image.open(file_abs_path)

        frames = frame_map.values()
        return info, sorted(frames, key=lambda f: f.filename)

    def as_channel(self):
        pass


class SlideCreator:
    def __init__(self, assets: SlideAssets, logger: LoggerItf):
        self._assets = assets

    def as_channel(self) -> PyChannel:
        chan = PyChannel(name="creator")

        return chan


class SlidePlayer:
    def __init__(self, assets: SlideAssets, logger: LoggerItf):
        self._assets = assets
        self.logger = logger

        # gui
        self.viewer = SimpleImageViewer(window_title="Slide Studio Player")

        # current playing slide information
        self.is_playing = False
        self.loaded: Optional[SlideAssetInfo] = None
        self.frames: List[SlideFrame] = []
        self.current_frame_index = -1

    def _load_frames(self, slide_name: str):
        self._clear_frames()
        self.loaded, self.frames = self._assets.load_frames(slide_name)

    def _clear_frames(self):
        self.is_playing = False
        self.loaded = None
        self.frames.clear()
        self.current_frame_index = -1

    def _check_playing(self):
        if not self.is_playing:
            raise CommandError(CommandErrorCode.FAILED, message="slide studio is not playing")

    def _is_last(self):
        return self.current_frame_index == len(self.frames) - 1

    @property
    def _current_frame(self):
        return self.frames[self.current_frame_index]

    async def play(self, name: str):
        """
        Start playing target slide asset.

        @param name: slide asset name
        """
        try:
            self._load_frames(name)
        except Exception as ex:
            self.logger.warning(f"Player _load {name} failed with: {ex}")
            await self.stop()
            raise
        self.is_playing = True
        return await self.to_page(1)

    async def to_page(self, index: int = 1):
        """
        Jump to target page

        @param index: target page index
        """
        self._check_playing()
        if index >= len(self.frames):
            return False
        self.current_frame_index = index - 1
        self.viewer.set_pil_image(self._current_frame.image)
        self._current_frame.finish()
        return True

    async def to_next_page(self):
        """
        Jump to next page. Call this only after finishing the current page presentation. Never call it first.
        """
        self._check_playing()
        if self._is_last():
            return False
        self.current_frame_index += 1
        self.viewer.set_pil_image(self._current_frame.image)
        self._current_frame.finish()
        return True

    async def stop(self):
        """
        Stop to play.
        """
        self._clear_frames()

    def description(self) -> str:
        return """
You must complete the presentation on the current page firstly, then call the command to jump to the next page.
"""

    async def context_messages(self):
        message = Message.new(role="user", name="__slide_frame__")

        if not self.is_playing:
            message.with_content(Text(text="Not play any slides yet"))
        else:
            frames = [
                f"Page number:{i + 1} Title={f.title} Outline={f.outline} Finished={f.finished}"
                for i, f in enumerate(self.frames)
            ]
            message.with_content(
                Text(text=f"Current playing slide asset name: {self.loaded.name}"),
                Text(text=f"Frames and Progress:\n{'\n'.join(frames)}"),
                Text(
                    text=f"Current frame name: {self._current_frame.title} content: {self._current_frame.content} image is under"
                ),
                Base64Image.from_pil_image(self._current_frame.image),
            )

        return [message]

    def as_channel(self) -> PyChannel:
        player_chan = PyChannel(name="player")

        player_chan.build.description()(self.description)
        player_chan.build.context_messages(self.context_messages)

        player_chan.build.command()(self.play)
        player_chan.build.command()(self.to_page)
        player_chan.build.command()(self.to_next_page)
        player_chan.build.command()(self.stop)

        return player_chan


class SlideStudio:
    def __init__(self, assets: SlideAssets, container: IoCContainer = None):
        # context
        self._container = Container(parent=container, name="slide_studio")
        self._logger = self._container.get(LoggerItf) or logging.getLogger(__name__)

        # studio parts
        self._assets = assets
        self.player = SlidePlayer(self._assets, self._logger)
        self.creator = SlideCreator(self._assets, self._logger)

    def description(self) -> str:
        return ""

    async def show(self, module="player"):
        self.player.viewer.show()

    async def hide(self, module="player"):
        self.player.viewer.hide()

    async def context_messages(self):
        message = Message.new(role="user", name="__studio__")
        slide_texts = [f"name:{s.name} description:{s.description}" for s in self._assets.refresh()]
        if not slide_texts:
            message.with_content("There has no slides in Slide Studio")
        else:
            message.with_content(f"All assets in Slide Studio:\n{'\n'.join(slide_texts)}")
        return [message]

    def as_channel(self):
        studio_chan = PyChannel(name="slide_studio", blocking=True)

        studio_chan.build.description()(self.description)
        studio_chan.build.context_messages(self.context_messages)

        studio_chan.build.command()(self.show)
        studio_chan.build.command()(self.hide)

        player_chan = self.player.as_channel()
        creator_chan = self.creator.as_channel()

        studio_chan.import_channels(player_chan, creator_chan)

        return studio_chan
