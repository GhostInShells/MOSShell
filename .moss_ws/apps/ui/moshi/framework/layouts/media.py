import reflex as rx
from PIL import Image

from framework.helpers.mixin import NameMixin


class MediaLayout(rx.ComponentState, NameMixin):
    """文字 + 媒体混排布局。

    文字部分（title / sub_title / tags / paragraph / markdown）复用 simple 骨架渲染模式，
    媒体部分（media: list[MediaItem]）按槽位展示，支持 pending / ready / failed 三种状态。
    """

    title: str = ""
    sub_title: str = ""
    image: list[Image.Image] = []

    @classmethod
    def name(cls) -> str:
        return "media"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.vstack(
            # ── 文字区域（复用 simple 骨架模式）──
            rx.skeleton(
                rx.heading(cls.title, size="7"),
                width="500px",
                height="30px",
                loading=cls.title == "",
            ),
            rx.skeleton(
                rx.text(cls.sub_title, color_scheme="gray"),
                width="400px",
                height="25px",
                loading=cls.sub_title == "",
            ),
            # ── 媒体区域 ──
            rx.skeleton(
                rx.hstack(
                    rx.foreach(
                        cls.image, lambda i: rx.image(i, width="400px", height="auto"),
                    ),
                ),
                width="200px",
                height="200px",
                loading=cls.image.length() == 0
            ),
            **props,
        )
