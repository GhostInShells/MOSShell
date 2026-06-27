import reflex as rx

from framework.events import VideoLocator
from framework.helpers.mixin import NameMixin


class HeroLayout(rx.ComponentState, NameMixin):
    """沉浸式全屏视频布局。无控制条、autoplay、纯黑背景。"""

    videos: list[VideoLocator] = []

    @classmethod
    def name(cls) -> str:
        return "hero"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            rx.foreach(
                cls.videos,
                lambda v: rx.box(
                    rx.video(
                        src=v,
                        playing=True,
                        controls=False,
                        muted=True,
                        loop=True,
                        width="100%",
                        height="100%",
                    ),
                    width="100%",
                    height="100%",
                ),
            ),
            width="100vw",
            height="100vh",
            overflow="hidden",
            background="#000000",
            **props,
        )
