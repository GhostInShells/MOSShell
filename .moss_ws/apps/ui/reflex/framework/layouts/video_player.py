"""Video Player 布局 — 左右分栏：左侧大字文本，右侧图片/视频（自动播放）。

暗色沉浸背景。文字区带凝聚浮现动画，媒体区自动播放、静音、循环。
video_player 从纯视频播放升级为媒体+文本展示布局。
"""

import reflex as rx
from PIL import Image

from framework.events import VideoLocator
from framework.helpers.mixin import NameMixin

_VIDEO_PLAYER_CSS = """
@keyframes vp-text-coalesce {
  0% {
    opacity: 0;
    filter: blur(12px);
    transform: translateY(28px);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0);
  }
}

@keyframes vp-media-coalesce {
  0% {
    opacity: 0;
    filter: blur(6px);
    transform: scale(1.04);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: scale(1);
  }
}

.vp-title {
  animation: vp-text-coalesce 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

.vp-subtitle {
  animation: vp-text-coalesce 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) 0.08s both;
}

.vp-body {
  animation: vp-text-coalesce 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) 0.16s both;
}

.vp-media {
  animation: vp-media-coalesce 0.9s cubic-bezier(0.22, 0.61, 0.36, 1) 0.2s both;
}
"""


class VideoPlayerLayout(rx.ComponentState, NameMixin):
    """左右分栏布局：左侧大字文本，右侧图片/视频（自动播放）。

    左侧：title / sub_title / body 三级文本，带交错凝聚动画。
    右侧：优先展示视频（自动播放、静音、循环），无视频时展示图片。
    空态时有 skeleton 过渡。
    """

    title: str = ""
    sub_title: str = ""
    body: str = ""
    image: list[Image.Image] = []
    videos: list[VideoLocator] = []

    @classmethod
    def name(cls) -> str:
        return "video_player"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            rx.hstack(
                # ── 左侧：文字区域 ──
                rx.center(
                    rx.vstack(
                        rx.skeleton(
                            rx.heading(
                                cls.title,
                                size="9",
                                weight="bold",
                                color="#e4e4f2",
                                letter_spacing="0.04em",
                                line_height="1.15",
                                class_name="vp-title",
                            ),
                            loading=cls.title == "",
                        ),
                        rx.skeleton(
                            rx.text(
                                cls.sub_title,
                                size="5",
                                color="#8888c8",
                                letter_spacing="0.06em",
                                weight="medium",
                                class_name="vp-subtitle",
                            ),
                            loading=cls.sub_title == "",
                        ),
                        rx.skeleton(
                            rx.text(
                                cls.body,
                                size="3",
                                color="#6868a0",
                                line_height="1.85",
                                class_name="vp-body",
                            ),
                            loading=cls.body == "",
                        ),
                        spacing="5",
                        align="start",
                        max_width="520px",
                        padding_x="64px",
                    ),
                    width="50%",
                    height="100vh",
                ),
                # ── 分隔线 ──
                rx.divider(
                    orientation="vertical",
                    height="50vh",
                    border_color="#181840",
                ),
                # ── 右侧：媒体区域 ──
                # 视频优先：与 hero.py 完全一致，video 必须包在 rx.box 里建立尺寸上下文
                # foreach 中间层无显式尺寸，不加 wrapper 100% 解析到 0 → 视频不可见
                rx.box(
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
                            class_name="vp-media",
                        ),
                    ),
                    rx.foreach(
                        cls.image,
                        lambda i: rx.box(
                            rx.image(
                                i,
                                width="100%",
                                height="100%",
                            ),
                            width="100%",
                            height="100%",
                            class_name="vp-media",
                        ),
                    ),
                    rx.cond(
                        (cls.videos.length() == 0) & (cls.image.length() == 0),
                        rx.center(
                            rx.icon("clapperboard", size=48, color="#141430"),
                            width="100%",
                            height="100%",
                        ),
                    ),
                    width="50%",
                    height="100vh",
                    overflow="hidden",
                    background="#020210",
                ),
                width="100%",
                height="100vh",
                spacing="0",
            ),
            rx.html(f"<style>{_VIDEO_PLAYER_CSS}</style>"),
            width="100vw",
            height="100vh",
            overflow="hidden",
            background="#060616",
            **props,
        )
