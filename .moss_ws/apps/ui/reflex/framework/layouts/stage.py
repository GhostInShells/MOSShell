import reflex as rx
from PIL import Image
from pydantic import BaseModel, Field

from framework.helpers.mixin import NameMixin


class StatusBar(BaseModel):
    """意识维度"""
    label: str = Field(default="", description="维度名称")
    value: int = Field(default=0, description="当前值 0-100")
    color: str = Field(default="#6366f1", description="进度条颜色")


class ChannelCard(BaseModel):
    """Channel 能力卡片"""
    name: str = Field(default="", description="Channel 名称")
    description: str = Field(default="", description="一行描述")
    status: str = Field(default="idle", description="idle | active | error")


_STATUS_COLORS = {
    "idle": "#e5e7eb",
    "active": "#10b981",
    "error": "#ef4444",
}


def _status_bar(bar: StatusBar) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(bar.label, size="1", color="#9ca3af", weight="medium"),
            rx.text(f"{bar.value}%", size="1", color="#6b7280"),
            width="100%",
            justify="between",
        ),
        rx.box(
            rx.box(
                width=f"{bar.value}%",
                height="100%",
                background=bar.color,
                border_radius="2px",
                transition="width 0.6s ease",
            ),
            width="100%",
            height="4px",
            background="#f3f4f6",
            border_radius="2px",
            overflow="hidden",
        ),
        spacing="1",
        flex="1",
        min_width="0",
    )


def _channel_card(card: ChannelCard) -> rx.Component:
    status_color = _STATUS_COLORS.get(card.status, "#e5e7eb")
    return rx.box(
        rx.hstack(
            rx.box(
                width="8px",
                height="8px",
                border_radius="50%",
                background=status_color,
                flex_shrink="0",
            ),
            rx.vstack(
                rx.text(card.name, size="2", weight="bold", color="#1f2937"),
                rx.text(card.description, size="1", color="#9ca3af"),
                spacing="0",
                flex="1",
                min_width="0",
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        padding="10px 14px",
        background="white",
        border_radius="8px",
        border="1px solid #f3f4f6",
        flex="1 1 200px",
        max_width="320px",
    )


class StageLayout(rx.ComponentState, NameMixin):
    """Ghost 的舞台 —— 意识状态、流式意识流、视觉素材、Channel 能力清单。"""

    status_bars: list[StatusBar] = []
    title: str = ""
    subtitle: str = ""
    body: str = ""  # markdown
    images: list[Image.Image] = []
    cards: list[ChannelCard] = []

    @classmethod
    def name(cls) -> str:
        return "stage"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.vstack(
            # ── 顶部状态条 ──
            rx.skeleton(
                rx.hstack(
                    rx.foreach(cls.status_bars, _status_bar),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
                height="36px",
                loading=cls.status_bars.length() == 0,
            ),
            # ── 主体：左意识流 + 右视觉素材 ──
            rx.hstack(
                rx.vstack(
                    rx.skeleton(
                        rx.heading(cls.title, size="8", weight="bold"),
                        width="100%",
                        height="48px",
                        loading=cls.title == "",
                    ),
                    rx.skeleton(
                        rx.text(cls.subtitle, color_scheme="gray", size="4"),
                        width="100%",
                        height="24px",
                        loading=cls.subtitle == "",
                    ),
                    rx.skeleton(
                        rx.markdown(cls.body),
                        width="100%",
                        height="200px",
                        loading=cls.body == "",
                    ),
                    flex="4",
                    min_width="0",
                    spacing="5",
                ),
                rx.box(
                    rx.skeleton(
                        rx.vstack(
                            rx.foreach(
                                cls.images,
                                lambda img: rx.image(
                                    img,
                                    width="100%",
                                    height="auto",
                                    border_radius="12px",
                                ),
                            ),
                            spacing="4",
                        ),
                        width="100%",
                        height="500px",
                        loading=cls.images.length() == 0,
                    ),
                    flex="5",
                    min_width="0",
                ),
                width="100%",
                align="start",
                spacing="8",
            ),
            # ── Channel 能力卡片 ──
            rx.vstack(
                rx.text("Channels", size="3", weight="bold", color="#6b7280"),
                rx.cond(
                    cls.cards.length() == 0,
                    rx.vstack(
                        rx.skeleton(
                            rx.box(width="100%", height="56px"),
                            width="100%",
                            height="56px",
                            loading=True,
                        ),
                        rx.skeleton(
                            rx.box(width="100%", height="56px"),
                            width="100%",
                            height="56px",
                            loading=True,
                        ),
                        rx.skeleton(
                            rx.box(width="100%", height="56px"),
                            width="100%",
                            height="56px",
                            loading=True,
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    rx.flex(
                        rx.foreach(cls.cards, _channel_card),
                        spacing="3",
                        flex_wrap="wrap",
                        width="100%",
                    ),
                ),
                width="100%",
                spacing="3",
            ),
            width="100%",
            min_height="100vh",
            padding="32px 64px 48px",
            spacing="6",
            **props,
        )
