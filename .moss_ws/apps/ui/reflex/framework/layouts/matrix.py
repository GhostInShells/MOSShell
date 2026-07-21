import reflex as rx
from pydantic import BaseModel, Field

from framework.helpers.mixin import NameMixin


class CellBar(BaseModel):
    """Matrix Cell 连接状态条"""
    label: str = Field(default="", description="Cell 名称")
    value: int = Field(default=0, description="连接强度 0-100")
    color: str = Field(default="#10b981", description="进度条颜色")


def _cell_bar(bar: CellBar) -> rx.Component:
    """大号 Cell 连接状态条 —— 黑色背景 + 粗进度条 + 0.8s 过渡动画"""
    return rx.vstack(
        rx.hstack(
            rx.text(bar.label, color="white", size="5", weight="bold"),
            rx.text(f"{bar.value}%", color="#4b5563", size="4"),
            width="100%",
            justify="between",
        ),
        rx.box(
            rx.box(
                width=f"{bar.value}%",
                height="100%",
                background=bar.color,
                border_radius="6px",
                transition="width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
            ),
            width="100%",
            height="28px",
            background="#1a1a1a",
            border_radius="6px",
            overflow="hidden",
        ),
        spacing="2",
        width="100%",
    )


class MatrixLayout(rx.ComponentState, NameMixin):
    """Matrix 总线拓扑 —— 黑底大号状态条，逐个展示 Cell 接入 Matrix 的过程。

    只保留 title 和 status_bars 两个字段，去掉 StageLayout 的正文、图片、
    卡片等内容。status_bars 超大、居中、黑底，适合演示"节点接入"的脉冲效果。
    """

    status_bars: list[CellBar] = []
    title: str = ""

    @classmethod
    def name(cls) -> str:
        return "matrix"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.center(
            rx.vstack(
                rx.heading(
                    cls.title,
                    size="8",
                    color="white",
                    text_align="center",
                ),
                rx.vstack(
                    rx.foreach(cls.status_bars, _cell_bar),
                    spacing="6",
                    width="min(600px, 80vw)",
                ),
                spacing="9",
            ),
            width="100vw",
            height="100vh",
            background="#0a0a0a",
            **props,
        )
