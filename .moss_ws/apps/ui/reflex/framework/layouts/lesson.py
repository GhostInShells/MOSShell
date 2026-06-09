import reflex as rx
from PIL import Image

from framework.helpers.mixin import NameMixin


class LessonLayout(rx.ComponentState, NameMixin):
    """语文备课布局：课程标题、作者信息、课文插图、正文、注释、赏析。"""

    title: str = ""
    sub_title: str = ""
    image: list[Image.Image] = []
    main_text: str = ""
    annotations: list[str] = []
    appreciation: str = ""

    @classmethod
    def name(cls) -> str:
        return "lesson"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.vstack(
            # ── 课程标题 ──
            rx.skeleton(
                rx.heading(cls.title, size="8"),
                width="500px",
                height="36px",
                loading=cls.title == "",
            ),
            # ── 副标题（作者/朝代）──
            rx.skeleton(
                rx.text(cls.sub_title, color_scheme="gray", size="3"),
                width="300px",
                height="20px",
                loading=cls.sub_title == "",
            ),
            rx.divider(),
            # ── 课文插图 ──
            rx.skeleton(
                rx.hstack(
                    rx.foreach(
                        cls.image,
                        lambda img: rx.image(img, width="400px", height="auto", border_radius="8px"),
                    ),
                ),
                width="400px",
                height="250px",
                loading=cls.image.length() == 0,
            ),
            # ── 正文 ──
            rx.skeleton(
                rx.markdown(cls.main_text),
                width="800px",
                height="150px",
                loading=cls.main_text == "",
            ),
            # ── 注释 ──
            rx.skeleton(
                rx.box(
                    rx.heading("注释", size="5"),
                    rx.foreach(
                        cls.annotations,
                        lambda text, idx: rx.text(
                            rx.text.span(f"[{idx + 1}] ", color_scheme="blue"),
                            text,
                            size="2",
                        ),
                    ),
                ),
                width="800px",
                height="100px",
                loading=cls.annotations.length() == 0,
            ),
            # ── 赏析 ──
            rx.skeleton(
                rx.markdown(cls.appreciation),
                width="800px",
                height="100px",
                loading=cls.appreciation == "",
            ),
            **props,
        )
