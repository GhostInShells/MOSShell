import reflex as rx
from PIL import Image

from framework.helpers.mixin import NameMixin


class CourseLayout(rx.ComponentState, NameMixin):
    """课程演示布局：左图右文，全屏适配演示场景。
    """

    title: str = ""
    sub_title: str = ""
    image: list[Image.Image] = []
    main_text: str = ""
    annotations: list[str] = []
    appreciation: str = ""

    @classmethod
    def name(cls) -> str:
        return "course"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            rx.hstack(
                # ── 左侧：大幅配图（50% 宽度）──
                rx.box(
                    rx.skeleton(
                        rx.image(
                            cls.image[0],
                            width="100%",
                            height="auto",
                            max_height="80vh",
                            border_radius="12px",
                        ),
                        width="100%",
                        height="60vh",
                        loading=cls.image.length() == 0,
                    ),
                    width="50%",
                    padding="32px",
                ),
                # ── 右侧：文字内容（50% 宽度）──
                rx.box(
                    rx.vstack(
                        # 标题
                        rx.skeleton(
                            rx.heading(cls.title, size="8"),
                            width="500px",
                            height="40px",
                            loading=cls.title == "",
                        ),
                        # 副标题
                        rx.skeleton(
                            rx.text(
                                cls.sub_title, size="3", color_scheme="gray"
                            ),
                            width="300px",
                            height="24px",
                            loading=cls.sub_title == "",
                        ),
                        rx.divider(),
                        # 正文（讲课场景每章仅一个 talking point，内容极简）
                        rx.skeleton(
                            rx.markdown(cls.main_text),
                            width="100%",
                            height="60px",
                            loading=cls.main_text == "",
                        ),
                        # 注释
                        rx.skeleton(
                            rx.box(
                                rx.heading("注释", size="5"),
                                rx.foreach(
                                    cls.annotations,
                                    lambda text, idx: rx.text(
                                        rx.text.span(
                                            f"[{idx + 1}] ",
                                            color_scheme="blue",
                                        ),
                                        text,
                                        size="2",
                                    ),
                                ),
                            ),
                            width="100%",
                            height="60px",
                            loading=cls.annotations.length() == 0,
                        ),
                        # 赏析
                        rx.skeleton(
                            rx.markdown(cls.appreciation),
                            width="100%",
                            height="60px",
                            loading=cls.appreciation == "",
                        ),
                        align_items="start",
                        spacing="4",
                    ),
                    width="50%",
                    padding="32px 48px",
                ),
                align_items="center",
                spacing="0",
                width="100%",
                height="100%",
            ),
            width="100%",
            height="100vh",
            overflow="hidden",
            background="#fafbfc",
            **props,
        )
