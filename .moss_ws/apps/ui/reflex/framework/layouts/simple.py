import reflex as rx

from framework.helpers.mixin import NameMixin


class SimpleLayout(rx.ComponentState, NameMixin):
    """
    A Simple Layout
    """
    title: str = ""
    sub_title: str = ""
    tags: list[str] = []
    paragraph: list[str] = []
    markdown: str = ""

    @classmethod
    def name(cls) -> str:
        return "simple"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.vstack(
            rx.skeleton(
                rx.heading(cls.title, size="7"),
                width="500px",
                height="30px",
                loading=cls.title=="",
            ),
            rx.skeleton(
                rx.text(cls.sub_title, color_scheme="gray"),
                width="400px",
                height="25px",
                loading=cls.sub_title=="",
            ),
            rx.skeleton(
                rx.hstack(
                    rx.foreach(cls.tags, rx.badge),
                    spacing="1",
                ),
                width="400px",
                height="25px",
                loading=cls.tags.length()==0,
            ),
            rx.skeleton(
                rx.foreach(
                    cls.paragraph, rx.text
                ),
                width="800px",
                height="100px",
                loading=cls.paragraph.length()==0,
            ),
            rx.skeleton(
                rx.markdown(cls.markdown),
                width="800px",
                height="100px",
                loading=cls.markdown == "",
            ),
            **props,
        )
