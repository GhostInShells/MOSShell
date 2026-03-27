import base64
import io
import mimetypes
import pathlib
from typing import Optional, Any

from PIL import Image
from typing_extensions import Self

from ghoshell_moss.message.contents.abcd import ContentModel

__all__ = ["Base64Image"]


class Base64Image(ContentModel):
    """
    By: Gemini
    基于 Base64 的图像消息体。
    结构完全对齐 Anthropic 的 Base64ImageSourceParam:
    {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "..."
    }
    """

    @classmethod
    def content_type(cls) -> str:
        return 'image'

    @classmethod
    def from_binary(cls, media_type: str, data: bytes) -> Self:
        """从二进制数据直接创建"""
        b64_data = base64.b64encode(data).decode("utf-8")
        source = {
            "type": "base64",
            "media_type": media_type,
            "data": b64_data
        }
        return cls(source=source)

    @classmethod
    def from_pil_image(cls, image: Image.Image, format: Optional[str] = None) -> Self:
        """
        从 PIL 对象转换。
        在机器人实时视觉流（如 G1 的摄像头快照）中这是最高频的入口。
        """
        img_format = format or image.format or "PNG"
        # 统一下 media_type 的表达
        ext = img_format.lower()
        if ext == "jpg": ext = "jpeg"
        media_type = f"image/{ext}"

        buffered = io.BytesIO()
        image.save(buffered, format=img_format)
        return cls.from_binary(media_type, buffered.getvalue())

    @classmethod
    def from_file(cls, file_path: str | pathlib.Path) -> Self:
        """从本地文件读取"""
        path = pathlib.Path(file_path)
        media_type, _ = mimetypes.guess_type(path)
        if not media_type:
            # 默认兜底
            media_type = f"image/{path.suffix.lstrip('.')}" or "image/png"

        with open(path, "rb") as f:
            return cls.from_binary(media_type, f.read())

    def to_pil_image(self) -> Image.Image:
        """还原回 PIL 对象，方便本地做图像处理或在 TUI/UI 中展示"""
        if not self.source or "data" not in self.source:
            raise ValueError("Invalid image source")

        img_data = base64.b64decode(self.source["data"])
        return Image.open(io.BytesIO(img_data))

    @property
    def data_url(self) -> str:
        """生成可以直接在 HTML 或一些交互式终端里渲染的 Data URL"""
        if not self.source:
            return ""
        m_type = self.source.get("media_type", "image/png")
        data = self.source.get("data", "")
        return f"data:{m_type};base64,{data}"
