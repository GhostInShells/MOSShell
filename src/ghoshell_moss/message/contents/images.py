import base64
import io
import mimetypes
import pathlib
from typing import Optional, TypedDict

from PIL import Image
from typing_extensions import Self
from ghoshell_moss.message.contents.abcd import ContentModel

__all__ = ["Base64Image"]


class Base64ImageSourceParam(TypedDict, total=False):
    """
    Anthropic Base64ImageSourceParam 协议的本地拷贝.
    不 import anthropic — 那会拖进整个 SDK (启动 +0.69s). 运行时行为与 dict 等价.
    """
    type: str
    media_type: str
    data: str


# 图片字节头 → media_type. 扩展名会撒谎 (截图重命名 / 爬取的资源), 而下游严格校验
# 声明类型与字节是否一致 (dsh attachment admission 直接拒), 所以字节优先.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_media_type(data: bytes) -> Optional[str]:
    """从字节头识别图片格式; 不可识别返回 None (调用方退回扩展名猜测)."""
    for magic, media_type in _IMAGE_MAGIC:
        if data.startswith(magic):
            return media_type
    # WEBP: RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


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
    source: Base64ImageSourceParam | dict

    @classmethod
    def content_type(cls) -> str:
        return 'image'

    @classmethod
    def from_base64(cls, media_type: str, data: str) -> Self:
        source = dict(
            type="base64",
            media_type=media_type,
            data=data
        )
        return cls(source=source)

    @classmethod
    def from_binary(cls, media_type: str, data: bytes) -> Self:
        """从二进制数据直接创建"""
        b64_data = base64.b64encode(data).decode("utf-8")
        source = Base64ImageSourceParam(
            type="base64",
            media_type=media_type,
            data=b64_data
        )
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
        """从本地文件读取. media_type 取自字节头, 扩展名仅在字节不可识别时兜底."""
        path = pathlib.Path(file_path)
        data = path.read_bytes()
        media_type = _sniff_media_type(data)
        if media_type is None:
            media_type, _ = mimetypes.guess_type(path)
        if not media_type:
            # 默认兜底
            media_type = f"image/{path.suffix.lstrip('.')}" or "image/png"

        return cls.from_binary(media_type, data)

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
