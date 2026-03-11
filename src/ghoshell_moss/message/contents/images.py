import base64
import pathlib
from io import BytesIO
from typing import Optional

from PIL import Image
from pydantic import Field
from typing_extensions import Self

from ghoshell_moss.message.abcd import ContentModel, Delta, DeltaModel

__all__ = ["Base64Image", "ImageUrl"]


class Base64Image(ContentModel):
    """
    Base64 encoded image with metadata

    用法:
        msg = Message.new().with_content(Base64Image.from_pil_image(image))
    """

    CONTENT_TYPE = "base64_image"
    image_type: str = Field(
        description="Image format (e.g., 'png', 'jpeg', 'jpg', 'gif')",
    )
    encoded: str = Field(
        description="Base64 encoded image data",
        examples=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="],
    )

    @classmethod
    def from_binary(cls, image_type: str, binary: bytes) -> Self:
        """Create Base64Image from binary data"""
        encoded = base64.b64encode(binary).decode("utf-8")
        return cls(image_type=image_type, encoded=encoded)

    @classmethod
    def from_pil_image(cls, image: Image.Image, format: Optional[str] = None) -> Self:
        """
        Create Base64Image from PIL Image

        Args:
            image: PIL Image object
            format: Image format (e.g., 'PNG', 'JPEG'). If None, uses image.format or defaults to 'PNG'
        """
        if format is None:
            format = image.format or "PNG"

        # Convert format to lowercase for consistency
        image_type = format.lower()

        # Save image to bytes buffer
        buffer = BytesIO()
        image.save(buffer, format=format)
        binary_data = buffer.getvalue()

        return cls.from_binary(image_type, binary_data)

    @classmethod
    def from_file(cls, file_path: str | pathlib.Path) -> Self:
        """
        Create Base64Image from image file

        Args:
            file_path: Path to image file
        """
        if isinstance(file_path, pathlib.Path):
            file_path = str(file_path.absolute())

        # Open image with PIL to get format
        image = Image.open(file_path)
        format = image.format or "PNG"

        # Read binary data
        binary_data = pathlib.Path(file_path).read_bytes()

        return cls.from_binary(format.lower(), binary_data)

    def to_pil_image(self) -> Image.Image:
        """Convert Base64Image back to PIL Image"""
        # Decode base64
        binary_data = base64.b64decode(self.encoded)

        # Create PIL Image from bytes
        image = Image.open(BytesIO(binary_data))
        return image

    @property
    def mime_type(self) -> str:
        """Get MIME type for the image"""
        mime_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp",
            "tiff": "image/tiff",
        }
        return mime_map.get(self.image_type.lower(), "application/octet-stream")

    @property
    def data_url(self) -> str:
        """Get data URL for embedding in HTML or other contexts"""
        return f"data:{self.mime_type};base64,{self.encoded}"

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        return None

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        return False


class ImageUrl(ContentModel):
    """
    用 url 提供的图片类型.
    """

    CONTENT_TYPE = "image_url"
    url: str = Field(
        description="Image URL of the message",
    )

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        return None

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        return False
