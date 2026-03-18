from .text import Text
from .images import Base64Image, ImageUrl

"""
deprecated:
自定义的 content 不再迭代. 
"""

ContentModels = [
    Text,
    Base64Image,
    ImageUrl,
]
ContentModelsDict = {
    m.content_type(): m for m in ContentModels
}

