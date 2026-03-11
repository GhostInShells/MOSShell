from .text import Text, TextDelta
from .functions import FunctionOutput, FunctionCall, FunctionCallDelta
from .images import Base64Image, ImageUrl

ContentModels = [
    Text,
    FunctionOutput,
    FunctionCall,
    Base64Image,
    ImageUrl,
]
"""
可以用来解决粘包逻辑. 
"""
