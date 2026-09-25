"""Vision topic models.

Typed, self-describing topics published/consumed via TopicService at runtime.
Using a typed model (rather than a raw stream string) gives cross-node consumers
a shared schema — no ad-hoc protocol coupling between vision producer and consumer.
"""
from pydantic import Field

from ghoshell_moss.core.concepts.topic import TopicModel

__all__ = ["FaceTopic"]


class FaceTopic(TopicModel):
    """A face detected in a camera frame, coordinates normalized 0..1.

    Published by the camera vision node when watch is on. The topic's sender
    metadata (default = cell address) lets consumers know which vision source
    produced it — a future consumer can open that cell's channel to track it.
    """

    camera: str = Field(default="", description="camera node identity / index")
    x: float = Field(default=0.0, description="bbox left, normalized 0..1")
    y: float = Field(default=0.0, description="bbox top, normalized 0..1")
    w: float = Field(default=0.0, description="bbox width, normalized 0..1")
    h: float = Field(default=0.0, description="bbox height, normalized 0..1")
    cx: float = Field(default=0.0, description="bbox center x, normalized 0..1")
    cy: float = Field(default=0.0, description="bbox center y, normalized 0..1")
    ts: float = Field(default=0.0, description="frame timestamp (unix seconds)")

    @classmethod
    def topic_type(cls) -> str:
        return "vision/face"

    @classmethod
    def default_topic_name(cls) -> str:
        return "vision/face"
