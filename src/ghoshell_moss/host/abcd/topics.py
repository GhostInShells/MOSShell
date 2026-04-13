from ghoshell_moss.core.concepts.topic import TopicModel, TopicName
from pydantic import BaseModel, Field


class CTMLTopicModel(TopicModel):
    ctml: str = Field(
        description="ctml to run"
    )

    @classmethod
    def topic_type(cls) -> str:
        return "system/CTML"

    @classmethod
    def default_topic_name(cls) -> TopicName:
        return "system/ctml"
