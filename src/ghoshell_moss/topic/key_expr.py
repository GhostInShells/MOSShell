from ghoshell_moss.core.concepts.topic import TopicNamePattern
import re

__all__ = ["MOSSTopicExpr"]

topic_name_matcher = re.compile(TopicNamePattern)


class MOSSTopicExpr:

    def __init__(self, *, session_id: str, node_name: str):
        self.node_name = node_name
        self.session_id = session_id
        self.topic_prefix = "MOSS/{session_id}/topics".format(session_id=session_id)

    def topic_key_expr(self, topic_name: str) -> str:
        matched = topic_name_matcher.fullmatch(topic_name)
        if matched is None:
            raise ValueError(f"Invalid topic name: {topic_name}")
        return "/".join([self.topic_prefix, topic_name])
