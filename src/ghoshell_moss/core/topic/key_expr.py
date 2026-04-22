from ghoshell_moss.core.concepts.topic import TopicNamePattern
import re

__all__ = ["MOSSTopicExpr"]

topic_name_matcher = re.compile(TopicNamePattern)


class MOSSTopicExpr:

    def __init__(self, *, session_scope: str, address: str):
        self.address = address
        self.session_scope = session_scope
        self.topic_prefix = "MOSS/{session_scope}/topics".format(session_scope=session_scope)

    def topic_key_expr(self, topic_name: str) -> str:
        matched = topic_name_matcher.fullmatch(topic_name)
        if matched is None:
            raise ValueError(f"Invalid topic name: {topic_name}")
        return "/".join([self.topic_prefix, topic_name.strip('/')])
