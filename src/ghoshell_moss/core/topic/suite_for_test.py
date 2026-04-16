from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.topic import QueueBasedTopicService

__all__ = ["TopicServiceSuite", "QueueTopicServiceSuite"]


class TopicServiceSuite(ABC):
    @abstractmethod
    def name(self) -> str:
        """Suite 的名称，用于 pytest 报告显示"""
        pass

    @abstractmethod
    def create_service(self, sender: str) -> TopicService:
        """创建一个全新的、干净的 Service 实例"""
        pass

    def cleanup(self) -> None:
        pass


# --- 默认实现：QueueBased ---

class QueueTopicServiceSuite(TopicServiceSuite):
    def name(self) -> str:
        return "queue_based"

    def create_service(self, sender: str) -> TopicService:
        return QueueBasedTopicService(sender=sender)
