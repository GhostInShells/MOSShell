import abc


class NameMixin(abc.ABC):

    @abc.abstractmethod
    def name(self) -> str:
        pass