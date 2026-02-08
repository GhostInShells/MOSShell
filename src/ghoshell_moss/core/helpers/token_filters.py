from collections.abc import Iterable


class SpecialTokenMatcher:
    """
    一个简单的字符串过滤器, 用来加工特殊字符, 将它转换成指定字符.
    这样未来可以让模型自己增删特定的功能.
    """

    def __init__(self, matchers: dict[str, str]):
        self.matchers = matchers
        self._has_matchers = len(matchers) > 0
        self._matching = ""
        self._legal_tokens = set()
        for matcher in matchers:
            buffer = ""
            for c in matcher:
                buffer += c
                self._legal_tokens.add(buffer)

    def clear(self) -> str:
        if len(self._matching) > 0:
            self._matching = ""
            return self._matching
        return ""

    def buffer(self, delta: str) -> str:
        if not self._has_matchers:
            return delta
        outputs = ""
        matching = self._matching
        for c in delta:
            matching += c
            if matching not in self._legal_tokens:
                outputs += matching
                matching = ""
            elif matching in self.matchers:
                outputs += self.matchers[matching]
                matching = ""
        self._matching = matching
        return outputs

    def parse(self, texts: Iterable[str]) -> Iterable[str]:
        for text in texts:
            yield self.buffer(text)
        cleared = self.clear()
        if cleared:
            yield cleared
