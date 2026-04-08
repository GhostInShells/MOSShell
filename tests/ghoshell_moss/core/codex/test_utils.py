from typing import NamedTuple, List
from typing_extensions import is_protocol, is_typeddict
from ghoshell_moss.core.codex._utils import (
    get_class_def_from_source, replace_class_def_name, strip_source_indent, count_source_indent,
    parse_doc_string,
    escape_string_quotes,
    is_typing,
)


def test_replace_class_def_name():
    Case = NamedTuple('Case', [('origin', str), ('name', str), ('expect', str)])

    cases: List[Case] = [
        Case(
            """
class Foo:
    class Bar:
        bar = 1
""",
            'Boo',
            """
class Boo:
    class Bar:
        bar = 1
"""
        ),
        Case(
            """
class Foo(ABC):
    class Bar:
        bar = 1
""",
            'Boo',
            """
class Boo(ABC):
    class Bar:
        bar = 1
"""
        )
    ]

    for c in cases:
        assert replace_class_def_name(c.origin.strip(), c.name) == c.expect.strip()


def test_strip_source_indent():
    Case = NamedTuple('Case', [('source', str), ('indent', int), ('expect', str)])
    cases: List[Case] = [
        Case(
            """
class Foo(ABC):
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
            0,
            """
class Foo(ABC):
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
        ),
        Case(
            """
        class Foo(ABC):
            \"""
            test
            \"""

            foo: int = 1

            def bar(self) -> str:
                return "bar"
""",
            8,
            """
class Foo(ABC):
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
        ),
        Case(
            """
    # if
    class Foo(ABC):
        \"""
        test
        \"""

        foo: int = 1

        def bar(self) -> str:
            return "bar"
""",
            4,
            """
# if
class Foo(ABC):
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
        ),
    ]
    for c in cases:
        assert count_source_indent(c.source.strip("\n")) == c.indent
        assert strip_source_indent(c.source.strip("\n")) == c.expect.strip()


def test_get_class_def_from_source():
    Case = NamedTuple('Case', [('source', str), ('expect', str)])
    cases: List[Case] = [
        Case(
            """
class Foo(ABC):
    \"""
    test
    \"""
    
    foo: int = 1
    
    def bar(self) -> str:
        return "bar"
""",
            """
class Foo(ABC):
"""),
        Case(
            """
class Foo(A, B, C, metaclass=E):
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
            """
class Foo(A, B, C, metaclass=E):
"""),
        Case(
            """
class Foo: # some
    \"""
    test
    \"""

    foo: int = 1

    def bar(self) -> str:
        return "bar"
""",
            """
class Foo: # some
"""),

    ]

    for c in cases:
        assert get_class_def_from_source(c.source) == c.expect.strip()


def test_escape_string_quotes():
    r = escape_string_quotes('hello """ \\""" world')
    assert r == 'hello \\""" \\""" world'


def test_parse_doc_string():
    Case = NamedTuple('Case', [('doc', str), ('expect', str), ('inline', bool)])
    cases: List[Case] = [
        Case(
            'hello world',
            '"""hello world"""',
            True,
        ),
        Case(
            'hello world',
            '"""\nhello world\n"""',
            False,
        ),
        Case(
            'hello """ world',
            '"""\nhello \\""" world\n"""',
            False,
        ),
    ]

    for c in cases:
        assert parse_doc_string(c.doc, inline=c.inline) == c.expect.strip()


def test_is_class_and_is_subclass():
    import inspect
    a = dict[str, int]
    assert not inspect.isclass(a)
    assert not is_protocol(a)
    assert is_typing(a)
    assert not inspect.isabstract(a)
    e = False
    b = None
    try:
        b = issubclass(a, dict)
    except TypeError:
        e = True
    assert b is False
    assert e is False


def test_is_typing():
    import inspect
    a = dict[str, int]
    assert not inspect.isclass(a)
    assert not inspect.isbuiltin(a)
    assert is_typing(a)
    assert str(a) == "dict[str, int]"
    assert a.__module__ == "builtins"
