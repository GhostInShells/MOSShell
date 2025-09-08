from pydantic import BaseModel, Field


def test_model_with_enum():
    from enum import Enum
    from pydantic import BaseModel

    class Foo(str, Enum):
        foo = "foo"

    class Bar(BaseModel):
        foo: Foo = Field(default=Foo.foo)

    bar = Bar()
    assert bar.foo == "foo"
    assert isinstance(bar.foo, str)
    bar.foo = Foo.foo
    assert bar.foo == "foo"
    assert isinstance(bar.foo, str)
