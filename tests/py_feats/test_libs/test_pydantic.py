from pydantic import Field, dataclasses, BaseModel, Discriminator
from typing import TypeAlias, Union, Annotated, Literal


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


def test_pydantic_dataclass():
    @dataclasses.dataclass
    class Foo:
        val: str = "foo"

    class Bar(BaseModel):
        foo: Foo = Foo()

    bar = Bar()
    assert bar.foo.val == "foo"
    assert 'foo' in bar.model_dump()
    assert "foo" in bar.model_dump_json()
    assert len(bar.model_json_schema()) > 0
    assert dataclasses.is_pydantic_dataclass(Foo)
    new_bar = Bar.model_construct(**bar.model_dump())
    # cannot reconstruct the origin type
    assert isinstance(new_bar.foo, dict)


class Foo(BaseModel):
    kind: Literal['Foo'] = "Foo"
    foo: str = "foo"


class Bar(BaseModel):
    kind: Literal['Bar'] = "Bar"
    bar: str = "bar"


Item: TypeAlias = Annotated[
    Foo | Bar,
    Discriminator('kind')
]


def test_pydantic_multi_sub_type():
    import json

    class Baz(BaseModel):
        items: list[Item] = Field(
            default_factory=list,
        )

    baz = Baz(items=[Foo(), Bar()])
    assert baz.items[0].foo == "foo"
    assert baz.items[1].bar == "bar"

    js = baz.model_dump_json()
    new_baz = Baz.model_construct(**json.loads(js))
    # dataclass cannot be wrapped from new data
    assert isinstance(new_baz.items[0], dict)
    assert isinstance(new_baz.items[1], dict)
