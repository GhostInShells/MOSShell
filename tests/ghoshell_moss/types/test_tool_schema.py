"""simplify_json_schema — the schema a prompt can carry, not the one a validator wants.

Each test proves one promise: references are gone, aliases survive, constraints survive,
recursion fails loudly instead of emitting a dangling reference.
"""

from __future__ import annotations

import json
from enum import Enum

import pytest
from pydantic import BaseModel, Field

from ghoshell_moss.core.concepts.tools import model_json_schema_simplified


class Color(str, Enum):
    red = "red"
    blue = "blue"


class Point(BaseModel):
    x: int
    y: int = 0


class Shape(BaseModel):
    """a shape"""

    color: Color
    points: list[Point]
    label: str = Field(default="", alias="name")
    note: str | None = None


def test_nested_models_and_enums_are_inlined():
    schema = model_json_schema_simplified(Shape)

    assert "$defs" not in schema
    assert "$ref" not in json.dumps(schema)
    assert "title" not in json.dumps(schema)

    props = schema["properties"]
    assert props["color"]["enum"] == ["red", "blue"]
    assert props["points"]["items"]["properties"]["x"]["type"] == "integer"
    assert props["points"]["items"]["properties"]["y"]["default"] == 0


def test_alias_is_the_wire_name():
    props = model_json_schema_simplified(Shape)["properties"]

    assert "name" in props
    assert "label" not in props


def test_constraints_survive():
    class Patterned(BaseModel):
        code: str = Field(pattern=r"^[a-z]+$")

    schema = model_json_schema_simplified(Patterned)

    assert schema["properties"]["code"]["pattern"] == r"^[a-z]+$"


class Node(BaseModel):
    child: "Node | None" = None


Node.model_rebuild()


def test_recursive_schema_raises():
    with pytest.raises(ValueError, match="recursive"):
        model_json_schema_simplified(Node)
