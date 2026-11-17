"""load_debug_model — how the debug surface turns a reference into a model class.

The promise being tested is the sandbox's: a file may define a pydantic model, and it may not
reach the machine through an import it was not granted.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ghoshell_moss.ghosts.dolores.mindset.beside.debug import load_debug_model

_SOURCE = """
from pydantic import BaseModel, Field

class Answer(BaseModel):
    ok: bool
    note: str = Field(default="", description="why")
"""


def test_module_ref_resolves_an_importable_class():
    cls, box = load_debug_model("pydantic:BaseModel", source=False)

    assert cls is BaseModel
    assert box is None


def test_source_ref_compiles_a_model_in_the_sandbox(tmp_path):
    path = tmp_path / "m.py"
    path.write_text(_SOURCE)

    cls, box = load_debug_model(f"{path}:Answer", source=True)
    try:
        assert issubclass(cls, BaseModel)
        assert cls(ok=True).note == ""
    finally:
        box.close()


def test_source_ref_refuses_an_ungranted_import(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("import os\nfrom pydantic import BaseModel\n")

    with pytest.raises(ValueError, match="refused 'os'"):
        load_debug_model(f"{path}:X", source=True)


def test_source_ref_must_resolve_to_a_model(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("class Nope:\n    pass\n")

    with pytest.raises(TypeError, match="BaseModel"):
        load_debug_model(f"{path}:Nope", source=True)
