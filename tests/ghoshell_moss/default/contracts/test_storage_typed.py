"""Tests for Storage typed model methods and AsyncStorageProxy."""

from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ghoshell_moss.contracts.workspace import LocalStorage, LocalWorkspace


# -- test models ----------------------------------------------------------

class Item(BaseModel):
    name: str = "default"
    count: int = 0


class TaggedItem(BaseModel):
    name: str
    tags: list[str] = Field(default_factory=list)


# -- append ---------------------------------------------------------------


def test_append_basic(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.append("log.txt", b"line1\n")
    storage.append("log.txt", b"line2\n")
    content = storage.get("log.txt")
    assert content == b"line1\nline2\n"


def test_append_creates_parents(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.append("deep/nested/log.txt", b"data\n")
    assert storage.exists("deep/nested/log.txt")
    assert storage.get("deep/nested/log.txt") == b"data\n"


def _append_to_path(path: str, data: bytes) -> None:
    with open(path, "ab") as f:
        f.write(data)


def test_append_cross_process(tmp_path: Path):
    """Two processes appending to the same file — both writes must survive."""
    file_path = str(tmp_path / "shared.log")
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    p1 = multiprocessing.Process(target=_append_to_path, args=(file_path, b"p1\n"))
    p2 = multiprocessing.Process(target=_append_to_path, args=(file_path, b"p2\n"))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    lines = Path(file_path).read_bytes().splitlines()
    assert b"p1" in lines
    assert b"p2" in lines


# -- frontmatter: read_model / write_model --------------------------------


def test_write_and_read_model(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    obj = Item(name="test", count=42)
    storage.write_model("config", obj, content="## Usage notes\n")

    result = storage.read_model("config", Item)
    assert result is not None
    model, content = result
    assert model.name == "test"
    assert model.count == 42
    assert "## Usage notes" in content


def test_read_model_missing(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    assert storage.read_model("nonexistent", Item) is None


def test_write_model_content_preserved(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_model("cfg", Item(name="a"), content="Some markdown body")
    _, content = storage.read_model("cfg", Item)
    assert "Some markdown body" in content


def test_write_model_empty_content(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_model("cfg", Item(name="a"))
    result = storage.read_model("cfg", Item)
    assert result is not None
    _, content = result
    assert content == "" or content.strip() == ""


def test_frontmatter_file_is_markdown(tmp_path: Path):
    """Verify the serialized format is valid frontmatter."""
    storage = LocalStorage(tmp_path)
    storage.write_model("cfg", Item(name="x", count=5), content="body")
    raw = storage.get("cfg.md").decode("utf-8")
    assert "---" in raw          # YAML delimiter
    assert "name: x" in raw
    assert "count: 5" in raw
    assert "body" in raw


# -- JSONL: read_models / append_model ------------------------------------


def test_append_and_read_models(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.append_model("events", Item(name="a", count=1))
    storage.append_model("events", Item(name="b", count=2))

    items = list(storage.read_models("events", Item))
    assert len(items) == 2
    assert items[0].name == "a"
    assert items[1].name == "b"
    assert items[0].count == 1
    assert items[1].count == 2


def test_read_models_empty(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    items = list(storage.read_models("nonexistent", Item))
    assert items == []


def test_append_model_creates_file(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.append_model("events", Item(name="first"))
    assert storage.exists("events.jsonl")
    items = list(storage.read_models("events", Item))
    assert len(items) == 1


def test_jsonl_skips_empty_lines(tmp_path: Path):
    """Manually write JSONL with blank lines — reader should skip them."""
    storage = LocalStorage(tmp_path)
    storage.put("events.jsonl", b'{"name":"a","count":1}\n\n{"name":"b","count":2}\n\n')
    items = list(storage.read_models("events", Item))
    assert len(items) == 2


def test_jsonl_file_format(tmp_path: Path):
    """Verify the serialized format is pure JSONL."""
    storage = LocalStorage(tmp_path)
    storage.append_model("events", Item(name="x", count=7))
    raw = storage.get("events.jsonl").decode("utf-8")
    assert '"name":"x"' in raw
    assert '"count":7' in raw
    assert raw.endswith("\n")


# -- YAML: read_yaml / write_yaml -----------------------------------------


def test_write_and_read_yaml(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    obj = TaggedItem(name="service", tags=["prod", "critical"])
    storage.write_yaml("routing", obj)

    result = storage.read_yaml("routing", TaggedItem)
    assert result is not None
    assert result.name == "service"
    assert result.tags == ["prod", "critical"]


def test_read_yaml_missing(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    assert storage.read_yaml("nonexistent", Item) is None


def test_yaml_file_has_import_comment(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_yaml("cfg", Item(name="x"))
    raw = storage.get("cfg.yml").decode("utf-8")
    assert raw.startswith("# dump from `")


# -- name suffix resolution -----------------------------------------------


def test_name_auto_suffix_md(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_model("config", Item())
    assert storage.exists("config.md")


def test_name_auto_suffix_jsonl(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.append_model("log", Item())
    assert storage.exists("log.jsonl")


def test_name_auto_suffix_yml(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_yaml("data", Item())
    assert storage.exists("data.yml")


def test_name_already_has_correct_suffix(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    storage.write_model("config.md", Item())
    assert storage.exists("config.md")


def test_name_mismatched_suffix_raises(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        storage.write_model("data.jsonl", Item())


def test_name_mismatched_suffix_read_raises(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        storage.read_model("data.yml", Item)


# -- async_ proxy ---------------------------------------------------------


@pytest.mark.asyncio
async def test_async_proxy_get_put(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    await storage.async_.put("file.txt", b"hello async")
    result = await storage.async_.get("file.txt")
    assert result == b"hello async"


@pytest.mark.asyncio
async def test_async_proxy_exists_remove(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    await storage.async_.put("f.txt", b"x")
    assert await storage.async_.exists("f.txt") is True
    await storage.async_.remove("f.txt")
    assert await storage.async_.exists("f.txt") is False


@pytest.mark.asyncio
async def test_async_proxy_append(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    await storage.async_.append("log.txt", b"a\n")
    await storage.async_.append("log.txt", b"b\n")
    assert await storage.async_.get("log.txt") == b"a\nb\n"


@pytest.mark.asyncio
async def test_async_proxy_read_write_model(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    obj = Item(name="async", count=99)
    await storage.async_.write_model("cfg", obj, content="body")
    result = await storage.async_.read_model("cfg", Item)
    assert result is not None
    model, content = result
    assert model.name == "async"
    assert content == "body"


@pytest.mark.asyncio
async def test_async_proxy_append_read_models(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    await storage.async_.append_model("events", Item(name="a"))
    await storage.async_.append_model("events", Item(name="b"))
    items = await storage.async_.read_models("events", Item)
    assert len(items) == 2
    assert items[0].name == "a"


@pytest.mark.asyncio
async def test_async_proxy_read_write_yaml(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    await storage.async_.write_yaml("data", Item(name="y"))
    result = await storage.async_.read_yaml("data", Item)
    assert result is not None
    assert result.name == "y"


@pytest.mark.asyncio
async def test_async_proxy_sub_storage(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sub = await storage.async_.sub_storage("subdir")
    assert isinstance(sub, type(storage.async_))
    await sub.put("inner.txt", b"deep")
    assert await sub.get("inner.txt") == b"deep"


@pytest.mark.asyncio
async def test_async_proxy_abspath(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    path = await storage.async_.abspath()
    assert path == storage.abspath()
