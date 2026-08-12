"""MarkdownKnowledgeBase — K3 正规化测试 (glob+frontmatter 参数化)."""

import asyncio

import pytest

from ghoshell_moss.resources.markdown_kb import (
    MarkdownKnowledgeBase,
    MarkdownKnowledgeBaseMeta,
    extract_meta,
)

MD_TEMPLATE = """---
title: {title}
description: {description}
---

正文内容。
"""

SKILL_TEMPLATE = """---
name: {name}
description: {description}
---

技能正文。
"""


@pytest.fixture
def doc_tree(tmp_path):
    """默认 .md 树: README + 两个文档."""
    (tmp_path / "README.md").write_text(
        MD_TEMPLATE.format(title="Docs", description="知识库目录自身"), encoding="utf-8"
    )
    (tmp_path / "a.md").write_text(
        MD_TEMPLATE.format(title="Alpha", description="文档 alpha"), encoding="utf-8"
    )
    (tmp_path / "b.md").write_text(
        MD_TEMPLATE.format(title="Beta", description="文档 beta"), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def skill_tree(tmp_path):
    """skills 树: <name>/SKILL.md."""
    for name, desc in [("alpha", "技能 alpha"), ("beta", "技能 beta")]:
        skill = tmp_path / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(
            SKILL_TEMPLATE.format(name=f"{name}-skill", description=desc), encoding="utf-8"
        )
    (tmp_path / "noise.txt").write_text("不是技能文件", encoding="utf-8")
    return tmp_path


def test_scan_default_readme_first(doc_tree):
    kb = MarkdownKnowledgeBase(host="t", root=doc_tree)
    kb.scan()
    assert [m.path for m in kb.metas] == ["README.md", "a.md", "b.md"]


def test_scan_default_derives_title_and_description(doc_tree):
    kb = MarkdownKnowledgeBase(host="t", root=doc_tree)
    kb.scan()
    by_path = {m.path: m for m in kb.metas}
    assert by_path["a.md"].title == "Alpha"
    assert by_path["a.md"].description == "文档 alpha"


def test_scan_skills_pattern_maps_name_to_title(skill_tree):
    kb = MarkdownKnowledgeBase(
        host="t", root=skill_tree,
        pattern="*/SKILL.md", keys=["name", "description"],
    )
    kb.scan()
    by_path = {m.path: m for m in kb.metas}
    assert set(by_path) == {"alpha/SKILL.md", "beta/SKILL.md"}
    assert by_path["alpha/SKILL.md"].title == "alpha-skill"
    assert by_path["alpha/SKILL.md"].description == "技能 alpha"


def test_scan_limit(doc_tree):
    kb = MarkdownKnowledgeBase(host="t", root=doc_tree, limit=2)
    kb.scan()
    assert len(kb.metas) == 2


def test_scan_missing_root(tmp_path):
    kb = MarkdownKnowledgeBase(host="t", root=tmp_path / "nope")
    kb.scan()
    assert kb.metas == []


def test_extract_meta_default_derives(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("---\ntitle: T\n---\n\n正文第一句。", encoding="utf-8")
    assert extract_meta(p) == {"title": "T", "description": "正文第一句。"}


def test_extract_meta_keys(tmp_path):
    p = tmp_path / "skill.md"
    p.write_text("---\nname: s1\ndescription: d1\nextra: x1\n---\n\n正文。", encoding="utf-8")
    assert extract_meta(p, keys=["name", "description"]) == {
        "name": "s1", "description": "d1",
    }


def test_recall_without_llm_funcs_raises_not_implemented(doc_tree):
    kb = MarkdownKnowledgeBase(host="t", root=doc_tree)
    kb.scan()
    with pytest.raises(NotImplementedError):
        asyncio.run(kb.recall("查询什么"))


class _FakeResolver:
    def get_model(self, tag=None):
        return object()


class _FakeConfig:
    def resolve(self):
        return _FakeResolver()


class _FakeResult:
    def __init__(self):
        self.locators = ["markdown-kb://t/a.md"]
        self.reasoning = "匹配 alpha"


class _FakeCallReturn:
    def __init__(self):
        self.result = _FakeResult()


class _FakeFuncs:
    def __init__(self):
        self.kwargs = None

    async def call(self, **kwargs):
        self.kwargs = kwargs
        return _FakeCallReturn()


def test_recall_with_llm_funcs(monkeypatch, doc_tree):
    fake = _FakeFuncs()
    monkeypatch.setattr(
        "ghoshell_moss.contracts.llms.LLMConfig", _FakeConfig,
    )
    kb = MarkdownKnowledgeBase(host="t", root=doc_tree, llm_funcs=fake)
    kb.scan()
    rec = asyncio.run(kb.recall("查询 alpha"))
    assert rec.locators == ["markdown-kb://t/a.md"]
    assert rec.reasoning == "匹配 alpha"
    assert fake.kwargs["result_type"].__name__ == "_RecallResult"


def test_meta_factory_configuration():
    meta = MarkdownKnowledgeBaseMeta(
        host="moss-skills", root="/tmp/skills",
        pattern="*/SKILL.md", keys=["name", "description"], max_depth=3,
    )
    assert meta.scheme() == "markdown-kb"
    assert meta.host == "moss-skills"
    assert "参数化" in meta.description()
