import frontmatter


def test_frontmatter_returns_nested_dict():
    content = """---
name: demo
meta:
  author:
    name: moss
    tags:
      - shell
      - ghost
  version: 1
---
body text
"""
    post = frontmatter.loads(content)

    assert post.content == "body text"
    assert post["name"] == "demo"

    meta = post["meta"]
    assert isinstance(meta, dict)
    assert meta["version"] == 1
    assert isinstance(meta["author"], dict)
    assert meta["author"]["name"] == "moss"
    assert meta["author"]["tags"] == ["shell", "ghost"]


def test_frontmatter_nested_dict_roundtrip():
    post = frontmatter.Post("body", nested={"a": {"b": {"c": 1}}})
    dumped = frontmatter.dumps(post)

    reloaded = frontmatter.loads(dumped)
    assert reloaded["nested"] == {"a": {"b": {"c": 1}}}
