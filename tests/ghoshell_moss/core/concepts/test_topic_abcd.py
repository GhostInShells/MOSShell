import pytest
from ghoshell_moss.core.concepts.topic import TopicMeta
from pydantic import ValidationError


def test_topic_name_validation():
    # 格式: (输入值, 期望是否通过)
    test_cases = [
        # --- 合法 Case ---
        ("", True),  # 允许为空
        ("test", True),  # 单级简单名称
        ("test/foo/bar", True),  # 多级路径
        ("v1.0/sensor-01/status", True),  # 包含 . 和 -
        ("A/B/C", True),  # 大写字母
        ("123/456", True),  # 数字开头
        ("my_topic/v1", True),  # 如下划线已加入正则，则应为 True

        # --- 非法 Case ---
        ("/", False),  # 仅有一个斜杠
        ("/test", False),  # 以斜杠开头
        ("test/", False),  # 以斜杠结尾
        ("test//foo", False),  # 连续斜杠
        ("test/ /foo", False),  # 包含空格
        ("test/!@#", False),  # 包含非法特殊字符
        ("test/中文", False),  # 包含非 ASCII 字符 (除非你有意允许)
        ("..", False),  # 纯点号
        ("./foo", False),  # 点号开头
    ]

    for name, should_pass in test_cases:
        try:
            TopicMeta(name=name)
            passed = True
        except ValidationError:
            passed = False

        assert passed == should_pass, f"测试失败! 输入: '{name}', 期望: {should_pass}, 实际: {passed}"


# 如果你使用的是 pytest，可以写得更优雅一点：
@pytest.mark.parametrize("name, should_pass", [
    ("", True),
    ("a/b/c", True),
    ("/a", False),
    ("a/", False),
    ("a//b", False),
    ("a b", False),
])
def test_topic_name_parametrized(name, should_pass):
    if should_pass:
        assert TopicMeta(name=name).name == name
    else:
        with pytest.raises(ValidationError):
            TopicMeta(name=name)
