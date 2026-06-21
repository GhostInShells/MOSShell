import pytest
from ghoshell_moss.contracts.workspace import LocalStorage  # 替换为你的实际模块名


@pytest.fixture
def storage(tmp_path):
    """创建一个基于临时目录的 Storage 实例"""
    return LocalStorage(tmp_path)


def test_put_and_get(storage):
    """测试基本存取功能"""
    file_path = "test.txt"
    content = b"hello world"
    storage.put(file_path, content)

    assert storage.exists(file_path)
    assert storage.get(file_path) == content


def test_nested_path_auto_creation(storage):
    """测试存储深层目录时是否自动创建父文件夹"""
    deep_path = "a/b/c/file.dat"
    content = b"deep data"
    storage.put(deep_path, content)

    assert storage.exists(deep_path)
    assert storage.get(deep_path) == content


def test_sub_storage(storage):
    """测试子存储隔离"""
    storage.put("shared/base.txt", b"base")

    # 创建子存储
    sub = storage.sub_storage("shared")
    assert sub.get("base.txt") == b"base"

    # 在子存储中写入，父存储应能感知
    sub.put("sub.txt", b"sub_content")
    assert storage.get("shared/sub.txt") == b"sub_content"


def test_remove_file_and_dir(storage):
    """测试删除功能"""
    # 删除文件
    storage.put("file.txt", b"data")
    storage.remove("file.txt")
    assert not storage.exists("file.txt")

    # 删除文件夹
    storage.put("dir/f1.txt", b"1")
    storage.put("dir/f2.txt", b"2")
    storage.remove("dir")
    assert not storage.exists("dir")


def test_path_escape_prevention(storage):
    """测试路径泄露防御（核心安全测试）"""
    # 模拟一个外部文件（在 storage 根目录之外）
    outside_file = storage.abspath().parent / "danger.txt"
    outside_file.write_bytes(b"secret")

    # 尝试通过相对路径访问外部
    malicious_path = "../danger.txt"

    # 验证是否抛出 PermissionError
    with pytest.raises(PermissionError, match="Path escape detected"):
        storage.get(malicious_path)

    with pytest.raises(PermissionError, match="Path escape detected"):
        storage.put(malicious_path, b"hack")


def test_exists_with_invalid_path(storage):
    """测试探测外部路径时 exists 应安全返回 False 或报错"""
    # 根据你的实现，如果是 PermissionError，exists 捕捉并返回 False 也是合理的
    assert storage.exists("../../etc/passwd") is False


def test_abspath_property(storage, tmp_path):
    """验证绝对路径返回是否正确"""
    # resolve() 会处理软链接等，确保一致性
    assert storage.abspath() == tmp_path.resolve()
