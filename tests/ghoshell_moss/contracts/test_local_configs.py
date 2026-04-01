import pytest
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.contracts.configs import ConfigType, YamlConfigStore


# 1. 定义一个用于测试的配置模型
class AppConfig(ConfigType):
    name: str = "MOSS"
    version: str = "1.0.0"
    debug: bool = False

    @classmethod
    def conf_name(cls) -> str:
        return "app_config"


@pytest.fixture
def config_store(tmp_path):
    """创建基于临时目录的 YamlConfigStore"""
    storage = LocalStorage(tmp_path)
    return YamlConfigStore(storage)


def test_save_and_get_config(config_store):
    """测试基本的配置保存与读取"""
    conf = AppConfig(name="Ghoshell", version="2.0.0", debug=True)
    config_store.save(conf)

    # 验证磁盘上生成了文件 (YamlConfigStore 会自动加 .yml)
    assert (config_store._storage.abspath() / "app_config.yml").exists()

    # 读取并验证内容
    loaded = config_store.get(AppConfig)
    assert isinstance(loaded, AppConfig)
    assert loaded.name == "Ghoshell"
    assert loaded.debug is True


def test_config_memory_cache_consistency(config_store):
    """测试内存缓存：多次 get 应该返回同一个对象实例"""
    conf = AppConfig(name="CacheTest")
    config_store.save(conf)

    first_get = config_store.get(AppConfig)
    second_get = config_store.get(AppConfig)

    # 验证物理上是同一个 Python 对象（内存地址一致）
    assert first_get is second_get

    # 验证修改 save 后，缓存同步更新
    conf.name = "UpdatedName"
    config_store.save(conf)

    third_get = config_store.get(AppConfig)
    assert third_get.name == "UpdatedName"
    assert third_get is conf  # save 会更新缓存为当前对象


def test_get_or_create(config_store):
    """测试 get_or_create 逻辑"""
    # 初始状态：文件不存在
    default_conf = AppConfig(name="Default")

    # 第一次调用：应该创建并返回传入的对象
    result = config_store.get_or_create(default_conf)
    assert result.name == "Default"
    assert (config_store._storage.abspath() / "app_config.yml").exists()

    # 修改磁盘文件模拟外部变动（清空缓存后测试）
    config_store.invalidate()
    path = config_store._storage.abspath() / "app_config.yml"
    path.write_text("name: ExternalUpdate\nversion: 1.0.0\ndebug: false")

    # 第二次调用：文件已存在，应该加载磁盘内容而不是使用传入的对象
    another_default = AppConfig(name="ShouldIgnoreMe")
    existing = config_store.get_or_create(another_default)
    assert existing.name == "ExternalUpdate"


def test_yaml_marshal_with_header(config_store):
    """测试序列化时是否正确包含了 import path 注释"""
    conf = AppConfig()
    config_store.save(conf)

    # 直接通过 storage 读取原始 bytes
    raw_bytes = config_store._storage.get("app_config.yml")
    content = raw_bytes.decode('utf-8')

    # 验证包含注释行
    assert "# dump from" in content
    assert "AppConfig" in content
    # 验证 YAML 内容
    assert "name: MOSS" in content


def test_load_invalid_yaml_raises_error(config_store):
    """测试加载格式错误的 YAML 时应抛出异常"""
    # 手动写入坏数据
    config_store._storage.put("app_config.yml", b"invalid: [yaml: : structure")

    with pytest.raises(Exception):  # yaml.scanner.ScannerError 或 ValueError
        config_store.get(AppConfig)


def test_invalidate_cache(config_store):
    """测试缓存清理功能"""
    conf = AppConfig(name="Original")
    config_store.save(conf)

    # 预加载
    config_store.get(AppConfig)
    assert AppConfig in config_store._cache

    # 清理
    config_store.invalidate(AppConfig)
    assert AppConfig not in config_store._cache

    # 全局清理
    config_store.get(AppConfig)
    config_store.invalidate()
    assert len(config_store._cache) == 0
