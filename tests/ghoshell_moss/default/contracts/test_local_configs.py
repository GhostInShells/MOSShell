"""LocalConfigStore 单元测试 — 覆盖 basic CRUD / 缓存 / get_or_create / 序列化 /
invalidation / env 解析 / on_save 回调 / mode-aware 读写与 fallback."""

import os
import pytest

from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.contracts.configs import ConfigType, YamlConfigStore


# ══════════════════════════════════════════════════════════════════
# test model
# ══════════════════════════════════════════════════════════════════

class AppConfig(ConfigType):
    name: str = "MOSS"
    version: str = "1.0.0"
    debug: bool = False
    port: int = 8080

    @classmethod
    def conf_name(cls) -> str:
        return "app_config"


class ServerConfig(ConfigType):
    host: str = "127.0.0.1"
    port: int = 9000

    @classmethod
    def conf_name(cls) -> str:
        return "server"


# ══════════════════════════════════════════════════════════════════
# fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def store(tmp_path):
    """无 mode 的 YamlConfigStore."""
    return YamlConfigStore(LocalStorage(tmp_path))


@pytest.fixture
def mode_store(tmp_path):
    """mode_name='desktop' 的 YamlConfigStore."""
    return YamlConfigStore(LocalStorage(tmp_path), mode_name='desktop')


def _raw_write(store, filename: str, content: str) -> None:
    store._storage.put(filename, content.encode('utf-8'))


def _raw_exists(store, filename: str) -> bool:
    return store._storage.exists(filename)


def _disk_files(store) -> list[str]:
    """列出 storage 根目录下所有文件名."""
    p = store._storage.abspath()
    return sorted([f.name for f in p.iterdir() if f.is_file()])


# ══════════════════════════════════════════════════════════════════
# 1. Basic save / get
# ══════════════════════════════════════════════════════════════════

class TestSaveAndGet:
    def test_save_writes_yml_file(self, store):
        store.save(AppConfig(name="Ghoshell", version="2.0.0", debug=True))
        assert _raw_exists(store, "app_config.yml")

    def test_get_returns_saved_values(self, store):
        store.save(AppConfig(name="Ghoshell", version="2.0.0", debug=True))
        loaded = store.get(AppConfig)
        assert loaded.name == "Ghoshell"
        assert loaded.version == "2.0.0"
        assert loaded.debug is True
        assert loaded.port == 8080  # default

    def test_get_raises_when_no_file(self, store):
        with pytest.raises(FileNotFoundError, match="app_config"):
            store.get(AppConfig)

    def test_save_updates_disk_repeatedly(self, store):
        store.save(AppConfig(name="v1"))
        store.save(AppConfig(name="v2"))
        loaded = store.get(AppConfig)
        assert loaded.name == "v2"


# ══════════════════════════════════════════════════════════════════
# 2. Cache
# ══════════════════════════════════════════════════════════════════

class TestCache:
    def test_repeated_get_returns_same_instance(self, store):
        store.save(AppConfig(name="cache-test"))
        a = store.get(AppConfig)
        b = store.get(AppConfig)
        assert a is b

    def test_save_updates_cache(self, store):
        store.save(AppConfig(name="before"))
        first = store.get(AppConfig)
        store.save(AppConfig(name="after"))
        second = store.get(AppConfig)
        assert first.name == "before"
        assert second.name == "after"
        assert first is not second


# ══════════════════════════════════════════════════════════════════
# 3. get_or_create
# ══════════════════════════════════════════════════════════════════

class TestGetOrCreate:
    def test_creates_when_file_absent(self, store):
        conf = AppConfig(name="default-name")
        result = store.get_or_create(conf)
        assert result.name == "default-name"
        assert _raw_exists(store, "app_config.yml")

    def test_loads_disk_when_file_present(self, store):
        _raw_write(store, "app_config.yml",
                    "name: disk-value\nversion: 1.0.0\ndebug: false")
        fallback = AppConfig(name="should-be-ignored")
        result = store.get_or_create(fallback)
        assert result.name == "disk-value"

    def test_hits_cache(self, store):
        store.save(AppConfig(name="cached"))
        fallback = AppConfig(name="ignored")
        result = store.get_or_create(fallback)
        assert result.name == "cached"

    def test_multiple_config_types_independent(self, store):
        a = store.get_or_create(AppConfig(name="a"))
        b = store.get_or_create(ServerConfig(host="h"))
        assert a.name == "a"
        assert b.host == "h"
        assert _raw_exists(store, "app_config.yml")
        assert _raw_exists(store, "server.yml")


# ══════════════════════════════════════════════════════════════════
# 4. Serialization
# ══════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_yaml_includes_import_header(self, store):
        store.save(AppConfig())
        raw = store._storage.get("app_config.yml").decode('utf-8')
        assert "# dump from" in raw
        assert "AppConfig" in raw
        assert "name: MOSS" in raw

    def test_invalid_yaml_raises(self, store):
        _raw_write(store, "app_config.yml", "invalid: [yaml: : structure")
        with pytest.raises(Exception):
            store.get(AppConfig)

    def test_roundtrip_preserves_all_fields(self, store):
        conf = AppConfig(name="full", version="3.0", debug=True, port=3000)
        store.save(conf)
        loaded = store.get(AppConfig)
        assert loaded.model_dump() == conf.model_dump()


# ══════════════════════════════════════════════════════════════════
# 5. Invalidation
# ══════════════════════════════════════════════════════════════════

class TestInvalidation:
    def test_invalidate_by_type(self, store):
        store.save(AppConfig(name="disk"))
        store.get(AppConfig)  # cache it
        # mutate cache without writing disk
        store._cache["app_config"].name = "mutated"
        assert store.get(AppConfig).name == "mutated"
        store.invalidate(AppConfig)
        assert store.get(AppConfig).name == "disk"

    def test_invalidate_by_string(self, store):
        store.save(AppConfig(name="disk"))
        store.get(AppConfig)
        store._cache["app_config"].name = "mutated"
        store.invalidate("app_config")
        assert store.get(AppConfig).name == "disk"

    def test_invalidate_all(self, store):
        store.save(AppConfig(name="disk"))
        store.save(ServerConfig(host="1.2.3.4"))
        store.get(AppConfig)
        store.get(ServerConfig)
        store._cache["app_config"].name = "mutated"
        store._cache["server"].host = "mutated"
        store.invalidate()
        assert store.get(AppConfig).name == "disk"
        assert store.get(ServerConfig).host == "1.2.3.4"

    def test_invalidate_unknown_key_does_not_raise(self, store):
        store.invalidate("no_such_config")

    def test_invalidate_invalid_type_raises(self, store):
        with pytest.raises(TypeError):
            store.invalidate(42)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════
# 6. Env var resolution
# ══════════════════════════════════════════════════════════════════

class TestEnvResolution:
    def test_dollar_var_resolved(self, store, monkeypatch):
        monkeypatch.setenv("MY_APP_NAME", "from-env")
        store.save(AppConfig(name="$MY_APP_NAME"))
        loaded = store.get(AppConfig)
        assert loaded.name == "from-env"

    def test_unresolved_var_kept_as_is(self, store):
        store.save(AppConfig(name="$NO_SUCH_VAR_FOR_SURE"))
        loaded = store.get(AppConfig)
        assert loaded.name == "$NO_SUCH_VAR_FOR_SURE"


# ══════════════════════════════════════════════════════════════════
# 7. on_save callback
# ══════════════════════════════════════════════════════════════════

class TestOnSaveCallback:
    def test_fires_on_save(self, tmp_path):
        calls = []
        store = YamlConfigStore(LocalStorage(tmp_path),
                                on_save=lambda name: calls.append(name))
        store.save(AppConfig(name="x"))
        assert calls == ["app_config"]

    def test_fires_on_set_config_no_override(self, tmp_path):
        calls = []
        store = YamlConfigStore(LocalStorage(tmp_path),
                                on_save=lambda name: calls.append(name))
        store.set_config(AppConfig(name="mem-only"), override=False)
        assert calls == ["app_config"]

    def test_fires_on_set_config_override(self, tmp_path):
        calls = []
        store = YamlConfigStore(LocalStorage(tmp_path),
                                on_save=lambda name: calls.append(name))
        store.set_config(AppConfig(name="to-disk"), override=True)
        assert calls == ["app_config"]

    def test_not_configured_does_not_crash(self, tmp_path):
        store = YamlConfigStore(LocalStorage(tmp_path))
        store.save(AppConfig(name="x"))  # no on_save → must not raise


# ══════════════════════════════════════════════════════════════════
# 8. set_config
# ══════════════════════════════════════════════════════════════════

class TestSetConfig:
    def test_override_true_writes_to_disk(self, store):
        store.set_config(AppConfig(name="persisted"), override=True)
        assert _raw_exists(store, "app_config.yml")
        loaded = store.get(AppConfig)
        assert loaded.name == "persisted"

    def test_override_false_cache_only(self, store):
        store.set_config(AppConfig(name="cached"), override=False)
        assert not _raw_exists(store, "app_config.yml")
        # cached value is visible
        assert store.get(AppConfig).name == "cached"


# ══════════════════════════════════════════════════════════════════
# 9. get_config_path
# ══════════════════════════════════════════════════════════════════

class TestGetConfigPath:
    def test_no_mode(self, store):
        p = store.get_config_path("app_config")
        assert p.endswith("app_config.yml")

    def test_with_mode(self, mode_store):
        p = mode_store.get_config_path("app_config")
        assert p.endswith("app_config.desktop.yml")


# ══════════════════════════════════════════════════════════════════
# 10. Mode-aware read
# ══════════════════════════════════════════════════════════════════

class TestModeAwareRead:
    def test_mode_file_preferred_over_base(self, mode_store):
        _raw_write(mode_store, "app_config.yml",
                   "name: base-value\nversion: 1.0.0\ndebug: false")
        _raw_write(mode_store, "app_config.desktop.yml",
                   "name: desktop-value\nversion: 1.0.0\ndebug: true")
        loaded = mode_store.get(AppConfig)
        assert loaded.name == "desktop-value"
        assert loaded.debug is True

    def test_fallback_to_base_when_mode_file_absent(self, mode_store):
        _raw_write(mode_store, "app_config.yml",
                   "name: base-value\nversion: 1.0.0\ndebug: false")
        loaded = mode_store.get(AppConfig)
        assert loaded.name == "base-value"

    def test_raises_when_neither_exists(self, mode_store):
        with pytest.raises(FileNotFoundError, match="app_config"):
            mode_store.get(AppConfig)

    def test_read_path_ignores_other_mode_files(self, mode_store):
        """只匹配当前 mode，不错误匹配其他 mode 的文件."""
        _raw_write(mode_store, "app_config.other.yml",
                   "name: other-value\nversion: 1.0.0\ndebug: true")
        _raw_write(mode_store, "app_config.yml",
                   "name: base-value\nversion: 1.0.0\ndebug: false")
        loaded = mode_store.get(AppConfig)
        assert loaded.name == "base-value"


# ══════════════════════════════════════════════════════════════════
# 11. Mode-aware write
# ══════════════════════════════════════════════════════════════════

class TestModeAwareWrite:
    def test_save_writes_to_mode_file(self, mode_store):
        mode_store.save(AppConfig(name="desktop-only"))
        assert _raw_exists(mode_store, "app_config.desktop.yml")
        assert not _raw_exists(mode_store, "app_config.yml")

    def test_save_then_read_mode_file(self, mode_store):
        mode_store.save(AppConfig(name="mode-value", debug=True))
        loaded = mode_store.get(AppConfig)
        assert loaded.name == "mode-value"
        assert loaded.debug is True

    def test_independent_base_and_mode(self, mode_store):
        """base 和 mode 文件独立——各自保存不互相覆盖."""
        # save to base (via store without mode)
        base_store = YamlConfigStore(mode_store._storage)
        base_store.save(AppConfig(name="base-only"))
        assert _raw_exists(mode_store, "app_config.yml")

        # save to mode
        mode_store.save(AppConfig(name="mode-only"))
        assert _raw_exists(mode_store, "app_config.desktop.yml")

        # both exist independently
        assert base_store.get(AppConfig).name == "base-only"
        assert mode_store.get(AppConfig).name == "mode-only"


# ══════════════════════════════════════════════════════════════════
# 12. Mode-aware get_or_create
# ══════════════════════════════════════════════════════════════════

class TestModeAwareGetOrCreate:
    def test_creates_mode_file_when_none_exist(self, mode_store):
        conf = AppConfig(name="fresh")
        result = mode_store.get_or_create(conf)
        assert result.name == "fresh"
        assert _raw_exists(mode_store, "app_config.desktop.yml")
        assert not _raw_exists(mode_store, "app_config.yml")

    def test_falls_back_to_base_then_saves_to_mode(self, mode_store):
        """如果 base 文件存在但 mode 文件不存在，读 base，但后续 save 写 mode."""
        _raw_write(mode_store, "app_config.yml",
                   "name: base-value\nversion: 1.0.0\ndebug: false")
        # get_or_create 从 base 读到值
        result = mode_store.get_or_create(AppConfig(name="ignored"))
        assert result.name == "base-value"

        # 此时 save 写 mode 文件
        mode_store.save(AppConfig(name="now-mode"))
        assert _raw_exists(mode_store, "app_config.desktop.yml")
        loaded = mode_store.get(AppConfig)
        assert loaded.name == "now-mode"


# ══════════════════════════════════════════════════════════════════
# 13. Cache key is conf_name (mode-independent)
# ══════════════════════════════════════════════════════════════════

class TestCacheKeyModeIndependent:
    def test_same_config_name_shares_cache_regardless_of_mode(self, mode_store):
        """两个不同 mode 的 store 不应该共享缓存——但同一个 store 内 cache key 不含 mode."""
        mode_store.save(AppConfig(name="val"))
        # cache key is "app_config", not "app_config.desktop"
        assert "app_config" in mode_store._cache
        assert "app_config.desktop" not in mode_store._cache

    def test_invalidate_clears_cache_across_modes(self, mode_store):
        mode_store.save(AppConfig(name="val"))
        mode_store.invalidate("app_config")
        assert "app_config" not in mode_store._cache
