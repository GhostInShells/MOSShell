import pytest
from unittest.mock import patch

from ghoshell_moss.contracts.llms import (
    ServiceConfig,
    ModelConfig,
    Provider,
    ResolvedModel,
    LLMConfig,
    MessageContentConverter,
    register_converter,
    clear_converters,
)
from ghoshell_moss.message import Message, Content
from ghoshell_container import IoCContainer
from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalStorage


def _sample_config() -> LLMConfig:
    return LLMConfig(
        default=Provider(
            service=ServiceConfig(
                name="anthropic",
                base_url="https://api.anthropic.com",
                api_key="$ANTHROPIC_API_KEY",
                protocol="anthropic",
            ),
            default=ModelConfig(
                model="claude-sonnet-4-6",
                tags={"small_fast_model": "claude-haiku-4-5-20251001"},
                content_types=["text", "image"],
            ),
            models={
                "opus": ModelConfig(
                    model="claude-opus-4-7",
                    content_types=["text", "image"],
                ),
            },
        ),
        providers={
            "openai": Provider(
                service=ServiceConfig(
                    name="openai",
                    base_url="https://api.openai.com",
                    api_key="$OPENAI_API_KEY",
                    protocol="openai",
                ),
                default=ModelConfig(
                    model="gpt-5",
                    content_types=["text"],
                ),
            ),
        },
    )


# ---------------------------------------------------------------------------
# ServiceConfig
# ---------------------------------------------------------------------------

class TestServiceConfig:
    def test_default_protocol(self):
        s = ServiceConfig(name="test", base_url="http://localhost")
        assert s.protocol == "anthropic"

    def test_default_api_key(self):
        s = ServiceConfig(name="test", base_url="http://localhost")
        assert s.api_key == "$ANTHROPIC_API_KEY"


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig()
        assert m.model == "$ANTHROPIC_MODEL"
        assert m.context_window == 200000
        assert m.max_output_tokens == 4096
        assert m.content_types == ["text"]
        assert m.tags == {}
        assert m.converters == {}

    def test_accepts_known_type(self):
        m = ModelConfig(content_types=["text", "image"])
        assert m.accepts(Content(type="text"))
        assert m.accepts(Content(type="image"))

    def test_accepts_unknown_type(self):
        m = ModelConfig(content_types=["text"])
        assert not m.accepts(Content(type="audio"))
        assert not m.accepts(Content(type="image"))

    def test_accepts_empty_content_types(self):
        m = ModelConfig(content_types=[])
        assert not m.accepts(Content(type="text"))

    def test_accepts_wildcard(self):
        m = ModelConfig(content_types=["*"])
        assert m.accepts(Content(type="text"))
        assert m.accepts(Content(type="image"))
        assert m.accepts(Content(type="audio"))
        assert m.accepts(Content(type="anything_else"))

    def test_unwrap_tag_returns_self_on_empty_tag(self):
        m = ModelConfig(model="base", tags={"pro": "big-model"})
        assert m.unwrap_tag("") is m

    def test_unwrap_tag_returns_self_on_unknown_tag(self):
        m = ModelConfig(model="base", tags={"pro": "big-model"})
        assert m.unwrap_tag("nonexistent") is m

    def test_unwrap_tag_resolves_and_copies(self):
        m = ModelConfig(model="base", tags={"pro": "big-model"})
        result = m.unwrap_tag("pro")
        assert result is not m
        assert result.model == "big-model"
        assert "default" in result.tags
        # 原始对象未被污染
        assert "default" not in m.tags

    def test_convert_native_passthrough(self):
        m = ModelConfig(content_types=["text", "image"])
        msg = Message(contents=[
            Content(type="text", text="hello"),
            Content(type="image", source={"media_type": "image/png"}),
        ])
        result = m.convert(None, msg)
        assert len(result.contents) == 2
        assert result.contents[0]["type"] == "text"
        assert result.contents[0]["text"] == "hello"
        assert result.contents[1]["type"] == "image"

    def test_convert_degradation_to_text(self):
        """不支持的类型且无 converter 时，降级为文本表示。"""
        m = ModelConfig(content_types=["text"])
        msg = Message(contents=[
            Content(type="image", source={"media_type": "image/png"}),
            Content(type="text", text="hello"),
        ])
        with patch.object(Message, "content_as_string", return_value="[degraded image]"):
            result = m.convert(None, msg)

        assert len(result.contents) == 2
        # image 被降级为 text
        assert result.contents[0]["type"] == "text"
        assert result.contents[0]["text"] == "[degraded image]"
        # 原生 text 原样保留
        assert result.contents[1]["type"] == "text"
        assert result.contents[1]["text"] == "hello"

    def test_convert_degradation_empty_string_skipped(self):
        """降级为空字符串时不添加 content。"""
        m = ModelConfig(content_types=["text"])
        msg = Message(contents=[
            Content(type="image", source={}),
        ])
        with patch.object(Message, "content_as_string", return_value=""):
            result = m.convert(None, msg)

        assert len(result.contents) == 0


    # ---- converter 适配 ----

    def test_convert_with_registered_converter(self):
        """converter 适配优先于文本降级。"""
        register_converter("test.fake:img2txt", _FakeImgConverter())
        try:
            m = ModelConfig(
                content_types=["text"],
                converters={"image": "test.fake:img2txt"},
            )
            msg = Message(contents=[
                Content(type="image", source={"media_type": "image/png"}),
            ])
            result = m.convert(None, msg)

            assert len(result.contents) == 1
            assert result.contents[0]["type"] == "text"
            assert result.contents[0]["text"] == "[converted from image]"
        finally:
            clear_converters()

    def test_convert_converter_returns_empty_then_degrade(self):
        """converter 无输出时继续走降级。"""
        register_converter("test.fake:empty", _EmptyConverter())
        try:
            m = ModelConfig(
                content_types=["text"],
                converters={"image": "test.fake:empty"},
            )
            msg = Message(contents=[
                Content(type="image", source={}),
            ])
            with patch.object(Message, "content_as_string", return_value="[degraded]"):
                result = m.convert(None, msg)

            assert len(result.contents) == 1
            assert result.contents[0]["text"] == "[degraded]"
        finally:
            clear_converters()

    def test_convert_content_caches_missing_converter(self):
        """import 失败时缓存 None，后续不再重试。"""
        m = ModelConfig(
            content_types=["text"],
            converters={"image": "nonexistent.module:func"},
        )
        msg = Message(contents=[
            Content(type="image", source={}),
        ])
        with patch.object(Message, "content_as_string", return_value="[fallback]"):
            result1 = m.convert(None, msg)
            result2 = m.convert(None, msg)

        assert result1.contents[0]["text"] == "[fallback]"
        assert result2.contents[0]["text"] == "[fallback]"


# ---------------------------------------------------------------------------
# Fake converters for testing
# ---------------------------------------------------------------------------

class _FakeImgConverter(MessageContentConverter):
    def convert(self, container: IoCContainer, content: Content):
        yield Content(type="text", text="[converted from image]")


class _EmptyConverter(MessageContentConverter):
    def convert(self, container: IoCContainer, content: Content):
        yield from []


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class TestProvider:
    def test_get_model_by_name(self):
        p = Provider(
            service=ServiceConfig(name="test", base_url="http://x"),
            default=ModelConfig(model="default-model"),
            models={"special": ModelConfig(model="special-model")},
        )
        result = p.get_model("special")
        assert result.model == "special-model"

    def test_get_model_falls_back_to_default(self):
        p = Provider(
            service=ServiceConfig(name="test", base_url="http://x"),
            default=ModelConfig(model="default-model"),
            models={},
        )
        result = p.get_model("nonexistent")
        assert result.model == "default-model"

    def test_get_model_with_tag(self):
        p = Provider(
            service=ServiceConfig(name="test", base_url="http://x"),
            default=ModelConfig(
                model="default-model",
                tags={"small": "small-model"},
            ),
        )
        result = p.get_model("", "small")
        assert result.model == "small-model"


# ---------------------------------------------------------------------------
# ResolvedModel
# ---------------------------------------------------------------------------

class TestResolvedModel:
    def test_client_protocol(self):
        rm = ResolvedModel(
            service=ServiceConfig(name="test", base_url="http://x", protocol="openai"),
            model=ModelConfig(model="gpt-5"),
        )
        assert rm.client_protocol == "openai"

    def test_client_protocol_default(self):
        rm = ResolvedModel(
            service=ServiceConfig(name="test", base_url="http://x"),
            model=ModelConfig(model="claude"),
        )
        assert rm.client_protocol == "anthropic"


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    @pytest.fixture
    def config(self):
        return _sample_config()

    def test_conf_name(self):
        assert LLMConfig.conf_name() == "llms"

    # ---- get_model ----

    def test_get_model_zero_arg_returns_default(self, config):
        result = config.get_model()
        assert result.model.model == "claude-sonnet-4-6"
        assert result.service.name == "anthropic"

    def test_get_model_by_provider(self, config):
        result = config.get_model(provider="openai")
        assert result.model.model == "gpt-5"
        assert result.service.name == "openai"

    def test_get_model_by_provider_and_tag(self, config):
        """tag 作用于 provider 的 default model 上。"""
        result = config.get_model(provider="anthropic", tag="small_fast_model")
        assert result.model.model == "claude-haiku-4-5-20251001"
        assert result.service.name == "anthropic"

    def test_get_model_by_provider_and_model_name(self, config):
        result = config.get_model(provider="anthropic", model="opus")
        assert result.model.model == "claude-opus-4-7"

    def test_get_model_by_name_across_providers(self, config):
        """指定 model 名但未指定 provider 时，搜索所有 provider 的 models 字典。"""
        result = config.get_model(model="opus")
        assert result.model.model == "claude-opus-4-7"
        assert result.service.name == "anthropic"

    def test_get_model_by_name_not_found_falls_back(self, config):
        """model 名只在 models 字典中精确匹配，不匹配 default。"""
        result = config.get_model(model="gpt-5")
        assert result.model.model == "claude-sonnet-4-6"

    def test_get_model_provider_not_found_falls_back(self, config):
        result = config.get_model(provider="nonexistent")
        assert result.model.model == "claude-sonnet-4-6"

    def test_get_model_no_fallback_raises(self, config):
        with pytest.raises(KeyError, match="Provider"):
            config.get_model(provider="nonexistent", no_fallback=True)

    def test_get_model_name_not_found_no_fallback_raises(self, config):
        with pytest.raises(KeyError, match="not found in any provider"):
            config.get_model(model="nonexistent", no_fallback=True)

    # ---- list_models ----

    def test_list_models_all(self, config):
        result = config.list_models()
        assert len(result) == 3
        models = {(r.service.name, r.model.model) for r in result}
        assert models == {
            ("anthropic", "claude-sonnet-4-6"),
            ("anthropic", "claude-opus-4-7"),
            ("openai", "gpt-5"),
        }

    def test_list_models_filter_by_provider(self, config):
        result = config.list_models(provider="anthropic")
        assert len(result) == 2
        assert all(r.service.name == "anthropic" for r in result)

    def test_list_models_unknown_provider(self, config):
        assert config.list_models(provider="nonexistent") == []

    # ---- get_service ----

    def test_get_service(self, config):
        s = config.get_service("anthropic")
        assert s.base_url == "https://api.anthropic.com"

    def test_get_service_missing(self, config):
        with pytest.raises(KeyError, match="Service"):
            config.get_service("nonexistent")

    # ---- services property ----

    def test_services(self, config):
        result = config.services
        names = {s.name for s in result}
        assert names == {"anthropic", "openai"}

    def test_services_dedup(self):
        """default 和 providers 有同名服务时去重。"""
        svc = ServiceConfig(name="shared", base_url="http://x")
        c = LLMConfig(
            default=Provider(service=svc, default=ModelConfig(model="m1")),
            providers={
                "other": Provider(service=svc, default=ModelConfig(model="m2")),
            },
        )
        assert len(c.services) == 1


# ---------------------------------------------------------------------------
# LLMConfig + YamlConfigStore 集成
# ---------------------------------------------------------------------------

class TestLLMConfigIntegration:
    @pytest.fixture
    def store(self, tmp_path):
        storage = LocalStorage(tmp_path)
        return YamlConfigStore(storage)

    @pytest.fixture
    def store_with_env(self, tmp_path):
        storage = LocalStorage(tmp_path)
        return YamlConfigStore(
            storage,
            environ={
                "ANTHROPIC_API_KEY": "sk-ant-test-123",
                "OPENAI_API_KEY": "sk-oai-test-456",
            },
        )

    def test_save_and_load_roundtrip(self, store):
        config = _sample_config()
        store.save(config)
        loaded = store.get(LLMConfig)

        # 结构一致
        assert loaded.default.service.name == "anthropic"
        assert loaded.default.default.model == "claude-sonnet-4-6"
        assert "opus" in loaded.default.models
        assert "openai" in loaded.providers

    def test_env_var_resolved_on_load(self, store_with_env):
        config = _sample_config()
        store_with_env.save(config)
        loaded = store_with_env.get(LLMConfig)

        assert loaded.default.service.api_key == "sk-ant-test-123"
        assert loaded.providers["openai"].service.api_key == "sk-oai-test-456"

    def test_env_var_raw_on_disk(self, store_with_env):
        """磁盘上保留 $ENV_VAR 原始引用，不写入解析后的值。"""
        config = _sample_config()
        store_with_env.save(config)

        raw = store_with_env._storage.get("llms.yml")
        assert b"$ANTHROPIC_API_KEY" in raw
        assert b"sk-ant-test-123" not in raw

    def test_get_or_create_creates_default(self, store):
        """get_or_create 在文件不存在时写入默认配置。"""
        default_conf = LLMConfig()
        result = store.get_or_create(default_conf)

        assert isinstance(result, LLMConfig)
        assert result.default.service.name == "anthropic"
        # 确认文件已落盘
        path = store.get_config_path(LLMConfig.conf_name())
        import pathlib
        assert pathlib.Path(path).exists()

    def test_invalidate_cache(self, store_with_env):
        """invalidate 之后重新从磁盘读取。"""
        config = _sample_config()
        store_with_env.save(config)

        first = store_with_env.get(LLMConfig)
        store_with_env.invalidate(LLMConfig)
        second = store_with_env.get(LLMConfig)

        assert first.default.service.api_key == second.default.service.api_key
        assert first is not second  # 不同实例

    def test_mode_specific_config(self, tmp_path):
        """mode_name 非空时读写独立文件。"""
        storage = LocalStorage(tmp_path)
        base_store = YamlConfigStore(storage)
        mode_store = YamlConfigStore(storage, mode_name="dev")

        base = LLMConfig()
        base.default.service.name = "anthropic"
        base_store.save(base)

        dev = LLMConfig()
        dev.default.service.name = "deepseek"
        mode_store.save(dev)

        # 不同 mode 读到不同配置
        loaded_base = base_store.get(LLMConfig)
        loaded_dev = mode_store.get(LLMConfig)

        assert loaded_base.default.service.name == "anthropic"
        assert loaded_dev.default.service.name == "deepseek"
