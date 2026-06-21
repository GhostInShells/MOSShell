import pytest
from ghoshell_moss.contracts.llms import (
    LLMServiceConfig,
    LLMModelConfig,
    LLMModelAndService,
    LLMConfig,
    load_converter,
)


def _sample_config() -> LLMConfig:
    return LLMConfig(
        services=[
            LLMServiceConfig(
                name="anthropic",
                base_url="https://api.anthropic.com",
                api_key="$ANTHROPIC_API_KEY",
                api_type="anthropic",
            ),
            LLMServiceConfig(
                name="openai",
                base_url="https://api.openai.com",
                api_key="$OPENAI_API_KEY",
                api_type="openai",
            ),
        ],
        models=[
            LLMModelConfig(
                model="claude-sonnet-4-6",
                service="anthropic",
                model_type="default",
                protocols=["text", "image"],
            ),
            LLMModelConfig(
                model="claude-opus-4-7",
                service="anthropic",
                model_type="pro",
                protocols=["text", "image"],
            ),
            LLMModelConfig(
                model="claude-haiku-4-5-20251001",
                service="anthropic",
                model_type="flash",
                protocols=["text"],
            ),
            LLMModelConfig(
                model="gpt-5",
                service="openai",
                model_type="default",
                protocols=["text"],
            ),
        ],
        default_model="claude-sonnet-4-6",
    )


class TestLLMServiceConfig:
    def test_default_api_type(self):
        s = LLMServiceConfig(name="test", base_url="http://localhost")
        assert s.api_type == "anthropic"

    def test_default_api_key(self):
        s = LLMServiceConfig(name="test", base_url="http://localhost")
        assert s.api_key == "$ANTHROPIC_API_KEY"


class TestLLMModelConfig:
    def test_accepts_known_protocol(self):
        m = LLMModelConfig(
            model="test", service="s", protocols=["text", "image"]
        )
        assert m.accepts("text")
        assert m.accepts("image")

    def test_accepts_unknown_protocol(self):
        m = LLMModelConfig(model="test", service="s", protocols=["text"])
        assert not m.accepts("image")
        assert not m.accepts("audio")

    def test_accepts_empty_protocols(self):
        m = LLMModelConfig(model="test", service="s", protocols=[])
        assert not m.accepts("text")

    def test_defaults(self):
        m = LLMModelConfig(model="test", service="s")
        assert m.model_type == "default"
        assert m.context_window == 200000
        assert m.max_output_tokens == 4096
        assert m.protocols == ["text"]
        assert m.converter is None

    def test_model_type_constants(self):
        assert LLMModelConfig.MODEL_TYPE_DEFAULT == "default"
        assert LLMModelConfig.MODEL_TYPE_PRO == "pro"
        assert LLMModelConfig.MODEL_TYPE_FLASH == "flash"


class TestLLMConfig:

    @pytest.fixture
    def config(self):
        return _sample_config()

    def test_conf_name(self):
        assert LLMConfig.conf_name() == "llm"

    def test_yaml_roundtrip(self, config):
        yaml_str = config.to_yaml()
        reloaded = LLMConfig.from_yaml(yaml_str)
        assert reloaded.default_model == config.default_model
        assert len(reloaded.services) == len(config.services)
        assert len(reloaded.models) == len(config.models)

    def test_get_model_zero_arg_returns_default(self, config):
        result = config.get_model()
        assert result.model.model == "claude-sonnet-4-6"
        assert result.service.name == "anthropic"

    def test_get_model_by_service(self, config):
        result = config.get_model(service="openai")
        assert result.model.model == "gpt-5"
        assert result.service.name == "openai"

    def test_get_model_by_model_type(self, config):
        result = config.get_model(model_type="pro")
        assert result.model.model == "claude-opus-4-7"

    def test_get_model_by_service_and_type(self, config):
        result = config.get_model(service="anthropic", model_type="flash")
        assert result.model.model == "claude-haiku-4-5-20251001"

    def test_get_model_no_match_falls_back_to_default(self, config):
        result = config.get_model(service="anthropic", model_type="nonexistent")
        assert result.model.model == "claude-sonnet-4-6"

    def test_get_model_no_fallback_raises(self, config):
        with pytest.raises(KeyError, match="No model matched"):
            config.get_model(service="nonexistent", no_fallback=True)

    def test_get_model_empty_default_raises(self):
        empty = LLMConfig(services=[], models=[], default_model="")
        with pytest.raises(ValueError, match="No default_model configured"):
            empty.get_model()

    def test_get_model_default_not_found_raises(self):
        bad = LLMConfig(
            services=[LLMServiceConfig(name="s", base_url="http://x")],
            models=[LLMModelConfig(model="m", service="s")],
            default_model="nonexistent",
        )
        with pytest.raises(KeyError, match="Default model"):
            bad.get_model()

    def test_list_models_all(self, config):
        models = config.list_models()
        assert len(models) == 4

    def test_list_models_filter_by_service(self, config):
        models = config.list_models(service="anthropic")
        assert len(models) == 3
        assert all(m.service == "anthropic" for m in models)

    def test_list_models_no_match(self, config):
        assert config.list_models(service="nonexistent") == []

    def test_list_services(self, config):
        services = config.list_services()
        assert len(services) == 2
        names = {s.name for s in services}
        assert names == {"anthropic", "openai"}

    def test_get_service(self, config):
        s = config.get_service("anthropic")
        assert s.base_url == "https://api.anthropic.com"

    def test_get_service_missing(self, config):
        with pytest.raises(KeyError, match="Service"):
            config.get_service("nonexistent")

    def test_resolve_env_vars_in_services(self, config):
        resolved = config.resolve(
            environ={"ANTHROPIC_API_KEY": "sk-ant-123", "OPENAI_API_KEY": "sk-oai-456"}
        )
        assert resolved.services[0].api_key == "sk-ant-123"
        assert resolved.services[1].api_key == "sk-oai-456"

    def test_resolve_nested_list_items(self):
        """Resolve $ENV_VAR in models list (non-service fields are also covered)."""
        config = LLMConfig(
            services=[
                LLMServiceConfig(
                    name="s", base_url="http://x", api_key="$MY_KEY"
                )
            ],
            models=[LLMModelConfig(model="m", service="s")],
        )
        resolved = config.resolve(environ={"MY_KEY": "resolved-value"})
        assert resolved.services[0].api_key == "resolved-value"

    def test_llm_model_and_service_joined(self, config):
        result = config.get_model()
        assert isinstance(result, LLMModelAndService)
        assert result.model.model == "claude-sonnet-4-6"
        assert result.service.name == "anthropic"


class TestLoadConverter:
    def test_loads_function_from_module(self):
        converter = load_converter("json:dumps")
        import json
        assert converter is json.dumps

    def test_raises_on_missing_colon(self):
        with pytest.raises(ValueError, match="Invalid converter import path"):
            load_converter("no_colon_here")

    def test_raises_on_nonexistent_module(self):
        with pytest.raises(ImportError):
            load_converter("nonexistent.module:func")

    def test_raises_on_nonexistent_attr(self):
        with pytest.raises(AttributeError):
            load_converter("json:nonexistent_function")

    def test_raises_on_non_callable(self):
        with pytest.raises(TypeError, match="not callable"):
            load_converter("json:__version__")
