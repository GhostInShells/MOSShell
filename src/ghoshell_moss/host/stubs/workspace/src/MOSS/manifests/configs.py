from ghoshell_moss.contracts.configs import ConfigType


class TestConfig(ConfigType):
    foo: str = 'foo'
    bar: str = 'bar'

    @classmethod
    def conf_name(cls) -> str:
        return "test"


test_config = TestConfig()
