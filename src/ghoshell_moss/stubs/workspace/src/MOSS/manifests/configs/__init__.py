# Config manifest — global default configuration declarations.
#
# Define ConfigType instances with default values.  Matrix scans via
# isinstance(obj, ConfigType), then registers in ConfigStore via get_or_create().
#
# Mode extends by: from MOSS.manifests.configs import *
#
# --
# Config 清单 — 全局默认配置声明。
# 用 ConfigType 实例定义默认值，Matrix 扫描自动发现并注册到 ConfigStore。

from ghoshell_moss.contracts.llms import LLMConfig

# LLM provider configuration
llm_config = LLMConfig()
