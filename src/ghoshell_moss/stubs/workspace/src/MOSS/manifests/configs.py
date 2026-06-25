# MOSS Config manifest — 全局默认配置注册。
#
# 用 Config 实例定义默认值.
# 通过环境发现后, 会先尝试从本地文件中获取 [ws]/configs/[config_name].yml
# 如果不存在会创建文件.


from ghoshell_moss.contracts.llms import LLMConfig

# 模型的配置.
llm_config = LLMConfig()
