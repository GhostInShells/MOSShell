# configs

本目录存放 Atom 系统的各种核心模块配置项.
基本都考虑用 `ghoshell_common.contracts.configs` 的机制实现. 

保存用 Yaml pretty dump (`ghoshell_common.helpers.yaml_pretty_dump`) 方便人类识别. 
考虑通过一个 ConfigsMode 或者 MetaMode 让配置的过程本身也 AI 化. 

考虑会有的配置项: 

- 音频输出配置
- 音频输入配置
- tts 配置
- asr 配置
- 