# configs

本目录存放 MOSS 系统的各种核心模块配置项.
基本都考虑用 `ghoshell_moss.contracts.configs` 的机制实现.

## Aurelius 记忆配置

`memory.yml` 是 Aurelius 的 Memento 策略文件：它决定上下文窗口、机械 commit 阈值和
反思旁路，而不是存放实际对话。实际记忆位于 `../ghosts/aurelius/memento/`。

- 当前 workspace 的文件：`.moss/configs/memory.yml`；
- 其它 workspace 的文件：`<workspace root>/configs/memory.yml`；
- 修改后需重启 Aurelius 才会生效；
- 字段含义和安全边界见 `memory.yml` 内注释及
  `Docs/MOSS-Ghost-Memory集成技术评审与实施方案.md`。

考虑会有的配置项:

- 音频输出配置
- 音频输入配置
- tts 配置
- asr 配置
- 模型配置

...
