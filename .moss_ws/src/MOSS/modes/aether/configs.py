import os

from MOSS.manifests.configs import *  # noqa: F403

# aEther mode 的运行时默认值集中放在这里，避免把全双工语音 demo 的特殊
# 假设写死到 MOSS 主干模块。全部使用 setdefault：用户 shell/.env 中显式
# 配置的值优先级更高。

# listener wake-word 命中后，除 Mindflow interrupt signal 外，还允许通过
# AudioRuntimeTopic(device_name="interrupt") 直接清理 shell/TTS 缓冲。
os.environ.setdefault("MOSS_ENABLE_AUDIO_INTERRUPT_TOPIC", "1")

# aEther bringup 同时启动 vpio_capture/listener/web 时，Circus arbiter 偶发
# “already running” 竞争；仅在该 mode 下串行启动，默认 AppStore 仍保持并行。
os.environ.setdefault("MOSS_APPSTORE_BRINGUP_SERIAL", "1")

# OpenAI-compatible pydantic_ai 流式路径对多个 TextContent/history 兼容性较弱；
# aEther 语音场景先用单轮短上下文，后续如果换回兼容模型可取消这两个开关。
os.environ.setdefault("MOSS_ATOM_TEXT_PROMPT_COMPAT", "1")
os.environ.setdefault("MOSS_ATOM_DISABLE_HISTORY", "1")
os.environ.setdefault("MOSS_OPENAI_DISABLE_THINKING", "1")

# vpio_capture 输出 16kHz mono PCM；listener 默认仍跟随通用 capture 配置。
os.environ.setdefault("LISTENER_INPUT_SAMPLE_RATE", "16000")
os.environ.setdefault("LISTENER_GATE_DURING_TTS", "0")

# aEther 当前基线使用火山 ASR SAUC 优化双向流式端点；公共 ASR 配置仍保留
# 旧 URL 作为兼容默认，所以这里在 mode 内单独声明。
os.environ.setdefault(
    "VOLCENGINE_BM_ASR_URL",
    "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
)

# aEther 前端需要看到 ASR 服务端错误以做诊断/backoff，通用 ASR 默认只记录日志。
os.environ.setdefault("VOLCENGINE_BM_ASR_PROPAGATE_ERRORS", "1")
