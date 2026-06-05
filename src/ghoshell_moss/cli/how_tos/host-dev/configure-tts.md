---
title: 配置 TTS 音色与语音参数
description: 如何在 MOSS workspace 中配置 TTS（文字转语音），包括切换默认音色、调整语速/音量/情绪、自定义 Speaker。面向需要定制语音输出的开发者和 AI 协作者。
---

# 配置 TTS 音色与语音参数

## 背景

MOSS 的 TTS 由 `TTSServiceProvider` 提供，通过 `TTSManagerConfig` 配置。配置以
YAML 文件形式持久化在 workspace 的 `configs/tts_factory.yml` 中，启动时自动加载。

当前唯一可用的 TTS 后端是火山引擎流式 TTS（`volcengine_stream_tts_model`）。

查看配置模型：

```bash
moss codex get-interface ghoshell_moss.host.providers.tts_service_provider:TTSManagerConfig
moss codex get-interface ghoshell_moss.host.speech.volcengine_tts.tts:VolcengineTTSConf
moss codex get-interface ghoshell_moss.host.speech.volcengine_tts.tts:SpeakerConf
```

查看当前生效的配置：

```bash
moss manifests configs           # 列出所有配置项
cat .moss_ws/configs/tts_factory.yml
```

## 步骤

### 1. 切换默认音色

编辑 `.moss_ws/configs/tts_factory.yml`，修改 `default_speaker` 为 speakers 字典中
已有的音色名称：

```yaml
volcengine_stream_tts_model_config:
  default_speaker: 可爱女生
```

重启 MOSS 后生效。

可用音色一览：

| 名称 | tone ID | 语言 | 适用场景 |
|------|---------|------|----------|
| 大壹 | zh_male_dayi_saturn_bigtts | 中文 | 视频配音 |
| 黑猫侦探社咪仔 | zh_female_mizai_saturn_bigtts | 中文 | 视频配音 |
| 鸡汤女 | zh_female_jitangnv_saturn_bigtts | 中文 | 视频配音 |
| 魅力女友 | zh_female_meilinvyou_saturn_bigtts | 中文 | 视频配音 |
| 流畅女声 | zh_female_santongyongns_saturn_bigtts | 中文 | 视频配音 |
| 儒雅逸辰 | zh_male_ruyayichen_saturn_bigtts | 中文 | 角色扮演 |
| 可爱女生 | saturn_zh_female_keainvsheng_tob | 中文 | 角色扮演 |
| 调皮公主 | saturn_zh_female_tiaopigongzhu_tob | 中文 | 角色扮演 |
| 爽朗少年 | saturn_zh_male_shuanglangshaonian_tob | 中文 | 角色扮演 |
| 天才同桌 | saturn_zh_male_tiancaitongzhuo_tob | 中文 | 角色扮演 |
| 知性灿灿 | saturn_zh_female_cancan_tob | 中文 | 角色扮演 |

### 2. 调整语速、音量、情绪

在 Speaker 的 `voice` 字段中配置：

```yaml
volcengine_stream_tts_model_config:
  speakers:
    可爱女生:
      tone: saturn_zh_female_keainvsheng_tob
      description: 'language: (中文), support english: False, use case: 角色扮演'
      voice:
        speech_rate: 10     # -50~100，正数加速，负数减速，0 正常
        loudness_rate: 5    # -50~100，正数增大，负数减小，0 正常
        emotion: happy      # 情绪，可选值见下方
```

`speech_rate` 和 `loudness_rate` 可选，不设置时使用默认值 0。

支持的中文情绪（`emotion`）：

`neutral`（默认）, `happy`, `sad`, `angry`, `surprised`, `fear`, `hate`,
`excited`, `coldness`, `depressed`, `lovey-dovey`, `shy`, `comfort`,
`tension`, `tender`, `storytelling`, `radio`, `magnetic`, `advertising`,
`vocal-fry`, `ASMR`, `news`, `entertainment`, `dialect`

### 3. 调整连接参数

```yaml
volcengine_stream_tts_model_config:
  disconnect_on_idle: 600     # 闲置多少秒后断开连接，默认 300
  disable_markdown_filter: true  # 是否禁用 markdown 过滤
  sample_rate: 44100          # 采样率：8000/16000/22050/24000/32000/44100/48000
```

### 4. 验证配置

```bash
# 启动 MOSS runtime 并检查 TTS 输出音色
moss-run-ghost <ghost_name>
```

让 Ghost 说一句话，确认音色与配置一致。

## 配置文件结构

完整结构（展示关键字段）：

```yaml
use: volcengine_stream_tts_model
volcengine_stream_tts_model_config:
  app_key: $VOLCENGINE_STREAM_TTS_APP
  access_token: $VOLCENGINE_STREAM_TTS_ACCESS_TOKEN
  resource_id: seed-tts-2.0
  sample_rate: 44100
  audio_format: pcm
  disconnect_on_idle: 300
  disable_markdown_filter: true
  url: wss://openspeech.bytedance.com/api/v3/tts/bidirection
  speakers:
    <音色名称>:
      tone: <tone_id>
      description: '<描述>'
      voice:
        speech_rate: 0
        loudness_rate: 0
        emotion: neutral
  default_speaker: <音色名称>
```

- `$VOLCENGINE_STREAM_TTS_APP` / `$VOLCENGINE_STREAM_TTS_ACCESS_TOKEN` 是环境变量引用，启动时自动解析。在 `.moss_ws/.env` 中配置实际值
- `speakers` 可自由增减，每个 speaker 的 key 是自定义名称，`tone` 必须是火山引擎支持的 tone ID
- `default_speaker` 必须是 `speakers` 中存在的 key

## 常见问题

### 问题：修改了 default_speaker 但不生效

确认 YAML 中的 `default_speaker` 值与 `speakers` 中的某个 key 完全一致（包括中文
字符和大小写）。如果 key 不存在，会 fallback 到 `SpeakerConf` 的硬编码默认值
（`知性灿灿`）。

可以用以下命令验证配置是否正确加载：

```bash
python -c "
from ghoshell_moss.host.providers.tts_service_provider import TTSManagerConfig
from ghoshell_moss.contracts.configs import YamlConfigStore
from ghoshell_moss.contracts.workspace import LocalWorkspace
from ghoshell_moss.core.blueprint.environment import Environment

env = Environment.discover()
env.bootstrap()
ws = LocalWorkspace(env.workspace_path)
store = YamlConfigStore(ws.configs())
conf = store.get_or_create(TTSManagerConfig())
tts = conf.volcengine_stream_tts_model_config
print(f'default_speaker: {tts.default_speaker}')
print(f'available: {list(tts.speakers.keys())}')
print(f'tone: {tts.default_speaker_conf().tone}')
"
```

### 问题：想用不在列表里的音色

在 `speakers` 字典中添加新条目，`tone` 填写火山引擎支持的 tone ID。如果使用声音
复刻，额外设置 `resource_id`。

### 问题：环境变量未设置导致连接失败

在 `.moss_ws/.env` 中确认以下变量已配置：

```
VOLCENGINE_STREAM_TTS_APP=<your_app_key>
VOLCENGINE_STREAM_TTS_ACCESS_TOKEN=<your_access_token>
```

## 文档目标

读者按照本文档操作，应该能够：
1. 在 `.moss_ws/configs/tts_factory.yml` 中切换 `default_speaker`
2. 调整 Speaker 的语速、音量、情绪参数
3. 理解配置文件的结构和可用字段
4. 通过命令行验证 TTS 配置是否正确加载
