---
created: 2026-05-25
depends: []
description: Speech 体系治理：解耦 commands 权责泄漏，player 多后端轻量化，TTS 国际化多 provider， 自动容错降级，session
  logos stream 可选的跨进程流式。
milestone: null
priority: P2
status: completed
status_note: 2026-09-16 ClauseTopic 装线两侧落地 (说侧 moss_runtime / 听侧 controller.with_topic_service);
  收尾 D14 播放文本记账下沉 stream + say 表面/状态分离. 收口 completed.
title: Speech Governance — 解耦、多后端、容错降级
updated: '2026-09-23'
---

# Speech Governance

## Motivation

当前 MOSS 的 speech 体系有三个结构性问题，且彼此交织：

1. **权责泄漏**：`contracts/speech.py` 中的 `make_content_command_from_speech()` 和 `TTSSpeech.commands()` 直接生成 `Command` 对象。`CTMLShell._speech_context_manager()` (ctml_shell.py:185-203) 将 speech 以 command 形式注入 main channel。Speech 知道了 shell 的调度模型——这是反向依赖。

2. **Player 单点且老旧**：默认 audio player 只有 PyAudio（`depend_pyaudio()` in depends.py）。PyAudio 依赖 PortAudio C 库，安装经常失败，且 API 设计停留在 2000 年代。PulseAudio 实现只覆盖 Linux。没有一个跨平台、零系统依赖的轻量默认。

3. **TTS 单 provider + 国内锁定**：`TTSServiceProvider` 的 `use` 字段是 `Literal['volcengine_stream_tts_model']`，硬编码只有一个选项。`.env.example` 只提供火山引擎的凭证模板。国际用户无法使用。

4. **无容错降级**：如果 TTS API 挂了或 player 打不开，整个 speech 链路直接崩溃，没有 fallback 到 mock/文本输出的路径。

**时机**：zenoh fractal 改造已完成（`zenoh-fractal` feature），channel 可以通过 manifests 体系被动发现和集成。这为 speech channel module 化提供了基础设施——speech 的各种组合方式可以被 manifests 自动发现并集成到 shell，不需要 CTMLShell 硬编码 `_speech_context_manager`。

## Design Index

- 基准抽象: `src/ghoshell_moss/contracts/speech.py` (Speech, SpeechStream, StreamAudioPlayer, TTS, TTSBatch, TTSSpeech)
- 当前实现: `src/ghoshell_moss/core/speech/` (mock, stream_tts_speech, player/*, volcengine_tts/*, speech_module)
- Shell 耦合点: `src/ghoshell_moss/core/ctml/shell/ctml_shell.py:_speech_context_manager`
- TTS provider: `src/ghoshell_moss/host/providers/tts_service_provider.py`
- 依赖声明: `src/ghoshell_moss/depends.py`
- 环境配置: `src/ghoshell_moss/host/stubs/workspace/.env.example`
- Session stream: `src/ghoshell_moss/core/blueprint/session.py` (logos stream protocol)
- Channel 分形参考: `zenoh-fractal` feature (Hub/Provider 分离 + manifests 自动发现)
- pyproject.toml: `src/ghoshell_moss/pyproject.toml` (optional dependency groups)

## 实施进度

### Phase 2: Player 轻量化 (P1) — DONE (2026-05-26)

**完成内容**:

1. `miniaudio>=0.67` 加入核心依赖 (`pyproject.toml` dependencies)，零系统依赖，跨平台一致
2. 新增 `MiniAudioStreamPlayer` (`core/speech/player/miniaudio_player.py`)，实现 `BaseAudioStreamPlayer` 的三个抽象方法
3. `AudioPlayerProvider` 替代 `PyAudioPlayerProvider`，默认 backend 为 `miniaudio`，可通过 `AudioPlayerConfig.backend: "pyaudio"` 切换
4. PyAudio 仍可通过 `pip install ghoshell_moss[audio]` 安装，惰性导入，切换 backend 即可使用
5. 10 个单元测试覆盖：生命周期、格式转换、重采样、流式播放、clear、幂等性

**miniaudio 1.x 适配要点**:
- miniaudio 1.71 使用 generator 模式：`PlaybackDevice.start(gen)` 接受 callback generator
- 内部线程通过 `gen.send(frame_count)` 请求精确帧数，generator 必须返回恰好 `frame_count * channels * 2` 字节
- 实现内部 buffer `_buf` 机制：队列数据先入 buffer，按需切分 yield，多余留在 buffer 供下次请求
- 数据不足时用静音补齐，避免 underrun 噪声

**变更文件**:
| 文件 | 变更 |
|------|------|
| `pyproject.toml` | 核心依赖新增 `miniaudio>=0.67` |
| `core/speech/player/miniaudio_player.py` | **新文件** — MiniAudioStreamPlayer |
| `core/speech/player/__init__.py` | 导出 MiniAudioStreamPlayer |
| `core/speech/__init__.py` | `make_baseline_tts_speech()` 默认改用 MiniAudioStreamPlayer |
| `host/providers/audio_player_provider.py` | AudioPlayerProvider (miniaudio 默认 + pyaudio 可切换) |
| `host/stubs/workspace/src/MOSS/manifests/providers.py` | 引用 AudioPlayerProvider |
| `host/stubs/workspace/src/MOSS/manifests/configs.py` | 引用 AudioPlayerConfig |
| `tests/ghoshell_moss/speech/test_miniaudio_player.py` | **新文件** — 10 个单元测试 |

### Phase 1: 解耦 — DONE (2026-05-28)

**核心设计**：两层模型。

```
Speech (基础抽象)              TTSSpeech (高级扩展)
──────────────                ──────────────────
• 纯文本/音频输出              • 继承 Speech
• __content__ 内核命令         • say / set_voice 等命令
• Shell 自带                   • SpeechChannelModule 按需挂载
• MockSpeech 兜底              • 需要 TTS provider
```

**职责划分**:

```
Shell (_speech_context_manager)          SpeechChannelModule
────────────────────────────────        ─────────────────────
• 解析 speech → 注入容器                • register_content=False (默认)
• build_content_command() → __content__ • 仅 isinstance(speech, TTSSpeech)
• start / close 生命周期                  时注册 say
                                        • MockSpeech → 空命令集，无副作用
```

Shell 始终拥有 `__content__` 内核命令——无论 speech 是 MockSpeech 还是 TTSSpeech。`SpeechChannelModule` 负责发现 TTSSpeech 并挂载高级语音命令（say），没有 TTSSpeech 时静默为空。

**Bootstrap 顺序保证正确性**:
```
_ioc_context_manager       → container bootstrap
_speech_context_manager    → speech 注入容器 + 注册 __content__ + start
_runtime_context_manager   → module.on_startup() 从容器取 speech → 注册 say
```

**完成内容**:

1. 新建 `core/speech/speech_module.py`：
   - `build_content_command(speech) -> Command` — 纯函数，构建 `__content__` 内核命令，供 Shell 使用
   - `SpeechChannelModule(register_content=False)` — `ChannelModule`，`on_startup` 时从容器获取 Speech；仅当 `isinstance(speech, TTSSpeech)` 时注册 `say`；可选注册 `__content__`（供独立 speech channel 使用）
   - `_SpeechCommandFactory` — 内部工厂，`build_content_command()` + `build_say_command()` 两个 public 方法

2. 删除 `contracts/speech.py` 中的 `make_content_command_from_speech()` 和 `TTSSpeech.commands()`；TTSSpeech 退化为纯 ABC

3. `CTMLShell.__init__` 恢复 `speech` 参数：`self._speech: Speech = speech`

4. `_speech_context_manager`：
   - 解析 speech（参数 > 容器 > MockSpeech），注入容器
   - 调用 `build_content_command()` 注册 `__content__` 内核命令
   - 启停生命周期：`start()` → `yield` → `close()`

5. `StatefulChannel` 新增 `with_module()` 抽象方法（`states_channel.py`）

6. `speech_channel.py` 改用 `channel.with_module(SpeechChannelModule(register_content=True))`

7. `manifests/channels.py` 加入 `main.with_module(SpeechChannelModule())`，主机路径自动发现 TTSSpeech 时注册 say

**变更文件**:
| 文件 | 变更 |
|------|------|
| `contracts/speech.py` | 删除 `make_content_command_from_speech()` 和 `TTSSpeech.commands()`；TTSSpeech 退化为纯 ABC |
| `core/speech/speech_module.py` | **新文件** — `build_content_command` + `SpeechChannelModule` + `_SpeechCommandFactory` |
| `core/speech/__init__.py` | 导出 `SpeechChannelModule`, `build_content_command` |
| `core/ctml/shell/ctml_shell.py` | 恢复 `speech` 参数；`_speech_context_manager` 注入+注册content+启停 |
| `channels/speech_channel.py` | `inject_speech_commands` → `channel.with_module(SpeechChannelModule(register_content=True))` |
| `core/blueprint/states_channel.py` | `StatefulChannel` 新增 `with_module()` 抽象方法 |
| `host/providers/speech_service_provider.py` | `singleton()` 改为 `True` |
| `host/stubs/workspace/src/MOSS/manifests/channels.py` | 加入 `main.with_module(SpeechChannelModule())` |
| `tests/.../test_shell_speech.py` | 测试 `new_ctml_shell(speech=speech)` 简洁写法，无需 module |
| `tests/.../test_wait_primitive.py` | 同上 |
| `tests/.../test_elements.py` | `make_content_command_from_speech` → `build_content_command` |

### 已收口范围

完成的线: 解耦 (Phase 1/D1), Player 轻量 (Phase 2/D2), 默认空 speech + 测试无副作用 (Phase 5/D7),
docstring 示例 (D6), __content__ 可选化 (D10), 说侧可感知播放 (D11), cmd_task 交叉耦合清理。
未做的多 provider / 降级链不再承诺于本 feature (见 Implementation Plan out-of-scope 标注)。

### Phase 5: 默认空 speech + 播放器中断修复 + 测试无副作用 (P0) — IN PROGRESS (2026-05-29)

**动机**: MockSpeech 作为 CTMLShell 默认兜底有三个问题：
1. 内存泄漏 — `_outputs` 列表永久累积对话文本，`_streams` dict 永不清理
2. 定位错误 — MockSpeech 是为测试设计的（`outputted()` 断言、`typing_sleep`），不是为生产兜底
3. 测试副作用 — miniaudio player 测试真的出声

**完成内容**:

1. **NullSpeech** (`core/speech/null.py`) — 纯空操作 speech，零内存累积
   - `_NullSpeechStream`: 所有方法空操作，丢弃全部文本
   - `NullSpeech`: 无线程、无 buffer、无副作用
   - 替换 `ctml_shell.py` 和 `speech_module.py` 中的 `MockSpeech()` 兜底

2. **VirtualStreamPlayer** (`core/speech/player/virtual_player.py`) — 无音频输出的播放器
   - 继承 `BaseAudioStreamPlayer`，三个抽象方法空操作
   - 阻塞行为完全由基类时间估算驱动
   - 测试和降级兜底用

3. **MiniAudioStreamPlayer.clear() 中断修复**
   - 提取 `_make_generator()` / `_start_playback()` 辅助方法
   - `clear()` 重写：停止 playback 设备 → 清空 `_data_queue` + `_buf` → 重启设备 → 父类 clear
   - 之前 clear 只替换 `_audio_queue`，miniaudio generator 内 buffer 继续播放

4. **测试适配**
   - player 测试改用 `VirtualStreamPlayer`，不再出声
   - 新增 `test_clear_interrupts_playback` — 验证 clear 立即中断

**待完成**:

| # | 任务 | 影响文件 |
|---|------|----------|
| 5.1 | NullSpeech 打字机延时 | `core/speech/null.py` |
| 5.2 | Speech provider 配置化 delay | `host/providers/` |
| 5.3 | build_content_command 懒获取 Session → pub speech text | `core/speech/speech_module.py` |
| 5.4 | Session 抽象定义 SPEECH_KEY | `core/blueprint/session.py` |
| 5.5 | reachymini 验收 | MCP 端到端测试 |

**变更文件**:
| 文件 | 变更 |
|------|------|
| `core/speech/null.py` | **新文件** — NullSpeech + _NullSpeechStream |
| `core/speech/player/virtual_player.py` | **新文件** — VirtualStreamPlayer |
| `core/speech/player/__init__.py` | 导出 VirtualStreamPlayer |
| `core/speech/player/miniaudio_player.py` | `_make_generator()` + `_start_playback()` 提取；重写 `clear()` 中断播放设备 |
| `core/speech/__init__.py` | 导出 NullSpeech |
| `core/ctml/shell/ctml_shell.py` | `MockSpeech()` → `NullSpeech()` 兜底 |
| `core/speech/speech_module.py` | `MockSpeech()` → `NullSpeech()` 兜底 |
| `tests/.../test_miniaudio_player.py` | 改用 VirtualStreamPlayer + 新增 clear 中断测试 |

## Key Decisions

### D2: miniaudio 作为默认 Player (P1) — DONE

**决策**: 新增 `MiniAudioStreamPlayer(BaseAudioStreamPlayer)` 作为默认 player 实现。PyAudio 和 PulseAudio 降级为可选（通过 extras 安装）。

**实施**:
- `miniaudio>=0.67` 加入核心依赖，零系统依赖
- `AudioPlayerProvider` 支持 `backend: Literal["miniaudio", "pyaudio"]` 配置切换
- PyAudio 惰性导入，只在 backend="pyaudio" 且已安装 `ghoshell_moss[audio]` 时才加载
- miniaudio 1.x 使用 generator 模式，通过内部 buffer 实现帧精确 yield

**Why**:
- 零系统依赖，wheel 安装即用
- 跨平台一致
- 与 `BaseAudioStreamPlayer` 的三个抽象方法完全兼容
- PyAudio 历史上安装失败是 MOSS 试用者的第一道门槛

### D1: 两层模型 — Shell 拥有 content，Module 挂载 say (P0) — DONE (2026-05-28)

**决策**: 区分 `Speech` 和 `TTSSpeech` 两层。Shell 只依赖 `Speech` 抽象，拥有 `__content__` 内核命令。`SpeechChannelModule` 发现 `TTSSpeech` 后挂载 `say` 等高级命令。

**设计原则**:
- Shell（`_speech_context_manager`）：解析 speech → 注入容器 → `build_content_command()` 注册 `__content__` → 启停生命周期
- `__content__` 是内核命令，与 `wait`、`observe` 同级，Shell 始终拥有（MockSpeech 兜底）
- `SpeechChannelModule(register_content=False)`：仅在 `isinstance(speech, TTSSpeech)` 时注册 `say`；没有 TTSSpeech 时无副作用
- `register_content=True` 选项供独立 speech channel（如 `SpeechChannel`）使用
- `build_content_command(speech) -> Command` 是唯一的 content 构建入口

**Bootstrap 顺序保证**:
```
ioc_context_manager     → container.bootstrap()
speech_context_manager  → speech 注入容器 + 注册 __content__ + speech.start()
runtime_context_manager → module.on_startup() 从容器拿 speech → 注册 say
```

模块启动时 speech 已就绪、已在容器中。

**Why**: Shell 理解"能说话"是自身的基础能力，不需要 module 告诉它。但"怎么用 TTS 高级特性说话"是 TTSSpeech 层的知识。这种分离让 MockSpeech 和 TTSSpeech 各就其位：不传 speech → MockSpeech 也能说话；传了 TTSSpeech + module → 拥有完整语音能力。

**测试行为**:
| 场景 | `__content__` | `say` |
|------|:--:|:--:|
| `new_ctml_shell()` 不传 speech | MockSpeech 兜底 | 无 |
| `new_ctml_shell(speech=MockSpeech())` | 有 | 无 |
| `new_ctml_shell(speech=MockSpeech())` + `with_module(SpeechChannelModule())` | 有 | 无（MockSpeech 非 TTSSpeech） |
| `new_ctml_shell(speech=tts_speech)` + `with_module(SpeechChannelModule())` | 有 | 有 |

### D6: 强化 CTML docstring 中的 JSON 参数示例 (P1)

**决策**: 对涉及 `dict`/`list` 等复杂类型参数的命令，在 docstring 中显式提供 CTML 调用示例，展示 JSON 字符串的正确构造方式。不改变接口签名，不改变 CTML 解析规则。

**具体措施**:
- `say_doc()` 中加入 CTML 示例：`<say voice:dict=\"{'speed': 1.0, 'pitch': 'high'}\">你好</say>`
- 样本应覆盖：单层 dict、嵌套 dict、带默认值的省略写法
- 示例中展示 `:dict` 类型后缀的使用，利用 CTML parser 已有的 `AttrWithTypeSuffixParser` 机制
- 此模式作为约定推广到其它有 dict/list 参数的命令

**Why**: 改动最小，立即生效。模型对 docstring 中的示例有很强的跟随能力，好的示例可以显著降低出错率。长期看，如果某个 dict 参数频繁出错，再考虑拆分命令（方案 A）。

**非目标**: 不改 CTML 语法，不加新的 parser 约定。不强制所有命令都避免 dict 参数。

### D3: TTS provider 用 str + registry 替代 Literal (P1)

**决策**: `TTSServiceProvider.use` 改为 `str` 类型，支持多个 provider 名。provider 通过模块级注册表发现，而非 if/elif 硬编码。

```python
# 注册机制（每个 provider 模块自行注册）
TTS_PROVIDERS: dict[str, Callable[[IoCContainer, dict], TTS]] = {}

# TTSServiceProvider.factory() 改为
def factory(self, con):
    name = manager_conf.use
    if name not in TTS_PROVIDERS:
        raise LookupError(f"Unknown TTS: {name}. Available: {list(TTS_PROVIDERS)}")
    return TTS_PROVIDERS[name](con, manager_conf.provider_configs.get(name, {}))
```

**Why**: 新增 provider 不应修改 `TTSServiceProvider` 源码。每个 provider 在自己的模块中 `TTS_PROVIDERS["openai"] = factory_openai_tts`。

### D4: 降级链在 Speech 层实现 (P2)

**决策**: 新增 `FallbackSpeech` wrapper，接受 `[Speech, Speech, ...]` 优先级列表。启动时依次尝试 `speech.start()`，第一个成功者作为 active speech。当 active speech 失败时，自动切换到下一个。

```python
class FallbackSpeech(Speech):
    def __init__(self, *candidates: Speech):
        self._candidates = candidates
        self._active: Speech | None = None

    async def start(self):
        for candidate in self._candidates:
            try:
                await candidate.start()
                self._active = candidate
                return
            except Exception:
                logger.warning("Speech fallback: %s failed, trying next", candidate)
        self._active = MockSpeech()  # 最终兜底
        await self._active.start()
```

**Why**: 容错逻辑集中在 wrapper 中，不污染每个具体实现。MockSpeech 是永远可用的最终兜底。

### D5: 跨进程 speech 文本广播 (Out of Scope → 并入 D9)

> 历史摘要：D5 最初将跨进程 speech 列为 out of scope（音频流式协议过重），
> 2026-05-29 更新为只做文本广播（懒获取 Session 推送 speech 文本），最终定版见 D9。
> 完整反复轨迹见 `git log -- .ai_partners/features/workstreams/2026/05/speech-governance/FEATURE.md`。

### D7: NullSpeech 替代 MockSpeech 作为生产兜底 (P0)

**决策**: `NullSpeech` 作为 CTMLShell 的默认 speech。MockSpeech 退居纯测试角色。

**Why**:
- MockSpeech 的 `_outputs` 列表和 `_streams` dict 在生产环境中无限增长，内存泄漏
- MockSpeech 的 `outputted()`、`typing_sleep` 是为测试断言设计的，不应出现在生产路径
- NullSpeech 零分配、无线程、零副作用

### D8: NullSpeech 打字机延时 (P1)

**决策**: `NullSpeech(typing_delay: float = 0.0)` — `wait_played()` 按 `len(text) * typing_delay` sleep，模拟真实 TTS 的播放节奏。delay 通过 provider config 注入。

**Why**: 没有真实 speech 时，`__content__` 瞬时返回会导致后续命令无节奏地连续执行。打字机延时是零成本的自然 pacing。

### D9: build_content_command 集成 Session stream (P1)

**决策**: `__content__` 执行时懒获取 `Session`，若存在则 `session.pub_stream_delta(Session.SPEECH_KEY, text)` 推送 speech 文本。`SPEECH_KEY` 定义在 Session 抽象中，与 `LOGOS_KEY` 同级。

**Why**:
- 不改变 `build_content_command` 的纯函数语义（懒获取，拿不到就跳过）
- 遵循现有 logos stream 模式，不引入新协议
- 其他进程（GUI 字幕、机器人表情动画、日志系统）可订阅实时 speech 文本
- 这是 D5 "跨进程 speech" 的轻量落地 — 不做音频流式，只做文本广播

### D10: `__content__` 可选化，Shell 不默认注入 (P0) — 2026-09-04

**决策**: 推翻 D1 的"Shell 始终拥有 `__content__` 内核命令"。`__content__` 降级为
**可选机制**：Shell 不再无条件 `build_content_command()` + `add_command` 注入；唯一
入口是 `SpeechChannelModule(register_content=False)` 的显式 flag，默认不组装。
同时删除 `_feed_stream` 里的 `SpeechTopic` publisher 发布路径——说侧文本广播统一
走 D5/D9 的 Session stream delta。

**依据**（人类工程师 2026-09-04 重新同步）:

1. LLM 默认把 plain-text 输出成 markdown 语法，但 CTML 不假设 GUI 存在，于是
   plain-text 被当作语音；这需要额外 prompt 禁止 markdown 输出——默认语义与模型
   习惯相悖。
2. 模型要言说 CTML 本身时遇到自举困难（CDATA 里包裹 CDATA），`__content__` 输出
   plain text 仍是降级路径（对 markdown 支持差）。
3. 过去模型首 token 速度慢，3-5 token 承载语义有额外成本；现在模型输出 token
   速度显著变快，语义承载成本下降。
4. 模型在 coding agent 后训练中越来越倾向输出 markdown（视觉展示）而非"说话"，
   默认绑定语音的 `__content__` 在对抗这个重力。
5. Dolores Ghost Prototype 用 deepseek harness 做推理核 + dsh web 做 GUI，
   plain-text 有了独立的视觉展示方式，不再必须走语音。
6. 结合 `<|CTML|>` tag 可隔离多条 CTML 输出，与 markdown 输出不冲突。
7. `__content__` 仍然有用——GUI 交互与语音同步场景——所以**保留但不默认**。

**影响**:

- `ctml_shell.py` `_speech_context_manager` 删除 `build_content_command` 注入。
- `speech_module.py` 删除 `SpeechTopic` / `Publisher` 依赖及 `_feed_stream` 发布逻辑；
  `role` / `name` / `publishing` 参数随之删除。
- `SpeechChannelModule` 保留 `register_content` flag（默认 False），作为 `__content__`
  的唯一显式入口；`speech_channel.py` 的 `register_content=True` 是显式用例。
- `SpeechTopic` schema 本身待重设计（`topics/audio.py` 已有三个 todo：字段缺
  description、timestamp 冗余、audio_key 未实现），后续单独治理。
- 依赖"free text 默认变语音"的单测需显式注册 `__content__` 或改用 `say`。

### D11: 说侧可感知播放 — 真实样本回调 + 返回描述秒数 + 中断 STOPPED (P2) — 2026-09-04

**动机**: `say` / `__content__` 过去无返回值, 模型不知道"这句播了多久、说到哪被掐断"。
`buffered()` 是喂入文本总量, cancel 时含未合成/未播放部分, 不可靠; 真实播放文本必须靠音频
真正写入设备后的 `PlaybackSample` 对齐. 目标: 命令能感知真实播放进度并回报.

**顺带清理**: 删 `contracts/speech.py` 的 `cmd_task` / `as_command_task()` (0.0 时代残留,
contracts 反向依赖 core.concepts.command), mock / stream_tts_speech 同步清残留.

**机制分层**:

| 层 | 改动 |
|----|------|
| player | `StreamAudioPlayer.add(text)` → `PlaybackSample.text`, 样本自解释真实播放文本 |
| stream | `SpeechStream.on_sample(callback) -> disposer` 暴露真实样本回调 (基类默认 no-op);
  `TTSSpeechStream` 订阅 `player.observe` 过滤 `stream_id == self.id`; `say(samples)` 注册
  `samples.append`, `finally` 摘除 — discard 兜底在 say |
| command error | `CommandErrorCode.STOPPED = 301` (notify 档); `CommandUtil.reraise_stopped(message)` |
| speech_module | `say` / `__content__` 收集 samples, 见返回契约 |

**命令返回契约**:
- 正常结束且有真实播放样本 → 返回描述字符串 `"played {n:.1f}s"`
- 无播放样本 (MockSpeech / 尚未出声) → 返回 `None`, 不报误导性的 0.0s
- 被中断 (cancel) → 捕获最后真实播放片段, `reraise_stopped("played {n}s, stopped at ...{tail}")`

**Why**:
- `buffered()` 不可靠 → 真实播放文本必须由写出设备的 PlaybackSample 提供, 故 player 透传 text.
- `on_sample` 把对齐机制封装成回调, command 传 list 收样本即算秒数/取尾文本 — **command 内完成装线**,
  不触碰 player / CommandTask 内部.
- STOPPED(301) 落 notify 档 (is_notifiable 记录成可读 message, 不触发 observe/中断).
- 正常/中断统一描述文案, 模型据此判断"续说 / 承认被打断 / 跳过".

**变更文件**: `contracts/speech.py`, `core/speech/base_player.py`, `core/speech/stream_tts_speech.py`,
`core/speech/mock.py`, `core/speech/speech_module.py`, `core/concepts/errors.py`,
`core/blueprint/channel_builder.py`, 测试 (`test_player_playback_sample.py`, `test_mock.py`,
`test_elements.py`, `test_command_task.py`).

**后续未决**:
- `SpeechChannel.say` (speech_channel.py) 仍走 `speak()`, 未接 samples — 是否统一待定.
- 返回值对 channel / ghost 消费侧的观察 (dolores 装线 dogfood).

**后续修订 (2026-09-05)**: 尾帧文本改为 backend 降级提供 — volcengine/mimo 拿不到
text↔音频对齐时, 在最后一个 `PlaybackSample.text` 附上已喂文本尾部 (`speech_tail`,
中英混排 token 切分); `stopped_message` 直接消费 `samples[-1].text` 而非 `[-_TAIL_LEN:]`
切片段, 并补词数。`split_speech_tokens` / `speech_tail` 落在 `contracts/speech.py`。

### D12: SpeechTopic → ClauseTopic，additional 上移，AudioNucleus 退役 (P2) — 2026-09-13

**决策**: 本 workstream 声明的最后一个未收口项 (SpeechTopic schema 重设计) 于此落地。它不再
等 voice-input-state-machine 完成——(语音) 话语 topic 的命名与结构在本线先定义，装线 (谁
publish / subscribe) 另做。

**命名: `ClauseTopic`**（`topic_type = "clause"`），不是 `ConversationTopic` / `UtteranceTopic`：

- **和既有词汇同源**。`contracts/asr.py` 已把引擎的 `utterance (definite=true)` 重命名为
  `Clause`（`asr.py:43`、`:59`）。`utterance` 是引擎词/输入粒度，`Clause` 是项目词/分句粒度。
  用 `ClauseTopic` 是与既有 `Clause` 类型组合；用 `UtteranceTopic` 是把项目已抛弃的引擎词引回。
- **粒度 = 分句级，不是 turn 级**。一个 turn 可含多个 clause，只有 clause 在说话人之间干净交错。
  `TopicWindow[ClauseTopic]` 即交错对话轨迹。
- **`ConversationTopic` 是拿容器命名单元**——对话是整个窗口的性质，不是单个元素的性质。
- **双边**：听侧 ASR 定稿一个 clause、说侧 TTS 渲染一个 clause，都产出同一形状。

**字段**（只承载 clause 的语义内容）：

| 字段 | 说明 |
|------|------|
| `text` | clause 自身文本 |
| `speaker_id` / `speaker_name` | 身份维度——谁说的 |
| `role` | 功能维度——ghost / user。与身份**正交**：多个 speaker 可共享一个 role |
| `lang` | 语种 |

- **去 timestamp** → 走 `meta.created_at`（旧 todo）。
- **去 audio_key** → 协议 / 存储细节不进字段，挂 `additional`（旧 todo）。旧 docstring 的
  "每个属性都没有 description" todo 同时收口。

**additional 机制修正**: 原设计把扩展槽放在 `Topic(BaseModel, WithAdditional)`（信封级），
但 model → topic 转换时 model 自身的扩展无处安放。改为在 `TopicModel(BaseModel, ABC, WithAdditional)`
——`to_topic()` 把 additional 从 `data` 提到信封、`from_topic()` 还原，wire 上只有一个 addition
槽，与 publisher 级 additions（`Publisher.with_additions`）共用。`TopicMeta` 保持纯净，`Topic`
保持 WithAdditional。（曾误置于 `TopicMeta`，已改正。）

**退役**: `AudioNucleus` / `AudioSignal` / `AudioAction` 全删（`core/mindflow/audio_nucleus.py`、
`audio_signal.py`、`signals.py` 条目）。这是旧 alpha listener 线（`.moss_ws/apps/sensors/listener`）
的载体；该 app 已于 `15267d72` 删除，此后 `AudioSignal.speech_topic` 成为**只写不读**的死字段，
`AudioNucleus` 也从未注册进 `matrix/openbox/nuclei.py`（活路径是 `ListenerNucleus` / `ListenerSignal`）。
`nodes/sensors/listener` 一并删除。

**未决**: 装线——ClauseTopic 的 publish / subscribe 方。**不动**: `BufferNucleus`（多测试依赖，非
AudioNucleus 专属）。

**变更文件**: `topics/audio.py`, `topics/__init__.py`, `core/concepts/topic.py`, `signals.py`,
删除 `core/mindflow/audio_nucleus.py` / `audio_signal.py` / `nodes/sensors/listener/`。

### D13: 说侧 clause/segment 双回调 + 对齐算法 (P0) — 2026-09-14

**动机**: 装线——说侧 (ghost 说话) 也要产出分句级的 ClauseTopic。核心是"说侧真实播放情况"：
分句边界、字级时序、实际播到哪、是否被打断。经实测火山 bidirection TTS 协议，真值都在服务端，
当前 `tts.py` 全丢了 (`pass`)。

**实测结论（`moss audio speak` 打点 + 逐字喂实验验证，非猜）**:

| 事件 | event 值 | 粒度 | payload |
|------|---------|------|---------|
| `TTSSentenceStart` | 350 | 整个请求 | `{phonemes:[],text:"",words:[]}` |
| `TTSResponse`（音频包） | 352 | 音频片段 | 纯 int16 PCM，`flag=WithEvent`，`seq=0`，无 text |
| `TTSSentenceEnd` | 351 | 整个请求 | `{text:全文, words:[]}`（words 恒空） |
| **`TTSSubtitle`** | **364** | **每句（分句）** | `{text:该句, words:[{word,startTime,endTime,confidence}]}` |

- **subtitle 是分句级，不是音频片段级**：整段喂和逐字喂都出 N 句 N 个 subtitle（标点驱动分句）。
- 字级时间戳需 `enable_subtitle=true`，否则 `words:[]`。时间单位是**秒**（float，服务端原值），
  听侧 `Clause` 用**毫秒**——两侧单位不同，对齐时换算。
- 音频包无 text、无 seq 尾标记——"播到哪"靠 subtitle 的 `words[].end_time` + 真实播放时长对齐。
- **TTS 合成快于播放**：subtitle（分句）先于其音频播放到位，所以 clause 回调不能由"TTS 返回"触发。

**对齐模型（对称听侧 `stream > segment > clause`）**:

- 听侧 `stream = n*segment`，说侧 `stream = 1*segment`，两边 `segment = n*clause`。
- `segment` 是**音频存储单位**（音频 + 完整 text + clauses 列表），`clause` 是一句。
- 说侧 `segment = 1 say`，其 clause 边界由服务端标点分句给好（不自己做文本分句）。

**两个回调（对齐听侧）**:

| 回调 | 载荷 | 触发时机 | 对齐听侧 |
|------|------|---------|---------|
| `on_clause` | `SpeechClause`（文本） | **真实播放追到该句边界**（非 TTS 返回） | `RecognitionEvent.clause` |
| `on_segment` | `SpeechSegment`（文本+音频） | segment 播放结束一次 | `RecognitionSegment` |

**对齐算法（关键 trick）**: `on_clause` 不是 TTS 返回 clause 就触发，而是 **playsample 对齐 clause 生效**——
TTS 返回 clause → 更新本地数组（含 `words[].end_time`）；每个 `PlaybackSample`（真实 `duration`）累加
`played_duration`，当 `played_duration >= clause.words[-1].end_time` 时发 `on_clause`，游标前移。segment
结束时发 `on_segment`（`interrupted = 游标 < len(clauses)`）。

**数据结构（`contracts/speech.py`，去 TTS 前缀、去 Subtitle）**:

- `Word`（word/start_time/end_time/confidence，秒，JSON camelCase 别名）
- `SpeechClause`（text/words/timestamp）—— 一句
- `SpeechSegment`（segment_id/timestamp/text/clauses/**audio**(int16 PCM 供存文件回放)/sample_rate/channels/interrupted）—— 音频存储单位

**契约最小化**: `Speech.on_clause` + `Speech.on_segment` 默认 no-op；`TTSBatch.clauses() -> []` 默认空；
融合逻辑在 `BaseTTSSpeech`/`TTSSpeechStream` 惰性计算。`NullSpeech`/`MockSpeech` 不受累。

**变更文件**: `contracts/speech.py`, `core/speech/stream_tts_speech.py`,
`host/speech/volcengine_tts/config.py` (enable_subtitle + `TTSSubtitle=364`),
`host/speech/volcengine_tts/tts.py`（解析 subtitle 事件 → `SpeechClause` 累积到 batch）。

**未决**: cancel 语义——被打断时 receive task 被 cancel，在途 subtitle 拿不到，`interrupted` 与 clause
在 cancel 路径不完整。要"半句也成 clause"需改 cancel 时序（等流自然结束拿尾包），本次未做（tts 边界抠得细，
先不动）。`phonemes` 音素级未返回，字级已够。`on_clause` 在 worker 线程回调（与 `on_sample` 一致），消费方
需自行 marshal 到 event loop。

### D14: 播放文本记账下沉到 stream + say 表面/状态分离 (P2) — 2026-09-16

**动机（收尾两件）**: (1) 打断时报"说到哪"要真的有内容; (2) say 的命令表面混着此刻的
voice/tone 状态, 状态一变整块接口就重发.

**实测（`VirtualStreamPlayer` + 假后端复现，非推断）**: D11 的尾帧文本只附在**最后一个**
audio frame 上, 而打断时 `samples[-1]` 是**最后播出的那一帧**（末尾帧根本还没播到, text 为空）。
于是 `stopped at ...{tail}` 在实践中几乎不触发, 反而落到 "stopped before audible output"——
明明已经出声。正常播完则完全不消费 tail。附带 bug: `speech_tail` 用 `" ".join(tokens)` 拼回
文本, 中文逐字成 token → "已 经 听 到 了 ."。

**决策 1 — 播放记账在 stream, 不在 sample**:

| 层 | 改动 |
|----|------|
| `SpeechStream` | 新增 `played_text() -> str`, 默认空串: **只报对齐结果, 不做降级**——空串即"这个实现给不出", 由调用方决定近似 |
| `TTSSpeechStream` | 用 D13 已有的对齐游标 `_clause_cursor` 拼 `clauses[:cursor].text`: 只有**整句播完**的 clause 计入（保守, 半句不报）; batch 无字幕能力 → 空串 |
| `stopped_message` | `stopped_message(samples, played_text)`: 对齐优先 → 空则回落 `samples[-1].text`（无字幕后端的尾帧提示）→ 都没有只报秒数, 不再断言"没有出声"; 词数改为**已播出文本**的 token 数（原来数的是 ≤6 token 的 tail 本身, 无意义） |
| `speech_tail` | 按 token 跨度**切原串**, 不再 join tokens（拼接会插入原文没有的空格） |

**决策 2 — 命令表面写契约, 状态走 named notice**:

- `say` 的 `doc` 从 callable 改成常量字符串: schema（voice_schema）/ tone 目录 / 参数语义都是
  启动期静态内容。命令因此不再是 dynamic command, meta 不再逐轮重生成。
- `SpeechChannelModule.get_named_notices()` 出两个片段: `voice`（此刻默认 voice 的 JSON）、
  `tone`（此刻音色）。非 TTS speech (NullSpeech) 不产出。渲染层按 name delta 比对, 只有真的
  变了才重发那一片——换音色不再触发整块命令接口重发。
- 边界: **契约留在表面**（schema / tone 目录让模型不依赖 notice 就能看懂参数）, **只有此刻的状态进 notice**。

**变更文件**: `contracts/speech.py`, `core/speech/stream_tts_speech.py`,
`core/speech/speech_module.py`, `channels/speech_channel.py`,
测试 `tests/.../host/speech/test_tts_stream_play.py`（played_text 只计已播 clause）、
`tests/ghoshell_moss/channels/test_speech_module.py`（状态进 notice, 表面不动）。

**未决**:
- 正常播完是否也带 tail: 目前只报 `played N.Ns`（模型自己说了什么它知道, tail 只在被打断时有用）。
- backend 尾帧 hack 对 volcengine 已冗余（它有 clause 对齐）, 是否删除待定; mimo 无字幕, 仍要靠它降级。
- tail 精度受 D13 的 cancel 未决影响: 在途 subtitle 拿不到, 半句不成 clause, 报的是"最后播完的整句"。

### D15: mute 命令 — 旁听模式的安全闸 (P1) — 2026-09-17

**动机**: 旁听模式——会议 / 路演时人类说"等下你先别说话, 我要你说时再说", 之后人类对别人
说的话照样喂给 ghost, ghost 仍可思考与行动, 但**零误说风险**。mute 是模型自控的 toggle, 对抗
实机里"它会忘"的本能。

**决策**:

| 层 | 改动 |
|----|------|
| 真相 | `mute(on: bool)` 命令是唯一写入点, flag 在 `SpeechChannelModule` |
| 状态 | 走 named notice `mute` 片段, **off/on 恒非空**（空串会被渲染层当静默, 模型残留旧记忆）; 不碰命令 surface |
| 闸门 | mute 时 `say` / `__content__` 在 partial 就 raise `NOT_AVAILABLE(403)`, 不建 TTS batch、不 start_synthesis、不出声 |
| 语义 | 用 `NOT_AVAILABLE` 而非新码——"这个命令此刻不可用"; message 写清可恢复: `mute(on=false)` |

**为什么是 403 而不是软闸**: `NOT_AVAILABLE`(403) 落在 critical 档 (`is_critical` = code ≥ 400),
会 cancel in-flight batch + stop interpreter。这是**故意的硬闸**——旁听模式下漏嘴就是违规, 要当场
打断、迫使模型重新定向, 而不是让其余命令继续跑完（可能还带着那次误说的上下文）。代价是同一批
里已排队的合法行动也会被冲掉, 接受——漏嘴本身就是该重新想一遍的信号。

**为什么不用 available=false**: 它改的是命令 meta 的 availability → 整个 channel facade 重绘,
每个 mute 切换都要重发全量界面, 不值。命令保持 available/可见, 只在运行时拒绝。

**变更文件**: `core/speech/speech_module.py`,
测试 `tests/ghoshell_moss/channels/test_speech_module.py`（toggle 进 notice + 静音 say 拒 403）。

**未决**: mute 是否该随 session 结束自动复位（现在跨 turn 持久, 靠 notice 提醒）; 是否要
"N 轮后自动解除"的保险。

### D16: ChannelModule 补 available 轴 — `is_available()` (P2) — 2026-09-23

**动机**: 裸起一个绑了 `SpeechChannelModule` 的主 channel（无 speech、容器也无 `Speech`）时,
命令侧是干净的（say/mute 都不挂）, 但表面不干净:

| 面 | 实测 |
|----|------|
| `meta.modules` | `['speech']` — 没装线却露名字 |
| `named_notices` | `{"mute": "off"}` — 报了状态, 却没有对应命令 |
| `meta.dynamic` | `True` — 空 module 让通道永不静态缓存（见下文: 这行有意不修） |
根因不是"少个判断", 是**同一条件写在两处**: 命令闸在 `on_startup`（`speech_module.py:252`）,
状态闸在 `get_named_notices`（`speech_module.py:228`）。D14 / D15 两轮优化各补了一侧,
改这处漏那处 —— "这个模块此刻成不成立"隐式躺在其中一边, 抽象面上看不见。

**决策**: `ChannelModule` Protocol 加 sync `is_available()`, 与 `ChannelState.is_available()`
同名同形。**同步是关键**: `is_dynamic()` 是 sync 结构刷新路径, 直接读它即可, 不需要新造
async 评估 + 缓存那一套。runtime 以它为唯一闸门过滤 `meta.modules` / `own_commands` /
`get_command` / notice / named_notices / context messages —— 表面下架必须等于调用路径下架,
否则模型看不见却调得到, 比看得见调不到更坏。

**生命周期不在这条轴上**: `on_startup` / `on_close` / `on_refresh_meta` 仍遍历**全部** module,
对齐 state（`_get_current_state()` 也只是把不可用 state 从 meta 剔掉, 不阻止其生命周期）。
这是硬约束而非风格: `_speech` 是 `on_startup` 从 IoC resolve 出来的, 若闸住 startup,
`_speech` 永远是 None → 永远不可用, 死锁。

speech 侧谓词一句 `_speech is not None and _speech.is_running()`, 作为唯一真相源,
`on_startup` 与 `get_named_notices` 都**调用**它而不再各自重抄条件。收益: 可用性成了持续谓词
而非启动时的一次性分叉 —— 语音中途挂掉, say/mute 连同 notice 自动下架, 回来则恢复。

**`meta.dynamic` 上表那行不修**: 它仍按注册的 module 数判定。改成按 `is_available()` 会让
模块下架时整个 meta 被判为静态而进 `_static_meta_cache`, 之后恢复也读不回来（实现时先写成
按 available 判定, 恢复路径当场测挂）。"注册即 dynamic" 正是让可用性保持活的代价。

**与 D15 的边界**: D15 否决的是用**命令 meta 的 available** 做 mute 闸（每次切换重发全量界面）。
这里是 **module 级**可用轴, 谓词是"speech 是否在跑"这类稀疏事件, 翻转时界面本就该变。

**否决 `bootstrap(container) -> Self | None` 装线分叉**: `ChannelState` ABC 上已有
`bootstrap(container) -> None`（`concepts/channel.py:367`）, 而 `states_channel.py:53` 明说
PyChannelBuilder 与任意 ChannelState 自动满足 module Protocol —— `None = 不绑` 会把
state-as-module 用**静默丢掉**（不报错, 能力凭空消失）; 救它要改 ABC 返回契约 + 所有 state 实现,
远不止 py_channel。且一次性分叉表达不了上面那种"中途挂掉自动下架"。

**变更文件**: `core/blueprint/states_channel.py`（Protocol + 默认体）、
`core/py_channel.py`（`_available_modules()` + 7 个消费点）、`core/speech/speech_module.py`、
测试 `tests/ghoshell_moss/channels/test_speech_module.py` 与
`tests/ghoshell_moss/default/core/channels/test_state_channel.py`。

**归属**: 契约是内核级（ChannelModule Protocol）, 但内核迭代通常不单开 feature —— 除 mindflow
那类巨型重构外都是如此。故按触发场景归此, 契约沉淀在 Protocol docstring 上, 决策轨迹靠 `git log`。

## Implementation Plan

### Phase 1: 解耦 (P0) — ✅ DONE (2026-05-27)

| # | 任务 | 影响文件 | 状态 |
|---|------|----------|------|
| 1.1 | 删除 `make_content_command_from_speech()` 和 `TTSSpeech.commands()` | `contracts/speech.py` | ✅ |
| 1.2 | 创建 `speech_module.py`（build_content_command + SpeechChannelModule） | `core/speech/speech_module.py` | ✅ |
| 1.3 | `CTMLShell.__init__` 恢复 `speech` 参数 | `ctml_shell.py` | ✅ |
| 1.4 | `_speech_context_manager` 注入 + 注册 __content__ + 启停 | `ctml_shell.py` | ✅ |
| 1.5 | `new_ctml_shell()` 恢复 `speech` 参数 | `ctml_shell.py` | ✅ |
| 1.6 | `StatefulChannel` 新增 `with_module()` 抽象 | `states_channel.py` | ✅ |
| 1.7 | `speech_channel.py` 改用 `channel.with_module(SpeechChannelModule(register_content=True))` | `channels/speech_channel.py` | ✅ |
| 1.8 | `manifests/channels.py` 加入 `main.with_module(SpeechChannelModule())` | `manifests/channels.py` | ✅ |
| 1.9 | `singleton()` 改为 `True` | `speech_service_provider.py` | ✅ |
| 1.10 | 测试适配（简洁写法，无需 module） | `test_shell_speech.py`, `test_wait_primitive.py`, `test_elements.py` | ✅ |

### Phase 2: Player 轻量化 (P1) — ✅ DONE (2026-05-26)

| # | 任务 | 影响文件 | 状态 |
|---|------|----------|------|
| 2.1 | 新增 `MiniAudioStreamPlayer` 实现 | `core/speech/player/miniaudio_player.py` | ✅ |
| 2.2 | miniaudio 加入核心依赖 | `pyproject.toml` | ✅ |
| 2.3 | `AudioPlayerProvider` 替代 `PyAudioPlayerProvider`，backend 可切换 | `host/providers/audio_player_provider.py` | ✅ |
| 2.4 | PyAudio 改惰性导入，按需加载 | `audio_player_provider.py` | ✅ |
| 2.5 | 更新 workspace manifests/stubs | stubs + .moss_ws | ✅ |
| 2.6 | 10 个单元测试 | `tests/ghoshell_moss/speech/test_miniaudio_player.py` | ✅ |

### Phase 3: TTS 多 provider (P1) — OUT OF SCOPE

现状: `TTSServiceProvider.use` 已支持 `volcengine_stream_tts_model` / `mimo_tts` 配置切换;
registry 抽象 / OpenAI / edge-tts 等扩展不在本 feature 迭代, 不再承诺.

| # | 任务 | 影响文件 |
|---|------|----------|
| 3.1 | `TTSServiceProvider.use` 从 Literal 改为 str | `host/providers/tts_service_provider.py` |
| 3.2 | 实现 TTS provider registry 机制 | `core/speech/tts_registry.py` |
| 3.3 | VolcengineTTS 迁移到 registry 模式 | `core/speech/volcengine_tts/` |
| 3.4 | 新增 OpenAI TTS provider | `core/speech/openai_tts/` |
| 3.5 | 新增 edge-tts provider (免费离线兜底) | `core/speech/edge_tts/` |
| 3.6 | `.env.example` 增加多 provider 凭证模板 | `host/stubs/workspace/.env.example` |
| 3.7 | `depends.py` + `pyproject.toml` 各 provider 可选依赖声明 | |

### Phase 5: 默认空 speech + 播放器中断修复 + Session 集成 (P0)

| # | 任务 | 影响文件 | 状态 |
|---|------|----------|------|
| 5.1 | NullSpeech + _NullSpeechStream 实现 | `core/speech/null.py` | ✅ |
| 5.2 | VirtualStreamPlayer 无副作用测试播放器 | `core/speech/player/virtual_player.py` | ✅ |
| 5.3 | MiniAudio clear() 中断修复 | `core/speech/player/miniaudio_player.py` | ✅ |
| 5.4 | MockSpeech → NullSpeech 生产路径替换 | `ctml_shell.py`, `speech_module.py` | ✅ |
| 5.5 | 测试改用 VirtualStreamPlayer + 中断测试 | `tests/.../test_miniaudio_player.py` | ✅ |
| 5.6 | NullSpeech 打字机延时 | `core/speech/null.py` | out-of-scope (价值低; Mock 已有 typing_sleep) |
| 5.7 | Speech provider 配置化 delay | `host/providers/` | out-of-scope |
| 5.8 | build_content_command 懒获取 Session → pub speech text | `core/speech/speech_module.py` | 废弃 (D9 广播方向随 D10/D11 转向) |
| 5.9 | Session 抽象定义 SPEECH_KEY | `core/blueprint/session.py` | 废弃 |

### Phase 4: 容错降级 (P2) — 未实现, 思路被现状替代

现实: speech 是 player + tts 的组装物; 无 `Speech` 时 shell 层兜底 `NullSpeech` (D7 已做);
provider 层 `force_fetch` 真组件, 无运行时 mock/null 降级; D4 `FallbackSpeech` wrapper 思路未采用.

| # | 任务 | 影响文件 |
|---|------|----------|
| 4.1 | 实现 `FallbackSpeech` wrapper | `core/speech/fallback.py` |
| 4.2 | `TTSServiceProvider` 支持 provider 优先级列表 | `host/providers/tts_service_provider.py` |
| 4.3 | Speech 启动时 health check 机制 | `core/speech/` |

### Phase 5 (Out of Scope, Future)

- Session logos stream 跨进程 speech
- Speech 作为独立 app cell 运行
- 音频格式协商（TTS 输出格式 × Player 接受格式的自动转换矩阵）

## Blast Radius Summary

| 改动 | 影响范围 |
|------|----------|
| PyAudio → miniaudio 默认 | ✅ 已实施。miniaudio 核心依赖，PyAudio audio extras 可选。`AudioPlayerConfig.backend` 可切换 |
| 删除 `make_content_command_from_speech()` | `test_elements.py`（改用 `build_content_command`） |
| 删除 `TTSSpeech.commands()` | 调用方改用 `SpeechChannelModule` 或 `build_content_command` |
| `CTMLShell._speech_context_manager` 重构 | 核心启动路径，需要全量回归 |
| 新增 `MiniAudioStreamPlayer` | ✅ 纯新增，不影响现有 player |
| 新增 `speech_module.py` | 纯新增 |
| `StatefulChannel.with_module()` | 纯新增抽象方法 |
| MockSpeech → NullSpeech 兜底 | `ctml_shell.py`, `speech_module.py` — 生产路径无内存泄漏 |
| `MiniAudioStreamPlayer.clear()` 重写 | 中断链路核心修复，`_playback.stop()` 立即掐断音频 |
| 新增 `VirtualStreamPlayer` | 纯新增，测试和降级用 |
| 新增 `NullSpeech` | 纯新增，零开销默认兜底 |
| `TTSServiceProvider.use` 类型变更 | 配置文件格式小幅调整（Phase 3） |
| 新增 TTS providers | 纯新增（Phase 3） |
| `FallbackSpeech` | 纯新增（Phase 4） |
| `Session.SPEECH_KEY` + build_content_command pub | 纯新增（Phase 5 pending） |

**不动**: `Speech`, `SpeechStream`, `StreamAudioPlayer`, `TTS`, `TTSBatch` 抽象。`MockSpeech`（保留给测试）。`BaseAudioStreamPlayer`（VirtualStreamPlayer 复用其时间估算逻辑）。

## Test Plan

### T1: 单元测试
- ✅ `VirtualStreamPlayer` 生命周期 (start/stop/close)
- ✅ `VirtualStreamPlayer` add + wait_play_done 阻塞正确
- ✅ `VirtualStreamPlayer` 格式转换 (PCM_S16LE, PCM_F32LE)
- ✅ `VirtualStreamPlayer` 重采样
- ✅ `VirtualStreamPlayer` 流式多次 add
- ✅ `VirtualStreamPlayer` clear 立即中断 + is_playing 状态
- ✅ `VirtualStreamPlayer` estimated_end_time 单调递增
- ✅ `VirtualStreamPlayer` 幂等 start / close 后 add
- ✅ `MiniAudioStreamPlayer` clear 中断 — playback.stop() 立即掐断音频（实际出声，但时长极短）
- `NullSpeech` 打字机延时：wait_played 按文本长度 sleep（Phase 5 pending）
- `FallbackSpeech` 降级链：第1个成功 → 不尝试第2个；第1个失败 → 自动切换到第2个；全部失败 → 兜底 NullSpeech
- TTS provider registry 注册/查找/异常
- Speech channel module commands 注册正确性

### T2: Shell 集成回归
- `ctml_shell_test()` 端到端：CTML 文本 → speech 输出
- MockSpeech 作为默认 speech 时 shell 启动正常
- TTSSpeech + miniaudio player 组合启动正常
- Speech clear 行为不变

### T3: 多 provider 切换
- VolcengineTTS provider 正常启停
- OpenAI TTS provider 正常启停
- edge-tts provider 正常启停
- 配置切换 provider 后重启 shell 正常

### T4: 降级行为
- TTS API 不可用时自动降级到下一级
- Player 不可用时自动降级到 MockSpeech
- 降级过程不抛异常，logger 中有 warning

## Related Features
- `zenoh-fractal` — Channel 分形改造，为本 feature 的 speech channel module 化提供基础设施
- `cell-discovery-refactor` — Cell 发现重构，影响 app cell 的启动/发现模式