---
created: 2026-05-25
depends: []
description: Speech 体系治理：解耦 commands 权责泄漏，player 多后端轻量化，TTS 国际化多 provider， 自动容错降级，session
  logos stream 可选的跨进程流式。
milestone: null
priority: P2
status: in-progress
status_note: 重开承载 __content__ 可选化 (D10) 与说侧可感知播放 (D11)：say/__content__ 有真实播放时返回描述秒数、无播放
  返回 None、中断 raise STOPPED(301) 带最后片段；真实播放文本经 on_sample → PlaybackSample.text 对齐；顺带清 cmd_task
  交叉耦合。SpeechTopic 需重设计。
title: Speech Governance — 解耦、多后端、容错降级
updated: '2026-09-04'
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

### 剩余 Phase

```
Phase 2 ✅ → Phase 1 ✅ → Phase 5 (默认空 speech) → D6 (docstring 示例) → Phase 3 (多 provider) → Phase 4 (降级)
```

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

### Phase 3: TTS 多 provider (P1)

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
| 5.6 | NullSpeech 打字机延时 | `core/speech/null.py` | pending |
| 5.7 | Speech provider 配置化 delay | `host/providers/` | pending |
| 5.8 | build_content_command 懒获取 Session → pub speech text | `core/speech/speech_module.py` | pending |
| 5.9 | Session 抽象定义 SPEECH_KEY | `core/blueprint/session.py` | pending |

### Phase 4: 容错降级 (P2)

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