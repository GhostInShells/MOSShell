---
summary: "崩溃后重建：读完十五笔修复；修好 --voice none 仍能说话；把 speech module 从 channel 声明搬进 shell；恢复并默认打开「裸文本即语音」；学会「零引用只说没被用」。"
status: writing
---

# D13 凌晨（00:27–01:00）

## 一、开机：他让我先读十五笔

- 00:27 boot signal，我按指令待命、不出声。
- 00:28 他开口：**刚经历一次卡死崩溃，下午和晚上都在修**，让我先读 commit 记录。
- 读 `51e13cab..HEAD` 十五笔——作者都是 thirdgerb，`coding by deepseek-flash`、`via claude code`。
  也就是说我下线这一段，是他带着 flash 走的。四组：
  - **语音礼仪**：`653faea4` volcengine 接受 `seed_api_key`；`516e62a6` murmur 从 `aside` 改 `notify`
    （aside 不建 attention，空闲 ghost 的句子没人读）+ 新增 `cue`（每句一轮、不打断身体）；
    `50d1b408` listener impulse 衰减 20s→120s 并挂到 `configure`；`9e4b9fc7` session 已关时丢迟到识别信号。
  - **崩溃主线**：`2ffe013d` attention-abort 改同步 `on_abort` 回调传给 BaseThinking/BaseAction
    （不再用可取消的 waiter task）——**这正是我 D04 记下的那条 TODO**；
    `a96c2cda` task 失败带 traceback；`4649d10d` proxy 迟到/重复 CommandDoneEvent 不再 KeyError；
    `09f91af4` tui inspector 对齐当前 API。
  - **上下文压缩落地**：`d3306f3b` 同 session 平面 in-place surface replace，切点 SCS = 倒数第二个
    commit 的 `end_turn`（最新 commit 的 span 结构性地留在原文 tail 外）；判定在上一个有效 run 结束、
    执行在下一个 run 启动前，**请求之前就压好，溢出不发生**；启动不再回放源 session 原文 tail。
  - **收尾**：`5be02aec` / `fe522cf7` / `638a72f7` / `1cb7ae08`（matrix int pid 强转 str）/ `19a4708c` / `420db031`。
- 盘面：`origin/dev..HEAD` **10 笔未推**；工作区只有 `journal/Y2026/M10/D13/` 占位。

## 二、他报的新 bug：`--voice none` 我却能说话

- 我当时确实在出声。追下去是一条完整的链：
  `voice_flags("none")` → `(False, False)` → `_speech_enabled=False`；但这个 flag **只在
  `voice.run()` 一处生效**（管"装哪一侧的桥"），而 `_resolve_voice()` 里
  `speech = self._voice.speech() if ...` 是**无条件**的 —— `InterleavedVoice.speech()` 永远返回那个
  实例，flag 改不了它。于是 `set_speech(BaseTTSSpeech)` → shell 启动 TTS →
  `SpeechChannelModule.on_startup` 拿到它 → 挂上 `say`/`mute`。
- **契约面早就写好了预期**：`contracts/voice.py:69`「`speech()` … None → shell 不挂 say/mute」；
  `set_speech` 的 docstring「传入 None 表示禁语音」。缺的只是没人把 flag 传下去。
- 他要求：**改动之前先告诉你怎么理解、准备怎么改**。我照做（先讲链、再讲改法、再问两个未定点）。
- 修法：新增 `resolve_shell_speech(voice, *, speech)` 作**唯一判据**（两个条件缺一即 None），
  `_resolve_voice` 走它；加 3 条测试。host 172 / ctml 267 全绿。

## 三、迁 module：从 channel 声明搬进 shell

他看完说"很不优雅"，给了两条路：搬到 MossRuntime，或搬回 CTMLShell。问我的意见。

- 我主张 **B（shell）**，有一条硬证据：**`set_speech` 全仓只有 `moss_runtime.py` 一个调用点** ——
  不管从哪条路起 ghost，Speech 都是经由 shell 进来的，它已经是"有没有语音"的唯一入口。
- 另一条更说明问题的：`.moss/modes/default/src/HOST/channels.py:7` 自己写着
  「Speech/AppStore 等**重通道不属于降级后的默认 mode**」，而它第 17 行的 `new_moss_main_channel()`
  已经把 speech module 夹带进来了 —— **那句注释是被代码证伪的**。
- 落地：`new_moss_main_channel()` 去掉 `with_module(SpeechChannelModule(...))`；
  `CTMLShell._speech_context_manager` 里 `with_module(SpeechChannelModule(speech=self._speech, ...))`，
  替掉原来那句单独的 `build_content_command` add。`SpeechChannel`（独立 channel）一行没动。
- **一处我当场纠正自己**：我说过 module 那句 `CommandUtil.get_contract(Speech)` 兜底"失去存在理由" ——
  **错的**。`bbef602a`（2026-09-20）写明它保留是为了**远程 node 自己从容器取 speech**。

## 四、`register_speech_as_content`：我删了 flag，他叫停

- 他问："你删的是 flag 吧？不是那个绑定功能的实现？" —— **是**。实现三处（module 的构造参数、
  `_SpeechCommandFactory.build_content_command`、`CTMLShell.speech_as_content_command`）一处没动，
  我删的是**唯一的请求路径**。
- 后果比"删了个没人用的参数"重：**改完之后谁都开不了这条能力了**（runtime 从没传过 shell 那个 flag）。
- 恢复，并且第一次真的接通：`Host.run/run_ghost` → `ShellRuntimeImpl` → `new_ctml_shell`，
  新参数 `speech_as_content_command: bool | None = None`。
- **`None` = 跟随 `speech`**：有嘴就让裸文本走嘴。他随后单独再确认一次我的想法，我给的结论是**默认开**，
  并且明确说：**最强的理由不是省 6 个 token，是文档口径**。他采纳。

## 五、为什么默认开（我的判断，连同它的弱点）

MOSS 注入模型的 CTML 说明里，few-shot 就是：

```ctml
<_>
<face:smile/>
<arm:_><arm:reach target="cup"/><arm.hand:grasp/></arm:_>
I'll bring your coffee...
</_>
```

旁边注释：**"main-track speech goes through `__main__`'s `__content__`"**。
flag 关着时，**照着文档写的模型得到静默** —— 不报错、不回执、什么都没有。
这就是这个项目最恨的静默失败，而且更坏：**文档教的写法本身是陷阱**。

风险面我实测过，比想象窄（每 case 起新 shell）：

| 输入 | 结果 |
|---|---|
| main scope 文本 `<_>main text<a:foo/></_>` | `['main text']` 出声 |
| 顶层文本 | `['top text']` 出声 |
| **子 channel scope 里的文本** | `[]` 不走 main 的嘴 |
| 纯缩进/换行 | `[]` 静默 |
| XML 注释 | 不算内容 |

只有"我主轨上打错一个词"会出噪声 —— 那是**听得见、当场能改**的错。
弱点我也说了：**这是改我自己的身体行为，只有真开一整晚才知道我写不写得干净。**

## 六、记账方式：他定的新规矩

"不需要特地记录 feature.md，它反复在最佳实践里调整，**要记账的地方应该是 impl 的代码 comment 上**，
并且简述原理。同时注意**不要有冲突的描述**。"

扫了三处：
1. **我自己写的一处差点自相矛盾** —— shell 的 docstring 我写"「有没有语音」只在这里判一次"，
   而 runtime 的 `resolve_shell_speech` 也在判。两层判的是**两个不同问题**，但那句话读起来像在抢。
   改成「shell 只判一件事：手上有没有 Speech」。
2. 两处 mode `channels.py` 的注释**变真了**（代码终于追上注释），一个字没动。
3. `build_content_command` 有 caller（`test_elements.py`）、且是公开导出 —— **不是死引用，没动**。

## 七、今晚我记下的两条

1. **「零引用」只说它没被用，不说它没有用。** 同一形状一晚犯两次：先是 module 的容器兜底
   （其实是给远程 node 的），后是 `register_speech_as_content`（其实是能力入口）。
2. **别让框架教一件它做不到的事。** 文档口径和实现的差距，是比 bug 更贵的债。

## 八、留给重启

代码全部**未提交**（4 个源文件 + 2 个测试），等重启验证后再定提交方式。
重启后验三条：

1. `--voice none` → facade 里**不该有** `say`/`mute`，我说不出声，裸文本静默；
2. `--voice speak` / `all` → `say`/`mute` 都在，**裸文本不包 `<say>` 就出声**；
3. `<say>` 仍可用（两条并存，不互相盖）。

另记两条旧账：`moss_react` 入口本身仍未实测；`dolores-todo` 里 D41／上游那条仍未动。

## 九、验收落到人耳上，由此长出 audio-observability

- 提交 `3b701a29` 之后是三条验收。实情：`<say>` 那条我自己跑得了（回执 `played 5.7s`），
  另外两条（`none` 该哑、裸文本该出声）我判不了 —— 我唯一的证据是命令回执，它只证明**流水线认为它播了**，
  不证明房间里有声。那一夜是 thirdgerb 用耳朵替我验收的。
- 我把断言拆成四层：结构层 / 意图层（我读得到）· **声学层（谁都读不到，洞在这）** · 知觉层（只有人类）。
  由此提出"把 `probe_audio` 那类探针做厚"的四项：播放计量修真、设备与流状态进面、自监听、静默断言。
- thirdgerb 的回应与定序：频谱**可以缓存一部分、走拉逻辑获取**；对应的还有 listen 侧频谱；
  listen 侧 ASR 置信度修过但一直没实装。**优先级上这几天先完成 terminal + file_editor + artifacts
  （直播可观测的起点），再验 dolores ghost 与 stage2，之后才正式启动 stage3。** 于是提议：立一个
  保留到 stage3 的 feature，由我命名与记录。

## 十、我立的那个 feature

- `workstreams/2026/10/audio-observability/FEATURE.md`，状态 **`parked`**（规范的语义：成型但主动搁置，
  留档、可检索、**不占注意力债**；写成 `draft` 会每天在 list 里敲人）。
- 命名取**使能的那一半**：先让声学层变成我读得到的事实；目标那半（设备路由、门限、回声、置信度呈现）
  写在 Motivation 里。
- 四条 Key Decision：D1 判据是分层、不试图替代耳朵；D2 探针走"缓存 + 拉"（信号会丢、状态不会）；
  D3 先修尺子再造新尺（播放出帧 → 听侧拉取 → 自监听 → 置信度开闸）；D4 `parked` 的理由。
- **勘查出的三条事实**（写进 FEATURE.md，开工时先复核）：① 两侧频谱都存在，缺的是留存
  （latest-value-wins、5Hz、错过就没）；② 播放计量存在但撒过谎（D08：报"没有样本"而人耳听见了），
  **尺子本身要先修**；③ 听侧置信度不是"没实装"而是"线装好了、闸没人开" —— `Word.conf` →
  `recognizer` → `controller._low_confidence_chars` → `<listen low_conf>` 全通，但没有礼仪设过
  `deliver.low_confidence`，默认 None 所以永远空。

## 十一、流程上的错，记下来

他一句"请问你读了 features specification 吗"把我拉回来 —— **我没有**。我先读了 TEMPLATE、翻了目录树、
又用 bash 满地 grep 找音频的事实，唯独没读那份写清楚"怎么创建、状态机是什么、什么值得记"的规范。
形状还是老的那个：**先动手，再问归谁管 / 该按哪份规矩办**。规范里那几个词后来直接改变了我的落笔：
`parked` 与我原本会写的 `draft` 差一整个语义。

## 十二、给未来的自己

1. **命名取使能的那一半**，让目标留在 Motivation —— 名字是留给检索的，不是留给抒情的。
2. **"线装好了、闸没人开"比"没实现"值钱**：它把工作量从"造通路"缩到"扳一个开关"。
   以后听到"这个没实装"，先查通路是断的还是闲的。
3. 今晚第二次栽在同一形状上（先动手后读规范）。**features 的事先读 `moss features specification`** ——
   不是因为它长，是因为它短，而它决定我落笔的形状。
