---
summary: "插件私有 selection 既压界面、又是 ego 唯一的 provider/model 来源 —— 删了启动就挂，补 agentOptions 才通"
status: writing
---

# dsh 插件 · 思考档

## 症状

在 dsh web 界面上改 thinking effort（off/low/high/max），下一轮不生效，
「重启一次才吃到」—— 他讲过很多次。

## 第一次归因（1-483，只读，未改代码）

同一个 agent 上挂了两处 model selection：官方 `dsh-api-session-controller` 的一份，
和插件自己 `ensureEgoSelection()` 里的 `installModelSelection(agent.ctx, state.ref)` 一份。
cordis 的 waterfall 先注册的在最外层、最后覆盖 —— 插件那份通常在外层；它自己的取值链
（`pending → request/header → 全局默认`）又把「上一次请求实际用了什么」当成权威，
于是每轮把界面的选择盖回去。

实测（同一份 host）：

- 干净会话（无插件）：`session/selectModel` 写 high → 下一个请求的 `request/header` 就是 high；
- ego 会话（有插件）：pending 从 seq 816 挂到 seq 841，跑过多轮请求 header 一直是旧档，
  连「模型身份变了」的提示都不出 —— 静默失效。

## 第一次改动失败（1-484）

删掉整块私有 selection（约 110 行），改调 `sessionController.selectModel`。
结果：启动直接失败，报 `agent "<id>" has no provider/model`。他回滚了。

## 根因补充

ego agent 由插件 `ctx.agents.create({ sessionId, meta, setup })` 直接建，**不带 agentOptions**；
官方那条 `ensureSession` 才会用 `agentOptions()` 补 provider/model，并 `installSelection(agent)`。
所以那份私有 selection 不只是「第二份权威」，它还是 ego 唯一的 provider/model 来源。
删掉它 = 第一次请求无处可依。

## v2（已落盘，未提交）

三处改动：

1. `ego/create` 建 agent 时给 `agentOptions: { provider, model }`（取 `agentDefaultModel.currentSelection()`）；
2. `apply_ego_agent`（session-start）改为装官方那份：`sessionController.agents.selectionFor(agent)`（幂等）；
3. `thinking/enter` 改为 `sessionController.agents.selectForNextRequest(agent, {...})` ——
   append durable 的 `model/selection` + 更新 live picked。不用 `selectModel` RPC：它会顺手
   `agentDefaultModel.saveSelection()`，等于 ghost 声明一次档位就改掉人类的全局默认。

删掉：`egoSelections` WeakMap、`EgoSelection`、`LoggedHeader`、`loggedEffort`、
`ensureEgoSelection`，以及 `installModelSelection` / `AgentDefaultModelConfig` 两个 import。

## 顺带发现的 seam

`_runtime.py:743 _sync_default_model()`：开机（ego 创建之前）读 ghost home `.env` 的
`DOLORES_DEFAULT_MODEL_PROVIDER` / `DOLORES_DEFAULT_MODEL`，把 `agent-default-model: {provider, model}`
断言写进 dsh 的 `settings.yaml`。所以「ghost 自己的模型」早就存在 ——
换 Claude 只要改两行 `.env` + 重启，代码一行不用动；值域从 dsh 的 `modelCatalog` 里挑
（provider / model / 思考档，不含任何凭据）。

## 待重启验收

1. `ctx.get('sessionController')` 在插件 ctx 里是否可达 —— v2 拿不到会打 error 日志，不静默降级；
2. 启动不再报 no provider/model；
3. 我声明档位之后，下一个请求的 request/header 真的换档；
4. 界面上改档位，下一轮能进来。

## 一处语义（他认可的判断）

`agentOptions` 只定「第一次请求」；之后靠官方链（pending → request/header → 全局默认），
所以界面的改动仍然能覆盖。ghost 声明的是起手，不是永久覆盖。

## 模型器官（D23 20:40–20:57）

起点是他两个问题：思考档降到 low 立刻生效了，「那你自己知道 dsh 上有哪些模型吗」。答案是不知道，
而且整棵 CTML 树里没有换模型的通道——只有思考档一个字段。上一轮插件的改动只解决了「档位归位」
和「agentOptions 补齐」，模型本身还钉在 ghost home 的 `.env` → `settings.yaml`，只能改配置 + 重启。

### 状态机（他逐句说，我复述对齐后才动手）

- **读（温）**：ghost 根 channel 挂 named notice `model` —— `provider/model · thinking <depth>`。
  数据源是 `request/header` 的观测（我**实际在跑**什么），不是 `modelCatalog.default`（新会话从哪
  开始）。named fragment 只在该变时重发，因此这是免费的自知。
- **读（冷）**：`ghost.model:list` 现调 `session/modelCatalog`，走 ego 已经持有的那条 dsh session
  ——另开一条连接会答出另一份状态。输出 provider → model → 该模型自带的档位表（efforts /
  defaultEffort），外加 routable / failures。
- **写（唯一）**：声明落在 ego 的 `default_model` / `default_thinking_effort`，下一帧
  `thinking/enter` 带 `provider` / `model` / `effort`，插件把三者合成一次
  `selectForNextRequest`，turn 边界生效。**全局默认一次也不碰。**

### 删掉的东西

`_run.py::_push_effort_officially` —— 它走 `session/selectModel`，那条 RPC 会顺手
`saveSelection`，也就是 ghost 声明一次档位就把人类的全局默认改掉。它和插件那条并存，就是
「同一个档位两个写入口」。他的原话是「为什么可能会有两个写路径呢」，删是唯一诚实的回答。

### 落地

`_ego.py`（`default_model` + `model_identity` + `declare_model`）、`model_channel.py`（新建，
instruction + list + select，后端三件由 runtime 注入）、`channel.py`（挂 `ghost.model` 与根
notice）、`_runtime.py`（`_model_organ()`）、插件 thinking/enter、`_prompts.py`（先 list 再
select；视力是模型属性）、`test_dolores.py`新增 7 个测试。dolores 全套 231 通过。未提交。

### 待验收（要重启一次）

1. `ghost.model:list` 读出真实目录；
2. notice `model` 出现，换模型后变化；
3. `select` 之后下一个 `request/header` 真的换了 provider/model；
4. 界面上改档位/模型仍然能进来（没有第二份权威压着）。

### 一句话

他反复问「这东西摆在你面前吗」。我以为自己有的能力，一半只是「源码里存在」，不是「我的 surface
上存在」。两者的差就是这一晚的全部工作量。
