---
created: 2026-09-21
depends: []
description: 'ConfigStore 递归解析 (层叠 store + materialize 策略), 让 ghost home 成为一层配置覆盖.
  最常见需求: per-ghost 改默认语音. 不物化 → workspace 保持唯一种子源, ghost home 只装 delta.'
milestone: null
priority: P1
status: completed
status_note: 机制 + 发现面 (source_path) 落地; 全量回归通过 (一条 flaky 二跑通过).
title: Ghost Config Layer
updated: '2026-09-22'
---

# Ghost Config Layer

> Use `moss features set-status ghost-config-layer <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

per-ghost 覆盖默认配置是**常态需求**, 不是特例。最典型的一条: ghost 想改自己的默认语音 —— 现在的配置只有 `workspace/configs/`(base) + mode 一层, 没有任何 ghost 落点, 于是改一声线就改了所有 ghost 共用的文件。

机制根源 (人类提出): **ConfigStore 初始化时可以传入下一层 store, 再加一个 flag 决定要不要在本地 materialize**。这跟 GhostMeta.providers() 覆盖契约的能力无关 —— 不被采纳, 见 D1。

驱动实例: `ghost-prototype-dolores`。

## Design Index

- Key design documents: `design/` （未写）
- Key discussion records: `discuss/` （未写）

## Key Decisions

- **D1. 分层是通用机制, 不是 ghost 专属 hack。** 不走 `GhostMeta.providers()` 覆盖 `ConfigStore` 契约那条路: 那条路只在 host 进程生效 (node/cell 子进程各建 container, 拿不到), 而且是隐形 magic。层叠 store 由核心从 Environment 构造。
- **D2. 两个正交旋钮, 不是两套概念。**
  - `inherit`(结构): 本层 miss 后是否递归到下一层 store。
  - `materialize`(策略): `get_or_create` 在本地物化, 还是把创建交给下一层。
  现有 mode 层正好是这套参数空间的一个点 —— 有 inherit(读回退 base) + materialize=True(写 mode 文件)。ghost 层是另一个点 —— 有 inherit + **materialize=False**。所以这不是为 ghost 发明的第二套机制, 而是把已有 mode 语义收成特例。
- **D3. ghost 层不物化。** `get_or_create` 全链 miss 时物化到终端层 (workspace)。保证 ghost home 只装 delta, workspace 保持唯一种子源; 否则每开一个 ghost 就在它 home 里复制一份全量配置, N ghost × M configs 复制 + 双份漂移 (改一次全局默认, 已跑过的 ghost 全留在旧副本上)。
- **D4. 每层内部同一套解析, ghost 层不 skip mode。** 每一层都跑同一条文件存在性检查: `{name}.{mode}.yml` → `{name}.yml`; 两者都不存在才 fall through 到下一层。所以 ghost home 里同样可以有 per-mode 覆盖 —— **文件存在性本身就是显式覆盖声明**, 不需要额外的声明机制。
  链 (dolores 在 mode `voice` 下): `ghosts/dolores/configs/tts.voice.yml` → `ghosts/dolores/configs/tts.yml` → `configs/tts.voice.yml` → `configs/tts.yml`。
  层优先级高于 mode 优先级: 本层的通用文件压过下一层的 mode 文件。
- **D5. 不开放写控制面。**
  现状 `ConfigStore` 已经有写路由的旋钮 (`save(conf, mode=...)` / `set_config(..., mode=...)`)。加了层链之后对称的动作是再加 `layer=` 参数, 或把 `inherit` 那层暴露成可写句柄, 让调用方指定"写 ghost 层"还是"写 workspace 层" —— 这就是被否掉的写控制面。
  落地两条:
  1. `save` / `set_config` 没有 `layer` 参数, 永远写本层 storage (ghost 进程里 = ghost home, 非 ghost 进程里 = workspace)。一条规则, 无例外分支。
  2. `inherit` 那层只读可见 (D6 发现面要报来源层), 不作为可写句柄外露。
  改全局默认的动作因此是文件操作: copy 一个到目标层再手改 / 写脚本改。共享配置文件本就是所有人读的同一个文件, 从某个活着的 ghost 进程通过 store 远程写它, 归属与义务都含糊 (谁改的 / 何时生效 / 要不要通知其它进程), 而该需求用文件操作已经满足。
  白拿的性质: `set_config(override=False)` (只进缓存) 在 ghost 层 = 会话级 ghost override —— 不落盘, 不影响其它进程, 重启即失。
- **D6. 读侧暴露解析链, 但只读。** 发现面要能报"这个值解析自哪一层 + 实际路径", 否则"我改了 voice 怎么没生效"会变成新的隐形摩擦。这是自省, 不是写路由 (与 D5 不冲突)。
- **D7. 上层不缓存 fall-through 结果。** 保住"层缓存 == 层文件"这条不变量: 人手工往 ghost home 放个新文件, 不该被上层旧缓存遮住; `invalidate` 也不必跨层级联。fallback 那层自己本来就缓存, 重复读代价很小。
- **D8. 构建点是 `EnvConfigStoreProvider` (env 参数: `ghost_name` + `mode_name`)。** 与 mode 走同一条 env 继承通道 (`Environment.seal()` 写 `os.environ`, 子进程继承, `Environment.__init__` 回读) → node/cell 子进程自动拿到 ghost 层。这正是核心构造胜过 provider 覆盖的地方。
- **D9. manifests 层不动。** ghost 没有 manifests, 也不需要: manifests 发现的是 Config **实例** (声明), 不是文件 (已核实 `manifests_cli.py:305` 取 `m.value()` 展示类默认值 + schema + `found_at`)。发现面要补的是"经 store 解析出的生效值 + 来源层", 落点是 `moss manifests configs <name>` 的 detail 分支。
- **D10. ghost 层目录不存在 = 正常态。** 不给模板 stub, 不 scaffold, 不创建: 目录缺失直接 fall-through (与 D3 同一条)。"这里能放配置"的发现由发现面承担 (D6/D9), 不靠 home 里预置一个空文件或注释示例。

## Terminology

概念的正式名用**递归解析 (recursive resolution)**, `fallback` 让给它已经在用的位置: `ConfigStore.get(..., fallback: bool)` 这个 kwarg 现指"mode 文件缺失→读 base"。同一个词承载两层意思必然歧义 (`ConfigInstanceRegisterBootstrapper(fallback=False)` 这类调用点会立刻分不清"不读 base"还是"不读整条链")。

构造参数已定名 `inherit` / `materialize`。

## Open Problems

无遗留 — O1 (save 落本层的 docstring 说明) 与 O2 (发现面 source_path + `moss manifests configs <name>` 生效值) 均已随实现落地。

## Acceptance

- **后代进程读到 ghost 层** (已做, `test_descendant_process_resolves_ghost_layer`) — 真起一个子进程, 只给环境变量, 断言它解析出 ghost home 的值。
  断言的是**行为** (这个进程属于 ghost X, 就该解析到 X 的覆盖), 不是机制 (env 必须逐字继承)。所以将来谁为了让 env 干净而加一层过滤, 只要行为保住, 测试照样绿; 只有行为真的回退了才红 —— 那是该停下来看一眼的时刻, 不是挡路石。
  (用裸 python 子进程而非完整 node: 被测的那一环是 env 继承 + 从 env 装配, 与 node 走的是同一条路。)
- **ghost 层目录 / 文件不存在时不创建 ghost home/configs, 直接 fall-through** (D3 + D10 的负向断言)。
- **`get_or_create` 全链 miss → 落在 workspace**, ghost home 不被写入。
- **`save` 落 ghost 层**, workspace 文件不动。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->