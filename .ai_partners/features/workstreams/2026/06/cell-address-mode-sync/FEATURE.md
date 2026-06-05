---
created: 2026-06-04
depends: []
description: 'Fix: Environment.set_mode() must sync _cell_address so MatrixImpl correctly
  identifies the host cell, enabling cross-process channel discovery for apps brought
  up by a mode.'
milestone: null
priority: P1
status: in-progress
status_note: root cause identified, fix applied to Environment.set_mode()
title: Cell Address Mode Sync
updated: '2026-06-05'
---

# Cell Address Mode Sync

## Motivation

`moss-run-ghost echo --mode douyin_chat` 启动 Ghost 后，Ghost 进程发现不了
mode 通过 `bringup_apps` 启动的两个 app 的 channel。

根因：`Environment.set_mode()` 只更新 `_moss_mode`，没有同步更新 `_cell_address`。
当 mode 在 `Environment.__init__` 之后通过 `set_mode()` 变更时，`_cell_address`
保持初始 mode 的值（如 `host/default`），而 `MatrixImpl` 中 `main_cell.address`
基于实际 Mode 对象计算（`host/douyin_chat`）。两者不匹配导致 `_is_main` 判定为
False，`channel_proxy()` 被阻止，AppStoreChannel 无法为 app 创建 Zenoh 代理 channel。

## Key Decisions

1. **在 `set_mode()` 内同步 `_cell_address`**，而非在 `MatrixImpl` 内做容错。
   理由：`_cell_address` 是 Environment 的派生属性，保持 self-consistent 是
   Environment 的职责。MatrixImpl 不应该猜测 cell_address 是否正确。

2. **仅当 `MOSS_CELL_ADDRESS` 未显式设置时才更新**。App 进程通过 `MOSS_CELL_ADDRESS`
   显式指定自己的 cell_address（如 `app/group/name`），`set_mode` 不应覆盖。
   Host 进程的 cell_address 从模板派生，mode 变更时必须同步。

## Implementation Notes

- 修改文件：`src/ghoshell_moss/core/blueprint/environment.py` (`set_mode` 方法)
- 新增测试：`tests/ghoshell_moss/host/test_environment_set_mode.py` (3 cases)
- 关联修改：`src/ghoshell_moss/host/matrix.py` (同一轮修复的配套改动 —— 将
  `DEFAULT_CELL_ADDRESS` 模板比较改为与 `main_cell.address` 的实际值比较)

## Additional Fix: Config YAML 不生效

### 症状

`.moss_ws/configs/tts_factory.yml` 中 `default_speaker` 配置为 `可爱女生`，但实际
TTS 输出始终为硬编码默认值 `知性灿灿`。

### 根因

`MatrixImpl._ensure_container_lifecycle_ctx_manager` 对**所有** merged config 调用了
`set_config`，包括全局 manifest 的默认实例。执行顺序：

1. `container.bootstrap()` → `HostEnvConfigStoreProvider.bootstrap()` → `get_or_create()` → 读 YAML → 缓存 = 正确值 (`可爱女生`)
2. `self.configs.set_config(config_info.config)` → `config_info.config` 是 merged manifests 中的实例（全局默认 + mode 覆盖合并后的结果）。对于无 mode 覆盖的 config，这就是全局默认实例 `TTSManagerConfig()` → `resolve()` = `default_speaker='知性灿灿'` → **覆盖缓存**
3. `self.configs.get_or_create(config_info.config)` → 缓存命中，返回错误值

`set_config` 的本意是让 mode 层做内存覆盖，但不应把全局 manifest 的默认实例也覆盖到
缓存中——全局实例只是 schema + 默认值声明，不应覆盖 YAML 文件的值。

### 修复

分离两条路径：
- **全局 config**：只走 `get_or_create`，从 YAML 加载，缓存 YAML 值
- **Mode config**：在 `get_or_create` 之后再走 `set_config`，将 mode 声明的覆盖值写入内存缓存

具体改动：`_ensure_container_lifecycle_ctx_manager` 中先对所有 merged config 调
`get_or_create`（确保 YAML 加载），再仅对 `self._current_mode.manifest` 中的 config
调 `set_config`（mode 覆盖）。全局默认实例不再经过 `set_config`。

- 修改文件：`src/ghoshell_moss/host/matrix.py` (`_ensure_container_lifecycle_ctx_manager`)