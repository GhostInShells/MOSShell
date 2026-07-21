# CLI Manifests Governance — 信息丰度恢复

从 blueprint manifests.py 提取设计意图，迁入 manifests_cli.py。

## 蓝图关注的维度 (per type)

### ProviderInfo
- contract import path (name) + aliases
- contract docstring (第一行就是 description)
- provider_type (Provider 实现类的 import path)
- singleton vs factory
- contract source code (code-as-prompt)
- found module + file (discovery origin)

### ConfigInfo
- name + description (from schema)
- is_override 语义 (内存覆盖 vs 文件持久化)
- schema fields/types/defaults (ConfigSchema.json_schema)
- source code of config class
- YAML 默认值
- found module + file

### TopicInfo
- name + type + description (from schema)
- TopicModel import path
- JSON Schema (payload wire format)
- model source code
- found module + file

### ResourceStorageInfo
- storage_scheme + storage_host
- description
- found module + file

### NucleusMetaInfo
- name + description
- signal_names (what signals it handles)
- found module + file

## 当前 CLI 问题

| # | 问题 | 修复 |
|---|------|------|
| 1 | providers 列表无 docstring/aliases | 加 Contract Docstring 列 + aliases |
| 2 | signals/topics 列表描述被表格截断 | 缩短列表描述，详情走 search |
| 3 | configs 列表无 fields/schema | 列表加 Fields 摘要 (field:type) |
| 4 | Active Context 每个子命令重复 | 只在 explain 显示 |
| 5 | No mode active warning 反复出现 | 只在 mode-only 命令 (channel/nuclei) 报错 |
| 6 | resources "Host" 列名混淆 | 改为 "Storage Host" |
| 7 | "none"/"default" 裸 sentinel | 用常量 |

## 步骤

### S1. 信息丰度恢复 — 列表视图
- providers: Contract + Aliases + Type + Docstring + Found At
- configs: Name + Fields (field:type 摘要) + Description + Found At
- topics: Name + Type + Model Path + Description + Found At
- signals: Name + Description + Found At (已有 detail 的 info)
- resources: Scheme + Storage Host + Description + Found At

### S2. 重复信息消除
- Context header 只在 explain 显示
- No-mode 提示只在 channel/nuclei 等 mode-only 命令显示
- 列表命令只显示数据，无警告

### S3. Detail 增强
- provider detail 已有 source，加 aliases
- topic detail 已有 JSON schema，加 model source
- config detail 已有 YAML + JSON Schema + source，加 is_override 标记

### S4. 删除 blueprint manifests.py (确认无依赖后)
