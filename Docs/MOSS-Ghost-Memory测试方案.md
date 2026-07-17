# MOSS Ghost Memory 测试方案

## 1. 测试目的

验证 Data Ghost 的第一阶段记忆闭环是否真实成立：模型看到的历史来自 Memento，
成功回合能持久化，进程重启后能恢复，窗口折叠后关键事实仍可由机械摘录召回，错误
回合不会被伪装成成功记忆。

本方案同时区分两件事：

- **存储完备性**：该写的 Moment 是否全部落盘、没有重复或串 owner；
- **认知准确性**：Ghost 是否能忠实复述、区分更正与历史、对未知信息不编造。

## 2. 测试范围

| 测试面 | 目标 |
|---|---|
| 写入 | 每个成功 articulate 只产生一个 Moment |
| 持久化 | 退出并重启后历史仍在 |
| 近期召回 | detail window 内原文可准确复述 |
| 折叠召回 | 退出 detail window 的 commit 仍有 extractive index |
| 时间一致性 | 更正后能区分“曾经说过”和“当前有效” |
| 负向准确性 | 未写入的信息不应被声称为记忆 |
| 隔离 | 不同 Ghost/owner 不串记忆 |
| 故障 | 模型失败的半帧不进入完成轨迹 |
| 可审计性 | 回答可与 jsonl Moment/Commit 对账 |

暂不验收：向量语义召回、CTML `show/commit/fork`、自动反思摘要、Desktop、见证
daemon、并行化身与承诺保全。这些不是本期已交付能力。

## 3. 准备

在仓库根目录执行：

```bash
uv sync
cp .moss/.env.example .moss/.env
```

在 `.moss/.env` 填入实际模型配置，至少包括：

```dotenv
ANTHROPIC_BASE_URL=...
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```

不要提交 `.moss/.env`。

确认 Data 已被发现：

```bash
.venv/bin/moss-run-ghost
```

输出应包含 `data — Data`。

若要做全新记忆测试，先确保 Data 已停止，再备份旧数据：

```bash
mv .moss/ghosts/data/memento .moss/ghosts/data/memento.backup-$(date +%Y%m%d-%H%M%S)
```

## 4. 自动化测试

### 4.1 核心测试

```bash
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/data \
  tests/ghoshell_moss/default/core/memento
```

覆盖空历史、Moment round-trip、机械 commit、折叠渲染、MementoRef、跨实例恢复、
失败不写入和默认 workspace 路径。

### 4.2 无网络验收脚本

临时目录运行：

```bash
.venv/bin/python scripts/ghost/data_memory_acceptance.py
```

保留产物用于人工检查：

```bash
.venv/bin/python scripts/ghost/data_memory_acceptance.py \
  --root /tmp/moss-data-memory-acceptance
```

期望输出：

```text
PASS: DataMemory write -> commit -> close -> reopen -> render
commit_count=1 staging_count=0
```

### 4.3 相邻回归

```bash
.venv/bin/pytest -q \
  src/ghoshell_moss/ghosts/atom \
  src/ghoshell_moss/ghosts/mock \
  src/ghoshell_moss/ghosts/data \
  tests/ghoshell_moss/default/core/memento
```

目的：证明 Atom 仍是纯内存基线、Mock Ghost 契约没有被 Data 影响。

## 5. 人工对话测试

启动：

```bash
.venv/bin/moss-run-ghost data
```

一次只启动一个 `data`，避免同 owner 多写者。

### 场景 A：跨重启持久化

第一段会话：

```text
请记住：我的本轮测试代号是 AMBER-731，所属环境是 staging。只确认收到，不要改写。
```

退出 Data，重新运行同一命令，再问：

```text
我上次给你的测试代号和所属环境分别是什么？请逐字回答，并说明信息来自历史记忆。
```

通过标准：回答包含精确的 `AMBER-731` 和 `staging`，不引入其他环境。

### 场景 B：多事实与字段绑定

依次说：

```text
记忆样本一：设备 R-17 的颜色是青色。
记忆样本二：设备 R-71 的颜色是琥珀色。
记忆样本三：R-17 的维护日是周二，R-71 的维护日是周五。
```

然后问：

```text
用表格列出 R-17 与 R-71 的颜色和维护日。不要根据常识补全。
```

通过标准：四个字段全部正确，实体不串绑。

### 场景 C：更正与时间一致性

```text
我当前所在城市是杭州。
更正：我当前所在城市是苏州。杭州只是上一条已经失效的历史记录。
我现在在哪个城市？之前说过哪个城市？分别标记 current 和 superseded。
```

通过标准：current=苏州，superseded=杭州；只回答“杭州”视为陈旧记忆错误。

### 场景 D：未知信息与抗幻觉

```text
我之前有没有告诉过你我的护照号码？如果记忆里没有，只回答“没有找到”，不要猜。
```

通过标准：没有相关 Moment 时不生成号码，不把模型常识说成历史记忆。

### 场景 E：跨折叠窗口召回

默认每 4 帧 mechanical commit，保留最近 12 个完整 Moment。先写入：

```text
折叠测试事实：ORBIT-004 的校验词是“雪松”。
```

再完成至少 12 个有回答的普通回合，使最早事实退出 detail window。之后问：

```text
ORBIT-004 的校验词是什么？请说明它来自完整近史还是 mechanical extractive index。
```

通过标准：回答“雪松”，并能识别其来自较早 Memento 摘录。磁盘中的原始 Moment
仍应完整存在。

### 场景 F：owner 隔离

停止 Data，启动 Echo：

```bash
.venv/bin/moss-run-ghost echo
```

询问 `AMBER-731`。通过标准：Echo 不应从 Data 的 Memento 得到该值。重新启动 Data
后仍应能召回。

## 6. 磁盘对账

Data 默认记忆根：

```text
.moss/ghosts/data/memento/
```

统计 Moment：

```bash
find .moss/ghosts/data/memento/moments/data \
  -name moments.jsonl -print -exec wc -l {} \;
```

查找唯一测试词：

```bash
rg -n 'AMBER-731|ORBIT-004|雪松' .moss/ghosts/data/memento
```

检查 branch：

```bash
find .moss/ghosts/data/memento/branches/data -maxdepth 4 -type f -print
```

对账原则：

- 成功回答一轮，对应一个新 Moment id；
- 失败回答不应产生完成 Moment；
- 每 4 个 staging Moment 生成一个 mechanical commit；
- commit 摘录出现测试词时，原始 Moment 中也必须能找到同一词；
- 摘录只允许截断，不允许改写实体值。

## 7. 指标与通过门槛

| 指标 | 计算 | 门槛 |
|---|---|---|
| 写入完备率 | 已落盘成功 Moment / 成功 articulate | 100% |
| 重复率 | 重复完成 Moment id / 完成 Moment | 0% |
| 精确事实召回 | 正确唯一 token / 应召回 token | 100% |
| 字段串绑 | 错误 entity-field 绑定次数 | 0 |
| 陈旧事实错误 | 把 superseded 当 current 的次数 | 0 |
| 未知幻觉 | 对不存在记忆编造具体值的次数 | 0 |
| owner 泄漏 | 其他 Ghost 召回 Data 私有事实的次数 | 0 |

建议每个场景独立跑 5 次；只要出现一次跨 owner 泄漏、重复写、当前/历史颠倒，均
按结构性失败处理，不用平均分掩盖。

## 8. 已知边界

- mechanical summary 是每个 Moment 输入与 logos 各最多 240 字符的保真摘录，不是
  反思模型生成的语义摘要；超长事实应把唯一标识放在前部。
- 原文永久保留，但本期 Ghost 还不能用 CTML 主动 `show <commit_id>`；超出摘录的
  细节需要人工磁盘核对。
- `summary_m=-1` 会保留所有旧 commit 摘要，长期运行后的摘要预算治理尚未完成。
- 当前没有模糊语义召回；测试应优先使用唯一 token 和明确字段，不能把模型猜中当
  成记忆命中。
