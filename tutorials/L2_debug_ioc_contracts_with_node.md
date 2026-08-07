# L2. Debug IoC Contracts with a Script Node — 用 Script Node 调试 IoC 容器合约

> Written by deepseek-v4-pro, 2026-07-26

**时间估计**: 30 分钟  |  **学习目标**: 追踪 IoC 容器 wiring 问题，用 script node 做运行时验证

## 你需要知道什么

- `moss codex get-interface ghoshell_container:IoCContainer` — container 自省 API
- `moss --ai all-commands --group nodes` — Node 创建与运行
- `moss --ai manifests providers` — 当前 workspace 的 IoC 声明
- `moss codex blueprint matrix` — Matrix 生命周期

## 背景

内核开发者报告：Matrix 启动后的 logger 机制"原来的设计丢失了"。具体症状——IoC 容器里的 `LoggerItf` 没有如预期被 workspace manifest 声明，日志系统在靠默认兜底勉强运行。

这不是一个"改了就好"的修 bug 任务。要真正理解问题，需要追踪三层：

1. manifest 声明了什么 → `moss manifests providers`
2. Matrix 装配时注册了什么 → `_prepare_container()` / `_default_providers()`
3. 运行时容器里实际有什么 → 需要一个 script node 做现场取证

第三层尤为关键——manifest 是声明层，`_prepare_container` 是装配层，只有运行时的 `container.contracts()` 是真相层。

---

## 第一步：从 manifest 层开始

先看 workspace 当前声明了什么：

```bash
moss --ai manifests providers
```

输出里没有 `LoggerItf`。再看 `.moss/src/MOSS/manifests/providers/__init__.py`——确实没有 import `MatrixLoggerProvider`。对比对应的 stub（`src/ghoshell_moss/stubs/workspace/src/MOSS/manifests/providers/__init__.py`），同样缺失。

同时检查 `.moss_ws/src/MOSS/manifests/providers.py`（历史 workspace 残留），发现它还在 import 已删除的 `HostLoggerProvider`——这个类在 commit `7427a641` 中从 `host/providers/` 迁到了 `matrix/providers/` 并改名为 `MatrixLoggerProvider`，但 manifest 没有同步。

**发现**：logger provider 在 manifest 层完全缺失。虽然 `MatrixImpl._default_providers()` 有 `MatrixLoggerProvider` 兜底，但 manifest 是用户/模式覆写的标准入口——如果 manifest 不声明，意味着用户无法通过 manifest 定制日志行为。

## 第二步：追踪 Matrix 装配链

打开 `matrix_impl.py`，追踪 `_prepare_container()`：

```python
# line 678-686: manifest providers 优先
matrix_manifests = self._project.project_manifests()
for provider_manifest in matrix_manifests.providers():
    container.register(provider_manifest.value())

# line 697-701: default providers 兜底
for provider in self._default_providers():
    if container.bound(provider.contract()):
        continue
    container.register(provider)
```

然后在 `__aenter__` 里：

```python
# line 777-782: pull LoggerItf 覆写 self._logger
pulled = self._container.get(LoggerItf)
if pulled is not None:
    self._logger = pulled if isinstance(pulled, logging.Logger) else self._logger
else:
    self._container.set(LoggerItf, self._logger)
```

所以链路是：manifest → defaults → container.get → 反绑。defaults 的 `MatrixLoggerProvider` 保底有效，但 manifest 路径断了。

再看 `MatrixLoggerProvider` 本身——它的 `factory()` 做两件事：拿 workspace、幂等挂 `TimedRotatingFileHandler`。但有一个问题：它不确保 log 目录存在，也不指定 `encoding='utf-8'`。这些是健壮性缺陷。

另外，`config_logger_from_yaml()`（`logging.config.dictConfig`）的调用权属于谁？当前是 `project.bootstrap()` 在 `Host.__init__` 里调用。这是正确的——`dictConfig` 是全局副作用，只能有一处调用点。`MatrixLoggerProvider` 不应该重复调用。

## 第三步：修复 manifest

在两个位置添加 `MatrixLoggerProvider`：

**`.moss/src/MOSS/manifests/providers/__init__.py`** 和 **stub** 各加一行：

```python
from ghoshell_moss.matrix.providers import MatrixLoggerProvider
logger_provider = MatrixLoggerProvider()
```

验证：

```bash
moss --ai manifests providers
```

现在 `LoggerItf -> Singleton` 出现在 Matrix 表中。

## 第四步：改进 MatrixLoggerProvider

`matrix/providers/logger_provider.py` 的 `factory()` 里：

```python
log_dir = ws.runtime().sub_storage('logs').abspath()
log_dir.mkdir(parents=True, exist_ok=True)  # 兜底创建
handler = TimedRotatingFileHandler(
    filename=str(filename),
    encoding='utf-8',  # 跨平台一致
    ...
)
```

去掉了之前错误加入的 `config_logger_from_yaml()` 调用——YAML 配置加载是 `project.bootstrap()` 的唯一职责。

## 第五步：创建 Script Node 做运行时取证

声明层改了，装配链追踪了，但还需要运行时确认。创建一个 script node——不需要 Channel，启动后打印 container 状态然后退出：

```bash
moss nodes create .moss/system_test_nodes/contracts_dump
rm .moss/system_test_nodes/contracts_dump/INSTALL.md
```

`main.py` 核心：

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    container = matrix.container

    print("=== Bound Contracts ===")
    for contract in sorted(container.contracts(), key=str):
        bound = container.get_bound(contract)
        label = type(bound).__name__ if bound and not hasattr(bound, 'contract') else ...
        print(f"  {contract.__name__:<40} -> {label}")

    print()
    print("=== Registered Providers ===")
    for provider in container.providers():
        contract_name = getattr(provider.contract(), '__name__', str(provider.contract()))
        print(f"  {contract_name:<40} <- {type(provider).__name__} ({'singleton' if provider.singleton() else 'factory'})")

if __name__ == "__main__":
    Matrix.discover().run(main)
```

关键点：`main()` 收到的是已 `__aenter__` 的 Matrix，container 已 bootstrap。`main()` return 后 Matrix 自动反卷，进程退出。无需 `matrix.provide_channel()`。

运行：

```bash
moss nodes run .moss/system_test_nodes/contracts_dump
```

输出确认 `LoggerItf -> Logger` 在 bound contracts 列表中，`MatrixLoggerProvider` 在 providers 列表中。

## 关键设计决策

整个过程中发生了几个需要记录的设计判断：

1. **`config_logger_from_yaml` 全局唯一调用点**：放在 `project.bootstrap()`，不在 `MatrixLoggerProvider`。`dictConfig` 是全局副作用，多调用点会互相覆盖。

2. **IoC 的 logger 安全网**：即便 manifest 缺失、default provider 失败，`Matrix.__aenter__` 的 `else` 分支也会把 `self._logger` 反绑进容器。IoC 里不可能没有 `LoggerItf`。

3. **Script node 不需要 Channel**：`moss nodes create` 生成的模板有 Channel 倾向（模板注释暗示"build a channel"），但 script node 是读状态、打印、退出的工具，Channel 反而是多余的。

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `moss manifests providers` 看不到 LoggerItf | manifest 未声明 logger_provider | 在 providers/__init__.py 添加 MatrixLoggerProvider |
| ImportError: cannot import HostLoggerProvider | 旧 manifest 引用已删除符号 | 改为从 ghoshell_moss.matrix.providers import MatrixLoggerProvider |
| script node 阻塞不退出 | main() 调用了 provide_channel | 去掉 provide_channel，main() return 即可 |
| moss.nodes run 提示未安装 | 缺少 .installed 标记 | 删除 INSTALL.md 表示无需安装，或跑 moss nodes install |

## 相关文档

```bash
moss --ai all-commands --group nodes                          # Node CLI 完整命令
moss codex get-interface ghoshell_container:IoCContainer      # contracts/providers/get_bound API
moss --ai manifests providers                                  # 当前 workspace 的 IoC 声明
moss codex blueprint matrix                                    # Matrix 装配与生命周期
moss howtos read debug-with-script-node.md                     # 本过程产出的 howto
```

| 时间 | 模型 | 备注 |
|------|------|------|
| 2026-07-26 | deepseek-v4-pro | 全链路走通：manifest 修复 → provider 改进 → script node 创建 → 运行时验证 LoggerItf 在容器中 |
