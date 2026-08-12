---
name: debug-with-script-node
description: 创建无 Channel 的 script node 用于运行时调试——检查 IoC 容器、网络状态、进程信息等 system-level 真相。
---

# How to Debug with Script Nodes

## 背景

MOSS 常规 Node 通过 Channel 暴露膜接口供模型调用。但有些场景不需要模型交互：

- 检查 IoC 容器里到底绑定了哪些 contract
- 打印当前 cell 网络里的所有 presence
- 扫描 runtime 目录里的 ledger 文件状态

这些是**一次性读操作**——启动、打印、退出。不需要 Channel，不需要 singleton lock，不需要 block。

## 核心模式

Script node = `main()` 收到已启动的 Matrix，直接读内部状态，打印完 return。Matrix 自动反卷退出。

```python
from ghoshell_moss.core.blueprint.matrix import Matrix

async def main(matrix: Matrix):
    # matrix 已经 bootstrapped，container / session / logger 全部可用
    print(matrix.container.contracts())  # 直接读
    # main 返回 → Matrix.__aexit__ → 进程退出

if __name__ == "__main__":
    Matrix.discover().run(main)
```

与常规 Node 的区别：

| | 常规 Node | Script Node |
|---|---|---|
| Channel | `matrix.provide_channel(chan)` 阻塞 | 无 |
| 生命周期 | 跑到 `close()` 或 SIGTERM | `main()` return 即退出 |
| singleton | 通常 yes（一个网络一份） | 不需要 |
| 典型用途 | 模型可调用的膜接口 | 人类/脚本一次性 debug |

## 第一步：创建 Node 骨架

```bash
moss nodes create .moss/system_test_nodes/<name>
```

删除 `INSTALL.md`（无外部依赖）和 `README.md`（不需要人类文档）。只留 `NODE.md` + `main.py`。

## 第二步：写 main()

`Matrix.discover().run(main)` 启动 Matrix 后调用 `main(matrix)`。此时 matrix 已经 `__aenter__`，container 已 bootstrap，所有 provider 已 resolve。可用的自省入口：

- `matrix.container.contracts()` — 所有 bound contract 类型
- `matrix.container.providers()` — 所有注册的 provider
- `matrix.container.get_bound(contract)` — 某个 contract 的绑定实例
- `matrix.handled_cells()` — 本 cell 拉起的子 cell
- `matrix.this` — 本 cell 的元信息
- `matrix.logger` — 当前 logger 实例

不要用 `matrix.provide_channel()`——那会阻塞住。

## 第三步：运行

```bash
moss nodes run .moss/system_test_nodes/<name>
```

Matrix 在子进程中启动，stdout 直接输出，进程在 `main()` 返回后自动退出。

## 相关文档

```bash
moss --ai all-commands --group nodes                # Node CLI 完整命令
moss codex get-interface ghoshell_container:IoCContainer  # contracts/providers/get_bound API
moss codex blueprint matrix                          # Matrix ABC 全接口
moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest  # NODE.md schema
```
