# MOSS Mode 开发

Mode 是叠加在全局 manifests 之上的能力视图，是 MOSS 最核心的隔离和复用机制。

## 核心概念

- **叠加语义**：mode 的 manifests 叠加在全局 manifests 之上
  - channels: mode 的 `__main__` 完全覆盖全局 main channel
  - 其余类型 (providers/configs/topics/nuclei/resources): 合并叠加
- **权限边界**：通过 MODE.md 的 `apps` 白名单控制哪些 app 可被访问
- **启动策略**：`bringup_apps` 声明启动时自动拉起哪些 app

## 目录约定

每个 mode 是 `src/MOSS/modes/<name>/` 下的一个 Python package，按子模块名约定各类声明：

| 文件 | 职责 |
|------|------|
| `MODE.md` | mode 入口，定义元数据和权限边界 |
| `channels.py` | 定义 `__main__` channel，覆盖或改造全局 main |
| `providers.py` | IoC Provider 声明 |
| `configs.py` | 配置模型声明 |
| `topics.py` | 事件协议声明 |
| `resources.py` | 资源存储声明 |
| `nuclei.py` | 感知核声明（Mindflow 输入信号源） |
| `contracts.py` | mode 专属契约绑定 |

所有文件都是可选的 — 按需添加。

## 创建 Mode

```bash
moss modes create <name> -a "group/*" -u "group/app" -d "description"
```

创建后编辑 `MODE.md` 配置权限边界，再按需添加 manifest 文件。

## 验证 Mode

```bash
moss modes show <name>                    # 查看 mode 详情和 manifest 文件清单
moss --mode <name> manifests explain       # 查看完整能力视图（全局 + mode 合并后）
moss --mode <name> manifests providers    # 查看可用的 IoC 绑定
moss --mode <name> manifests channels     # 查看 main channel 的命令树
```

## 深入路径

- 架构论述：`moss docs read workspace-and-mode.md`
- Mode 模型：`moss codex get-interface ghoshell_moss.core.blueprint.matrix:Mode`
- Manifests 体系：`moss manifests explain`
- 全局 manifests 约定：`src/MOSS/manifests/`
