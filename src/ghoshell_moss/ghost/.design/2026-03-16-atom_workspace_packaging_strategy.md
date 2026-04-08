# Atom Workspace 打包策略设计

## 背景
Atom 原型需要作为可分发的工作空间模板，用户通过 `ghoshell atom init` 命令可以初始化独立的 Atom 实例。需要确定如何将 `.atom` workspace 原型打包到 Python 包中，以及如何让运行时代码访问这些资源。

## 核心决策
采用 **静态模板 + 动态 workspace** 的混合策略：
1. **静态模板**：使用 `importlib.resources` 管理打包到 Python 包中的 `.atom` 原型目录
2. **动态 workspace**：`AtomWorkspace` 类管理用户创建的运行时实例目录

### 决策理由
- **版本控制**：模板随包版本化，易于升级和维护
- **干净分离**：静态模板（包内）vs 运行时数据（用户目录）
- **多实例支持**：每个用户目录都是独立的 Atom 实例
- **开发友好**：模板可编辑，不影响已创建的实例

## 技术实现方案

### 1. 目录结构调整
```
ghoshell_atom/
├── templates/           # 新增模板目录
│   └── atom/           # 原 .atom 目录内容
│       ├── configs/
│       ├── assets/
│       ├── memory/
│       ├── meta/
│       ├── runtime/
│       └── src/Atom/
├── framework/          # 系统框架代码
└── cli/               # 命令行工具
```

### 2. 打包配置 (pyproject.toml)
```toml
[tool.setuptools.package-data]
"ghoshell_atom" = [
    "templates/**/*",
    "templates/atom/**/*",
]

# 或使用 include_package_data
[tool.setuptools]
include-package-data = true
```

### 3. 模板访问 API
```python
import importlib.resources

# Python 3.9+ 推荐方式
template_files = importlib.resources.files("ghoshell_atom.templates.atom")
config_template = template_files / "configs" / "models.yaml"

# 或使用 path() 上下文管理器
with importlib.resources.path("ghoshell_atom.templates", "atom") as template_path:
    # 复制模板到用户目录
    copy_template(template_path, target_dir)
```

### 4. AtomWorkspace 类增强
```python
class AtomWorkspace:
    @classmethod
    def init(cls, target_dir: Path) -> Self:
        """从包内模板初始化 workspace"""
        # 1. 获取模板
        template = importlib.resources.files("ghoshell_atom.templates.atom")

        # 2. 复制模板（支持变量替换、文件过滤）
        copy_template(template, target_dir)

        # 3. 创建运行时实例
        return cls(target_dir)

    # 运行时管理接口
    def assets(self) -> Path: ...
    def memory(self) -> Path: ...
    def configs(self) -> Path: ...
    def env_file(self) -> Path: ...
```

### 5. CLI 命令设计
```bash
# 初始化新实例
ghoshell atom init /path/to/my-atom

# 运行指定实例
ghoshell atom run /path/to/my-atom

# 进入目录后直接运行
cd /path/to/my-atom
ghoshell atom run

# 管理多个实例
ghoshell atom list
ghoshell atom stop /path/to/my-atom
```

## 未来扩展点

### 1. 模板变量替换
- 支持在初始化时替换模板中的变量（如实例名称、路径等）
- 基于 Jinja2 或字符串模板的变量系统

### 2. 模板版本管理
- 模板版本与包版本解耦
- 支持模板升级和迁移脚本
- 向后兼容性检查

### 3. 插件化模板
- 支持从外部源加载额外模板
- 模板市场或仓库概念
- 按需组合模板组件

### 4. Workspace 验证与修复
- 自动验证 workspace 结构完整性
- 修复工具：检测并修复损坏的配置
- 健康检查机制

### 5. 热重载支持
- 运行时检测配置变更并重载
- 安全的状态迁移机制
- 原子性更新保证

## 实施优先级
1. ✅ 目录结构调整（将 `.atom` 移至 `templates/atom`）
2. ⬜ 更新 pyproject.toml 打包配置
3. ⬜ 实现模板复制工具函数
4. ⬜ 增强 AtomWorkspace.init() 方法
5. ⬜ 更新 CLI 命令实现
6. ⬜ 添加测试用例

## 相关文件
- `src/ghoshell_atom/framework/workspace/abcd.py`：AtomWorkspace 类定义
- `src/ghoshell_atom/cli/workspace_utils.py`：CLI 工具函数
- `src/ghoshell_atom/cli/__main__.py`：CLI 入口

---
*设计记录创建于 2026-03-16，由 AI 协作者基于与人类工程师的讨论整理*