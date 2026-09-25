# Control Node — Installation

独立 venv node。依赖 `ghoshell-moss[matrix]` (eclipse-zenoh) + `unitree-sdk2py` (SDK)。

## 1. 前置: Unitree SDK2 Python

SDK 不在版本控制中 (gitignored), 手动 clone 到 node 目录:

```bash
cd nodes/unitree/g1/control
mkdir -p src
git clone https://github.com/unitreerobotics/unitree_sdk2_python src/unitree_sdk2_python
```

cyclonedds 系统依赖 (Linux) 见 `../docs/moss-on-pc2.md` 装机日志。

## 2. 安装依赖

```bash
cd nodes/unitree/g1/control
uv sync
```

生成 `.venv` + `uv.lock`。`ghoshell-moss` 与 `unitree-sdk2py` 均 editable 引用 (pyproject.toml)。

## 3. 环境变量

```bash
export UNITREE_G1_NIC=eth0   # DDS 网卡, 见 ../docs/hardware.md
```

## 4. 标记安装

```bash
moss nodes install nodes/unitree/g1/control
```

## 5. 运行 (真机)

```bash
moss nodes run nodes/unitree/g1/control
```

macOS 不可测试 (cyclonedds 不编译)。等价代码在 G1 真机验证, 步骤见 `skills/verify/SKILL.md`。
