# 关于 Examples

本目录用来存放各种 Alpha 版本的测试用例. 用来展示不同的基线功能.
每个子目录内都有 README.md 提示如何使用.

> 建议使用 mac, 基线都是在 mac 上测试的. windows 可能兼容存在问题. 
 
使用 examples 的步骤: 

## 1. clone 仓库

```bash
git clone https://github.com/GhostInShells/MOSShell MOSShell
cd MOSShell
```

## 2. 创建环境

* 使用 `uv` 创建环境, 运行 `uv venv` . 由于依赖 live2d, 所以默认的 python 版本是 3.12
* 进入 uv 的环境: `source .venv/bin/activate`
* 安装所有依赖: 

```bash
uv sync --active --all-extras
```

## 3. 配置环境变量

todo

## 4. 运行个别例子或全部 

todo