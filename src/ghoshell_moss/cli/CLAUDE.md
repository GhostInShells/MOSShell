# 关于 ghoshell_cli

这个目录本应该是一个独立的代码仓库. 不过暂时先放入 ghoshell_moss 仓库中. 方便快速迭代.

ghoshell_cli 是整个 ghoshell 体系的命令行库. 它提供各个子库的调用工具, 和一些通用的工具.
也考虑用它来实现一些 Claude skills, 方便迭代. 

# 开发指南

这个目录里的代码结构应该遵循 python 用 click 开发脚本库的实现. 考虑: 

1. __main__.py 可以运行: 能够用 python -m ghoshell_cli 运行相同的脚本.
2. 安装后可以用 `ghoshell` 指令运行. 目前已经注册到根目录的 pyproject.toml 文件里. 
3. 基于 click group 分组实现命令. 在当前目录下, 每个文件为一个分组. 不过具体的实现可以放在 package 里. 
4. 使用英文来做代码的描述和注释. 人类协作者用中文写的说明, 考虑修改为英文. 