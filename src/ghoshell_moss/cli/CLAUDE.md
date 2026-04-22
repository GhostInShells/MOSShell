# 关于 ghoshell_moss.cli

为 MOSS 开发开箱的命令行工具. 

# 开发指南

这个目录里的代码结构应该遵循 python 用 typer 开发脚本库的实现. 考虑: 

1. __main__.py 可以运行: 能够用 python -m ghoshell_moss.cli 运行相同的脚本.
2. 安装后可以用 `moss` 指令运行. 目前已经注册到根目录的 pyproject.toml 文件里. 
3. 基于 click group 分组实现命令. 在当前目录下, 每个文件为一个分组. 不过具体的实现可以放在 package 里. 
4. 使用英文来做代码的描述和注释. 人类协作者用中文写的说明, 考虑修改为英文. 