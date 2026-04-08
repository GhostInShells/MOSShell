from ghoshell_moss.core.blueprint.builder import new_channel, MutableChannel
from ghoshell_moss.message import Message
from typer import Typer


# defined by gemini 3 but not test yet.

def build_typer_skill_channel(
        name: str,
        typer_app: Typer,
        module_path: str,
        experience_path: str  # 指向那个存储“经验”的 markdown
) -> MutableChannel:
    chan = new_channel(name=name)

    # --- 1. 静态指令：定义 CLI 的边界 ---
    @chan.build.instruction
    def get_instruction():
        import typer.main
        group = typer.main.get_group(typer_app)

        # 遍历一级命令生成帮助手册
        help_text = f"You can operate the '{name}' system via CLI commands.\n"
        help_text += "Available sub-commands:\n"
        for cmd_name, cmd_obj in group.commands.items():
            help_text += f"- {cmd_name}: {cmd_obj.help or 'No description'}\n"

        help_text += f"\nUsage: Use the 'exec' command to run these. Example: exec(cmd='{list(group.commands.keys())[0]} --help')"
        return help_text

    # --- 2. 动态上下文：注入“经验” ---
    @chan.build.context_messages
    async def get_experience():
        # 这里读取你提到的 markdown 文件
        # 里面可以记录用户手动执行成功的案例，或者 AI 自己总结的坑
        try:
            with open(experience_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            content = "No experience recorded yet."

        return [
            Message.new_system(f"### Skill Experience ({name})\n{content}")
        ]

    # --- 3. 唯一的执行入口 ---
    @chan.build.command(
        name="exec",
        doc="Execute a CLI command within this skill context."
    )
    async def exec_command(cmd: str) -> str:
        """
        :param cmd: The full command string after 'moss'.
                    e.g. 'configs test'
        """
        import sys, asyncio
        # 模拟你在 console 里的逻辑
        full_cmd = [sys.executable, "-m", "typer", module_path, "run"] + cmd.split()

        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return stdout.decode() + stderr.decode()

    # --- 4. 经验修正命令 ---
    @chan.build.command(name="record_experience")
    async def record_experience(note: str):
        """Append new usage experience or tips to this skill."""
        with open(experience_path, 'a') as f:
            f.write(f"\n- {note}")
        return "Experience recorded."

    return chan
