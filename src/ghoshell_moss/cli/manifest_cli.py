import typer
import json
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from ghoshell_moss.host.manifests.providers import (
    match_provider_infos,
    ProviderInfo
)

from ghoshell_moss.host.manifests.topics import (
    match_topic_infos,
    TopicInfo
)
from ghoshell_moss.host.manifests.configs import (
    ConfigInfo
)
from ghoshell_moss.host import Host
from ghoshell_common.helpers import generate_import_path
from .utils import console
import inspect

manifest_app = typer.Typer(
    help="MOSS Workspace Manifest Utilities. Handles environment discovery.",
    no_args_is_help=True
)


# TODO: MOSS CLI Discovery Utilities Optimization (by gemini 3)
# 1. [AI Optimization] 实现 --json 标志位。当检测到 AI 调用时，跳过 Rich 渲染，
#    直接输出纯净 JSON 以节省 Token 并避免格式解析错误。
# 2. [UX] 在所有 list 接口底部增加交互提示 (e.g., "Hint: Use 'moss-ctl <cmd> <name>' for detail")。
# 3. [Channel] 实现 Channel 详情页，补充运行时反射逻辑以获取 type(channel) 和所在模块路径。
# 4. [Command] 优化 Command 详情展示，优先暴露 meta().json_schema 和 __prompt__()，
#    确保 AI 能够根据输出直接构造合法的原语调用。
# 5. [Refactor] 抽象一个统一的 BaseDiscovery 类来处理 "匹配则显示详情，否则显示列表" 的分发逻辑。

@manifest_app.command(name="providers")
def list_providers(
        search: str = typer.Argument(
            "",
            help="Search pattern for ioc providers identity or provider path."
        ),
        mode: str | None = typer.Option(
            default=None,
            help="set specific mode"
        )
):
    """
    Explore and inspect providers discovered in the MOSS workspace.
    """
    host = Host(mode=mode)
    # 1. 执行发现逻辑
    # 默认从 MOSS.manifests.providers 扫描，这是我们在 Environment 中约定的路径
    all_providers = host.manifest.providers()

    # 2. 执行过滤逻辑
    results = list(match_provider_infos(all_providers, search)) if search else all_providers

    if search and not results:
        console.print(f"[yellow]No providers found matching: '{search}'[/yellow]")
        return

    # 3. 结果分发：唯一匹配显示详情，否则显示列表
    if search:
        if len(results) == 1:
            _display_provider_detail(results[0])
        else:
            _display_provider_table(results, is_filtered=bool(search))
    else:
        _display_provider_table(results, is_filtered=bool(search))


def _display_provider_table(providers: list[ProviderInfo], is_filtered: bool):
    """打印简洁的 Contract 列表"""
    title = "[bold cyan]Discovered MOSS providers[/bold cyan]"
    if is_filtered:
        title += " (Filtered)"

    table = Table(title=title, box=None, header_style="bold magenta")
    table.add_column("Identity", style="green", no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("Found At", style="blue")

    for info in providers:
        # 这里的 info.name 对应我们定义的 contract 类型导入路径
        # info.found 对应具体的 provider 实例化位置
        table.add_row(
            info.name,
            "Singleton" if info.singleton else "Factory",
            info.file
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(providers)} providers found.[/dim]")


def _display_provider_detail(info: ProviderInfo):
    """展示单个 Contract 的深度反射信息"""
    console.print(f"\n[bold cyan]Contract Detail:[/bold cyan] [green]{info.name}[/green]")
    console.print(f"[dim]Defined at: {info.file}[/dim]\n")

    # 打印 Docstring
    if info.docstring:
        console.print(f"[italic]{info.docstring}[/italic]\n")

    # 展示 Provider 及其配置（如果存在）
    console.print(f"[bold]Provider Instance:[/bold] {info.found}")
    console.print(f"[bold]Provider Type:[/bold] {info.provider_type}")

    # 核心：展示 Contract 的定义源码，让 AI 或开发者一目了然
    console.print("\n[bold]Contract Source Definition:[/bold]")
    syntax = Syntax(info.source, "python", theme="monokai", line_numbers=True)
    console.print(syntax)


@manifest_app.command(name="topics")
def list_topics(
        search: str = typer.Argument(
            "",
            help="Search pattern for topic name or topic type."
        ),
        mode: str | None = typer.Option(
            default=None,
            help="set specific mode"
        )
):
    """
    Introspect and discover event topics available in the MOSS ecosystem.
    """
    host = Host(mode=mode)
    # 1. 发现
    all_topics = host.manifest.topics()

    # 2. 过滤
    results = list(match_topic_infos(all_topics, search)) if search else list(all_topics.values())

    if search and not results:
        console.print(f"[yellow]No topics found matching: '{search}'[/yellow]")
        return

    # 3. 分发：唯一匹配显示 Schema 详情，否则显示列表
    if len(results) == 1 and search:
        _display_topic_detail(results[0])
    else:
        _display_topic_table(results, is_filtered=bool(search))


def _display_topic_table(topics: list[TopicInfo], is_filtered: bool):
    """展示 Topic 概览表"""
    title = "[bold magenta]MOSS Event Topics[/bold magenta]"
    if is_filtered:
        title += " (Filtered)"

    table = Table(title=title, box=None, header_style="bold cyan")
    table.add_column("Topic Name", style="green", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Description", style="dim", ratio=1)

    # 按照名称排序，方便模型阅读
    for info in sorted(topics, key=lambda x: x.name):
        table.add_row(
            info.name,
            info.type,
            info.description.split('\n')[0]  # 只取第一行描述
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(topics)} topics discovered.[/dim]")


def _display_topic_detail(info: TopicInfo):
    """展示 Topic 的深度定义和 JSON Schema，这是 AI 的“操作指南”"""
    console.print(f"\n[bold magenta]Topic Detail:[/bold magenta]")
    console.print(f"[dim]Name: {info.name}[/dim]")
    console.print(f"[dim]Type: {info.type}[/dim]")
    console.print(f"[dim]Found in: {info.found}[/dim]\n")

    # 1. 描述部分
    if info.description:
        console.print(Panel(info.description, title="Description", title_align="left", border_style="dim"))

    # 2. JSON Schema 部分 (模型最看重这个)
    console.print("\n[bold cyan]Payload JSON Schema:[/bold cyan]")
    schema_json = json.dumps(info.json_schema, indent=2, ensure_ascii=False)
    console.print(Syntax(schema_json, "json", theme="monokai", background_color="default"))

    # 3. 源码参考 (可选，如果模型想看具体的 Pydantic 逻辑)
    if info.model_source:
        console.print("\n[bold cyan]Python Model Definition:[/bold cyan]")
        console.print(Syntax(info.model_source, "python", theme="monokai", line_numbers=True))


@manifest_app.command(name="configs")
def list_configs(
        search: str = typer.Argument(
            "",
            help="Search pattern for config name."
        ),
        detail: bool = typer.Option(
            False, "--detail", "-d",
            help="Show detailed schema and default values."
        ),
        mode: str | None = typer.Option(
            default=None,
            help="set specific mode"
        )
):
    """
    Explore and manage environment configurations in MOSS.
    """
    host = Host(mode=mode)
    all_configs = host.manifest.configs()

    # 2. 匹配逻辑 (支持简单模糊匹配)
    results = [
        info for name, info in all_configs.items()
        if search.lower() in name.lower()
    ]

    if search and not results:
        console.print(f"[yellow]No configurations found matching: '{search}'[/yellow]")
        return

    # 3. 展示逻辑：唯一匹配或强制 detail 时显示详情
    if (len(results) == 1 and search) or detail:
        for info in results:
            _display_config_detail(info)
    else:
        _display_config_table(results)


def _display_config_table(configs: list[ConfigInfo]):
    """展示配置项全景图"""
    table = Table(title="[bold blue]MOSS Environment Configurations[/bold blue]", box=None)
    table.add_column("Config Name", style="green", no_wrap=True)
    table.add_column("Module Path", style="dim")
    table.add_column("Description", ratio=1)

    for info in sorted(configs, key=lambda x: x.name):
        table.add_row(
            info.name,
            info.found_import_path,
            info.description.split('\n')[0]
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(configs)} configuration definitions.[/dim]")


def _display_config_detail(info: ConfigInfo):
    """展示具体的配置契约与默认值"""
    console.print(f"\n[bold blue]Config Detail:[/bold blue] [green]{info.name}[/green]")
    console.print(f"[dim]Defined in: {info.found_at_file}[/dim]\n")
    console.print(f"[dim]ConfigType is: {info.model_path}[/dim]\n")

    # 1. 描述
    if info.description:
        console.print(Panel(info.description, title="Description", title_align="left", border_style="blue"))

    # 2. 默认值展示 (YAML 格式对模型非常友好)
    console.print("\n[bold cyan]Default Values (Seed):[/bold cyan]")
    console.print(Syntax(info.dump_yaml(), "yaml", theme="monokai", background_color="default"))

    # 3. JSON Schema (用于验证模型生成的配置是否合法)
    console.print("\n[bold cyan]Structure JSON Schema:[/bold cyan]")
    schema_json = json.dumps(info.schema.json_schema, indent=2, ensure_ascii=False)
    console.print(Syntax(schema_json, "json", theme="monokai", background_color="default"))

    # 4. 源码展示
    console.print("\n[bold cyan]Config Logic Source:[/bold cyan]")
    console.print(Syntax(info.source, "python", theme="monokai", line_numbers=True))
    console.print("-" * 40)


@manifest_app.command(name="channels")
def list_channels(
        search: str = typer.Argument("", help="Search pattern for channel name."),
        json_out: bool = typer.Option(False, "--json", help="Output as raw JSON for AI.")
):
    """
    List and inspect available communication channels.
    """
    host = Host()
    channels = host.manifest.channels()

    # 过滤
    results = {name: c for name, c in channels.items() if search.lower() in name.lower()}

    if json_out:
        # 给 AI 返回纯净数据
        data = {name: {"name": name, "desc": c.description(), "type": str(type(c))}
                for name, c in results.items()}
        console.print_json(data=data)
        return
    _display_channel_table(results, is_filtered=bool(search))


def _display_channel_table(channels: dict, is_filtered: bool):
    table = Table(title="MOSS Channels", box=None)
    table.add_column("Channel Name", style="green")
    table.add_column("Type", style="dim")
    table.add_column("Description", ratio=1)

    for name, c in channels.items():
        table.add_row(name, type(c).__name__, c.description().split('\n')[0])

    console.print(table)
    if not is_filtered:
        console.print("\n[dim]Hint: Use 'moss-cli channels <name>' to see full detail.[/dim]")


@manifest_app.command(name="primitives")
def list_primitives(
        search: str = typer.Argument("", help="Search pattern for command name."),
        json_out: bool = typer.Option(False, "--json", help="Output as raw JSON for AI.")
):
    """
    Explore MOSS Primitives (Commands).
    """
    host = Host()
    primitives = host.manifest.primitives()

    results = {name: cmd for name, cmd in primitives.items() if search.lower() in name.lower()}

    if json_out:
        # AI 模式只返回核心元数据和 Schema
        data = {name: {
            "name": cmd.meta().name,
            "description": cmd.meta().description,
            "params": cmd.meta().json_schema
        } for name, cmd in results.items()}
        console.print_json(data=data)
        return

    _display_command_detail(list(results.values())[0])


def _display_command_detail(cmd):
    meta = cmd.meta()
    console.print(f"\n[bold green]Command:[/bold green] {meta.name}")
    console.print(f"[dim]Dynamic: {cmd.is_dynamic()}[/dim]\n")

    # 重点展示接口定义
    console.print(Panel(cmd.meta().interface, title="Interface Prompt", border_style="yellow"))

    # 展示 JSON Schema
    console.print("\n[bold]Arguments Schema:[/bold]")
    console.print_json(data=meta.json_schema)


@manifest_app.command(name="contracts")
def list_contracts(
        search: str = typer.Argument("", help="Search pattern for contract name or module path."),
        json_out: bool = typer.Option(False, "--json", help="Output as raw JSON for AI.")
):
    """
    Introspect bound contracts in the MOSS IOC container.
    """
    host = Host()  # 根据需要传入 mode
    # 获取所有注册的 contracts
    all_contracts = list(host.matrix().container.contracts(recursively=True))
    all_contracts_info = []
    for contract in all_contracts:
        if not isinstance(contract, type):
            continue
        doc = inspect.getdoc(contract) or ''
        all_contracts_info.append(dict(
            name=contract.__name__,
            import_path=generate_import_path(contract),
            contract=contract,
            doc=doc,
            short_doc=doc.split('\n')[0],
        ))

    # 过滤
    results = [
        c for c in all_contracts_info
        if search.lower() in c['import_path'].lower()
    ]

    # 1. AI JSON 模式
    if json_out:
        data = {
            c['import_path']: {
                "name": c['name'],
                "doc": c['doc']
            } for c in results
        }
        console.print_json(data=data)
        return

    # 2. 唯一匹配显示详情，否则显示列表
    if len(results) == 1 and search:
        _display_contract_detail(results[0])
    else:
        _display_contract_table(results, is_filtered=bool(search))


def _display_contract_table(contracts: list, is_filtered: bool):
    table = Table(title="[bold yellow]MOSS Bound Contracts[/bold yellow]", box=None)
    table.add_column("Contract Name", style="green")
    table.add_column("Short Doc", style="italic")

    for c in sorted(contracts, key=lambda x: x['import_path']):
        table.add_row(c['import_path'], c['short_doc'])

    console.print(table)
    console.print(
        f"\n[dim]Total: {len(contracts)} contracts. Hint: Use 'moss-ctl contracts <name>' for source detail.[/dim]")


def _display_contract_detail(contract_info: dict):
    contract_type = contract_info['contract']
    console.print(f"\n[bold yellow]Contract:[/bold yellow] {contract_info['name']}")

    # 打印源码
    console.print("\n[bold]Source Code:[/bold]")
    try:
        source = inspect.getsource(contract_type)
        console.print(Syntax(source, "python", theme="monokai", line_numbers=True))
    except Exception as e:
        console.print(f"[red]Could not retrieve source: {e}[/red]")
