import typer
import json
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
from ghoshell_moss.host.manifests.resource_storages import (
    match_resource_storage_metas,
    ResourceStorageInfo,
)
from ghoshell_moss.host.manifests.nuclei import (
    match_nucleus_infos,
)
from ghoshell_moss.core.blueprint.manifests import NucleusMetaInfo
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.host import Host
from ghoshell_common.helpers import generate_import_path
from .utils import console, display_scan_errors, print_simple_table
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
        ),):
    """
    Explore and inspect providers discovered in the MOSS workspace.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    # 1. 执行发现逻辑
    # 默认从 MOSS.manifests.providers 扫描，这是我们在 Environment 中约定的路径
    all_providers = host.manifests.providers()

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
    title = "Discovered MOSS providers"
    if is_filtered:
        title += " (Filtered)"

    # 准备表格数据
    table_data = []
    for info in providers:
        table_data.append([
            f"[green]{info.name}[/green]",
            "Singleton" if info.singleton else "Factory",
            f"[blue]{info.file}[/blue]" if info.file else ""
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Identity", "Type", "Found At"],
        title=title,
        column_styles=["green", "dim", "blue"],
        title_style="bold cyan",
    )

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
        ),):
    """
    Introspect and discover event topics available in the MOSS ecosystem.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    # 1. 发现
    all_topics = host.manifests.topics()

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
    title = "MOSS Event Topics"
    if is_filtered:
        title += " (Filtered)"

    # 准备表格数据
    table_data = []
    for info in sorted(topics, key=lambda x: x.name):
        table_data.append([
            f"[green]{info.name}[/green]",
            f"[yellow]{info.type}[/yellow]",
            info.description.split('\n')[0] if info.description else ""
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Topic Name", "Type", "Description"],
        title=title,
        column_styles=["green", "yellow", "dim"],
        title_style="bold magenta",
    )

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
        ),):
    """
    Explore and manage environment configurations in MOSS.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    all_configs = host.manifests.configs()

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
    # 准备表格数据
    table_data = []
    for info in sorted(configs, key=lambda x: x.name):
        table_data.append([
            f"[green]{info.name}[/green]",
            f"[dim]{info.found_import_path}[/dim]",
            info.description.split('\n')[0] if info.description else ""
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Config Name", "Module Path", "Description"],
        title="MOSS Environment Configurations",
        column_styles=["green", "dim", ""],
        title_style="bold blue",
    )

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
        search: str = typer.Argument("", help="Search pattern for channel name."),json_out: bool = typer.Option(False, "--json", help="Output as raw JSON for AI.")
):
    """
    Inspect the __main__ channel discovered from MOSS manifests.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    channels = host.manifests.channels()
    main_channel = channels.get(Channel.MAIN_CHANNEL_NAME)

    if main_channel is None:
        console.print("[yellow]No __main__ channel discovered in manifests.[/yellow]")
        return

    if search and search.lower() not in "__main__":
        console.print(f"[yellow]Only __main__ channel exists. No match for: '{search}'[/yellow]")
        return

    # 发现位置
    source = getattr(host.manifests, 'main_channel_source', lambda: None)()
    found_at = source or "unknown"

    if json_out:
        data = {
            "name": "__main__",
            "type": str(type(main_channel)),
            "description": main_channel.description(),
            "found_module": found_at,
        }
        console.json(data=data)
        return
    _display_main_channel_detail(main_channel, found_at)


def _display_main_channel_detail(channel, found_module: str):
    """展示 __main__ channel 的详情视图。"""
    console.print(f"\n[bold cyan]__main__ Channel[/bold cyan]")
    console.print(f"[dim]Type:[/dim] {type(channel).__name__}")
    console.print(f"[dim]Description:[/dim] {channel.description() or '(none)'}")
    console.print(f"[dim]Discovered at:[/dim] {found_module}")


@manifest_app.command(name="contracts")
def list_contracts(
        search: str = typer.Argument("", help="Search pattern for contract name or module path."),json_out: bool = typer.Option(False, "--json", help="Output as raw JSON for AI.")
):
    """
    Introspect bound contracts in the MOSS IOC container.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
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
        console.json(data=data)
        return

    # 2. 唯一匹配显示详情，否则显示列表
    if len(results) == 1 and search:
        _display_contract_detail(results[0])
    else:
        _display_contract_table(results, is_filtered=bool(search))


def _display_contract_table(contracts: list, is_filtered: bool):
    # 准备表格数据
    table_data = []
    for c in sorted(contracts, key=lambda x: x['import_path']):
        table_data.append([
            f"[green]{c['import_path']}[/green]",
            c['short_doc'] or ""
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Contract Name", "Short Doc"],
        title="MOSS Bound Contracts",
        column_styles=["green", "italic"],
        title_style="bold yellow",
    )

    console.print(
        f"\n[dim]Total: {len(contracts)} contracts. Hint: Use [bold]moss manifest contracts <name>[/bold] for source detail.[/dim]")


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


@manifest_app.command(name="ctml-versions")
def list_ctml_versions():
    """
    list the environment provided ctml versions.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    for version, version_info in host.manifests.ctml_versions().items():
        console.print("%s: %s" % (version, version_info.file))


@manifest_app.command(name="resources")
def list_resources(
        search: str = typer.Argument(
            "",
            help="Search pattern for storage scheme, host, or description."
        ),json_out: bool = typer.Option(
            False, "--json",
            help="Output as raw JSON for AI."
        ),
):
    """
    List discovered ResourceStorageMeta instances in the environment.

    Each entry describes a ResourceStorage available for registration:
    scheme, host, description, and where it was found.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    resource_storage_items = host.manifests.resource_storage_manifests()

    # collect metas
    all_metas = [item.info for item in resource_storage_items]

    if search:
        metas_dict = {m.path: m for m in all_metas}
        results = match_resource_storage_metas(metas_dict, search)
    else:
        results = all_metas

    if json_out:
        data = [
            {
                "locator": m.locator,
                "storage_scheme": m.storage_scheme,
                "storage_host": m.storage_host,
                "description": m.description,
                "found_module": m.found_module,
            }
            for m in results
        ]
        import json as _json
        console.print(_json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not results:
        console.print(f"[yellow]No resource storages found matching: '{search}'[/yellow]")
        return

    _display_resource_storage_table(results)


def _display_resource_storage_table(metas: list[ResourceStorageInfo]):
    """展示发现的 ResourceStorage Meta"""
    table_data = []
    for info in sorted(metas, key=lambda x: x.path):
        table_data.append([
            f"[green]{info.storage_scheme}[/green]",
            f"[yellow]{info.storage_host}[/yellow]",
            info.description.split('\n')[0] if info.description else "",
            f"[dim]{info.found_module}[/dim]",
        ])

    print_simple_table(
        data=table_data,
        headers=["Scheme", "Host", "Description", "Found At"],
        title="Discovered Resource Storages",
        column_styles=["green", "yellow", "", "dim"],
        title_style="bold cyan",
    )

    console.print(f"\n[dim]Total: {len(metas)} resource storages discovered.[/dim]")


@manifest_app.command(name="nuclei")
def list_nuclei(
        search: str = typer.Argument(
            "",
            help="Search pattern for nucleus name, description, or signal."
        ),json_out: bool = typer.Option(
            False, "--json",
            help="Output as raw JSON for AI."
        ),
):
    """
    List discovered NucleusFactory instances in the environment.

    Each entry describes a NucleusFactory available for Ghost mindflow:
    name, description, signal_names, and where it was found.
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    all_nuclei = host.manifests.nuclei()

    if search:
        results = match_nucleus_infos(all_nuclei, search)
    else:
        results = list(all_nuclei.values())

    if json_out:
        data = [
            {
                "name": m.name,
                "description": m.description,
                "signal_names": m.signal_names,
                "found_module": m.found_module,
            }
            for m in results
        ]
        import json as _json
        console.print(_json.dumps(data, ensure_ascii=False, indent=2))
        return

    if not results:
        console.print(f"[yellow]No nucleus factories found matching: '{search}'[/yellow]")
        return

    _display_nuclei_table(results)


def _display_nuclei_table(metas: list[NucleusMetaInfo]):
    """展示发现的 NucleusFactory"""
    table_data = []
    for info in sorted(metas, key=lambda x: x.name):
        table_data.append([
            f"[green]{info.name}[/green]",
            info.description.split('\n')[0] if info.description else "",
            f"[yellow]{', '.join(info.signal_names)}[/yellow]",
            f"[dim]{info.found_module}[/dim]",
        ])

    print_simple_table(
        data=table_data,
        headers=["Name", "Description", "Signals", "Found At"],
        title="Discovered Nucleus Factories",
        column_styles=["green", "", "yellow", "dim"],
        title_style="bold cyan",
    )

    console.print(f"\n[dim]Total: {len(metas)} nucleus factories discovered.[/dim]")


@manifest_app.command(name="explain")
def explain_manifests():
    """
    用自然语言自描述当前环境 manifest 的结构与含义。
    这是 manifest 体系的唯一真相入口。
    """
    host = Host()
    display_scan_errors(host.scan_errors)
    explanation = host.manifests.explain()
    console.print(explanation)
