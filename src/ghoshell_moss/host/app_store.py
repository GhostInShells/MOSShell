import asyncio
import fnmatch
import configparser
import threading
from typing import Self, Iterable, Dict, Set, Optional

from ghoshell_moss.host.abcd.app import AppStore, AppInfo, AppState
from ghoshell_moss.host.environment import Environment
from ghoshell_moss.contracts import Workspace, LoggerItf, get_moss_logger
from pathlib import Path

from circus.arbiter import Arbiter
from circus.client import AsyncCircusClient

_AppAddress = str


def _is_match(address: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(address, p) for p in patterns)


class HostAppStore(AppStore):
    """
    HostAppStore 实现
    - 独占进程锁
    - 独立线程运行 Arbiter
    - 通过 AsyncCircusClient 异步管理子进程
    - 批量轮询状态
    """

    def __init__(
            self,
            env: Environment,
            workspace: Workspace,
            namespace: str,
            config_file: str = 'configs/circus.ini',
            app_store_name: str = "apps",
            runnable: bool = False,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._env_obj = env
        self._workspace_obj = workspace
        self._namespace = namespace
        self._name = app_store_name
        self._config_file_rel = config_file  # 相对路径，如 'configs/circus.ini'
        self._logger = logger or get_moss_logger()

        self.app_store_directory = self._workspace_obj.root_path().joinpath(app_store_name).resolve()
        self._sub_process_env = env.dump_moss_env()
        self._runnable = runnable
        # 状态维护
        self._found_apps: Dict[_AppAddress, AppInfo] | None = None
        self._managed_addresses: Set[_AppAddress] = set()
        self._include = include
        self._exclude = exclude or []

        # 锁与 Circus 组件
        self._lock = self._workspace_obj.lock(f"appstore-{self._namespace.replace('/', '-')}")
        self._arbiter: Optional[Arbiter] = None
        self._arbiter_thread: Optional[threading.Thread] = None
        self._client: Optional[AsyncCircusClient] = None
        self._polling_task: Optional[asyncio.Task] = None

        self._endpoint: str = ""
        self._pubsub_endpoint: str = ""
        self._is_running = False

    def _load_config(self) -> None:
        """从 Workspace 加载 Circus 配置"""
        config_path = self._workspace_obj.root_path().joinpath(self._config_file_rel)
        if not config_path.exists():
            # 默认兜底配置
            self._endpoint = "tcp://127.0.0.1:5555"
            self._pubsub_endpoint = "tcp://127.0.0.1:5556"
            return

        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        self._endpoint = cfg.get("circus", "endpoint", fallback="tcp://127.0.0.1:5555")
        self._pubsub_endpoint = cfg.get("circus", "pubsub_endpoint", fallback="tcp://127.0.0.1:5556")

    def name(self) -> str:
        return self._name

    def list_groups(self) -> list[str]:
        return list({app.group for app in self.list_apps()})

    def init_app(self, address: str, description: str = '') -> str:
        """
        创建一个 app, 返回创建后的讯息.
        1. 确保目录结构 apps/{group}/{name} 存在.
        2. 从 ghoshell_moss.host.app_stub 复制模板文件 (APP.md, main.py 等).
        3. 如果提供了 description, 更新 APP.md.
        """
        import shutil
        import importlib.util

        # 1. 规范化 address 并获取 group/name
        if address.startswith("app/"):
            parts = address.split('/')
            if len(parts) != 3:
                return f"Error: Invalid address format '{address}'. Expected 'app/group/name'."
            group, name = parts[1], parts[2]
        else:
            parts = address.split('/')
            if len(parts) != 2:
                return f"Error: Invalid address format '{address}'. Expected 'group/name'."
            group, name = parts[0], parts[1]

        # 2. 确定目标路径
        target_dir = self.app_store_directory.joinpath(group, name)
        if target_dir.exists():
            return f"Error: App directory already exists at {target_dir}"

        # 3. 寻找 stub 模板包的物理路径
        spec = importlib.util.find_spec("ghoshell_moss.host.app_stub")
        if not spec or not spec.origin:
            return "Error: Could not find template package 'ghoshell_moss.host.app_stub'"

        stub_dir = Path(spec.origin).parent

        try:
            # 4. 创建目标目录
            target_dir.mkdir(parents=True, exist_ok=True)

            # 5. 复制文件 (排除 __init__.py 和 __pycache__)
            for item in stub_dir.iterdir():
                if item.is_file() and item.name != "__init__.py" and item.suffix != ".pyc":
                    shutil.copy2(item, target_dir / item.name)

            # 6. 如果有描述，尝试更新 APP.md
            app_md_path = target_dir / "APP.md"
            if description and app_md_path.exists():
                # 我们采用简单的追加或者重写方式，这里假设 stub 里的 APP.md 是空的
                # 遵循之前定义的 AppInfo 格式，我们可以直接用 AppInfo 生成内容
                new_app_info = AppInfo(
                    name=name,
                    group=group,
                    description=description,
                    docstring=description,
                    work_directory=str(target_dir.absolute())
                )
                app_md_path.write_text(new_app_info.as_markdown(), encoding='utf-8')

            # 7. 刷新内存中的 app 列表
            self.list_apps(refresh=True)

            return f"Success: App '{address}' initialized at {target_dir}"

        except Exception as e:
            # 清理失败后的残留
            if target_dir.exists():
                shutil.rmtree(target_dir)
            self._logger.error(f"Failed to init app {address}: {e}")
            return f"Error: {e}"

    def found_apps(self, refresh: bool = False) -> dict[str, AppInfo]:
        if self._found_apps is None or refresh:
            discovered = AppInfo.from_apps_directory(self.app_store_directory)
            founds = self.match_apps(discovered, self._include, self._exclude)
            valid_apps = {}
            for app in founds:
                valid_apps[app.address] = app
            self._found_apps = valid_apps
        return self._found_apps

    def list_apps(self, refresh: bool = False) -> Iterable[AppInfo]:
        return self.found_apps().values()

    def get_app_info(self, address: str, running: bool = False) -> AppInfo | None:
        app = self.found_apps().get(address)
        if not app: return None
        if running and app.state != 'running': return None
        return app

    async def get_apps_context(self) -> str:
        apps = self.list_apps()
        if not apps: return "No apps discovered."

        lines = ["## Managed Apps Context"]
        for app in apps:
            state_str = f"[{app.state.upper()}]" if app.state else "[STOPPED]"
            lines.append(f"- **{app.address}**: {state_str} {app.description}")
            if app.error: lines.append(f"  > Error: {app.error}")
        return "\n".join(lines)

    async def start_app(self, app_address: str, argument: str = '') -> str:
        app = self.get_app_info(app_address)
        if not app: return f"Error: {app_address} not found."

        try:
            # 使用 to_circus_params 构造指令
            params = app.to_circus_params(self._sub_process_env, argument)

            # 1. 动态添加 Watcher
            await self._client.call({"command": "add", "properties": params})
            # 2. 显式启动
            await self._client.call({"command": "start", "name": app.address})

            self._managed_addresses.add(app.address)
            app.is_running = True
            app.state = 'starting'
            app.error = ''
            return f"Successfully issued start command for {app.address}."
        except Exception as e:
            app.error = str(e)
            return f"Failed to start {app_address}: {e}"

    async def stop_app(self, app_address: str) -> str:
        app = self.get_app_info(app_address)
        if not app or app.address not in self._managed_addresses:
            return f"App {app_address} is not under management."

        try:
            # 停止并移除，确保环境干净
            await self._client.call({"command": "rm", "name": app.address})
            self._managed_addresses.remove(app.address)
            app.is_running = False
            app.state = 'stopped'
            return f"Stopped and removed {app_address}."
        except Exception as e:
            return f"Error stopping {app_address}: {e}"

    def is_running(self) -> bool:
        return self._is_running

    async def _polling_loop(self) -> None:
        """全局状态批量查询"""
        while self._is_running:
            await asyncio.sleep(3)
            if not self._managed_addresses: continue

            try:
                # 获取所有 watcher 的状态快照
                # Circus 返回格式: {"statuses": {"app/g/n": "active", ...}, "status": "ok"}
                res = await self._client.call({"command": "status"})
                statuses = res.get("statuses", {})

                for addr in self._managed_addresses:
                    app = self.found_apps().get(addr)
                    if not app: continue

                    c_status = statuses.get(addr, "stopped")
                    if c_status == "active":
                        app.state = "running"
                    elif c_status == "stopped":
                        app.state = "stopped"
                    else:
                        app.state = "error"
            except Exception as e:
                self._logger.debug(f"Polling status failed: {e}")

    async def __aenter__(self) -> Self:
        if not self._runnable:
            raise RuntimeError(
                f'App Store setting is not not runnable'
            )
        if not self._lock.acquire(timeout=5):
            raise RuntimeError(f"Workspace {self._namespace} is locked by another Arbiter.")

        self._load_config()
        self.list_apps(refresh=True)

        # 1. 启动 Arbiter 线程
        self._arbiter = Arbiter(
            watchers=[],
            endpoint=self._endpoint,
            pubsub_endpoint=self._pubsub_endpoint,
            debug=False
        )
        self._arbiter_thread = threading.Thread(
            target=self._arbiter.start,
            name=f"Arbiter-{self._namespace}",
            daemon=True
        )
        self._arbiter_thread.start()

        # 2. 建立异步连接
        self._client = AsyncCircusClient(endpoint=self._endpoint)
        self._is_running = True

        # 3. 开启轮询任务
        self._polling_task = asyncio.create_task(self._polling_loop())

        # 4. 执行 Bring-up
        for addr in self._bring_up:
            asyncio.create_task(self.start_app(addr))

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        if self._polling_task:
            self._polling_task.cancel()

        if self._arbiter:
            self._arbiter.stop()

        if self._arbiter_thread:
            self._arbiter_thread.join(timeout=2)

        if self._client:
            self._client.stop()
        self._lock.release()
