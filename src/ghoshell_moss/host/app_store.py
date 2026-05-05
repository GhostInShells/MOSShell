import asyncio
import configparser
import os
import subprocess
import shutil
import importlib.util
from typing import Iterable, Dict, Set, Optional
from typing_extensions import Self
from pathlib import Path

from ghoshell_moss.core.concepts.errors import CommandErrorCode, CommandError
from ghoshell_moss.host.abcd.app import AppStore, AppInfo, AppState
from ghoshell_moss.host.abcd.environment import Environment
from ghoshell_moss.contracts import Workspace, LoggerItf, get_moss_logger
from circus.client import CircusClient
import sys

_AppAddress = str
_AppFullname = str


class HostAppStore(AppStore):
    """
    HostAppStore 实现 (方案一: 外部进程解耦版)
    - 独占进程锁
    - 使用 subprocess.Popen 启动独立的 circusd 进程，避开信号冲突
    - 通过 AsyncCircusClient 异步管理子进程
    - 批量轮询状态
    """

    def __init__(
            self,
            env: Environment,
            workspace: Workspace,
            namespace: str = 'MOSS/app_store',
            config_file: str = 'configs/circus.ini',
            app_store_name: str = "apps",
            runnable: bool = False,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            bringup: list[str] | None = None,
            logger: LoggerItf | None = None,
    ) -> None:
        self._env_obj = env
        self._workspace_obj = workspace
        self._namespace = namespace
        self._name = app_store_name
        self._config_file_rel = config_file
        self._logger = logger or get_moss_logger()

        self.app_store_directory = self._workspace_obj.root_path().joinpath(app_store_name).resolve()
        self._runnable = runnable
        self._bringup = bringup or []
        self._app_states: dict[str, AppState] = {}

        # 状态维护
        self._found_apps: Dict[_AppFullname, AppInfo] | None = None
        self._managed_apps_with_fullname: Set[_AppFullname] = set()
        self._include = include
        self._exclude = exclude or []

        # 锁与 Circus 外部进程
        self._lock = self._workspace_obj.lock(f"appstore-{self._namespace.replace('/', '-')}")
        self._circus_process: Optional[subprocess.Popen] = None
        # self._client: Optional[AsyncCircusClient] = None
        self._polling_task: Optional[asyncio.Task] = None

        self._endpoint: str = ""
        self._pubsub_endpoint: str = ""
        self._is_running = False
        self._log_prefix = f"<HostAppStore {self._namespace}>"

    def with_logger(self, logger: LoggerItf) -> Self:
        self._logger = logger
        return self

    def _ensure_config(self) -> str:
        """确保 Circus 配置存在，返回绝对路径"""
        config_path = self._workspace_obj.root_path().joinpath(self._config_file_rel)
        if not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)

        if not config_path.exists():
            # 自动生成默认配置
            cfg = configparser.ConfigParser()
            cfg.add_section("circus")
            cfg.set("circus", "endpoint", "tcp://127.0.0.1:20771")
            cfg.set("circus", "pubsub_endpoint", "tcp://127.0.0.1:20772")
            cfg.set("circus", "check_delay", "1")
            with open(config_path, "w") as f:
                cfg.write(f)

        # 加载 endpoint 用于 Client 连接
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        self._endpoint = cfg.get("circus", "endpoint", fallback="tcp://127.0.0.1:20771")
        self._pubsub_endpoint = cfg.get("circus", "pubsub_endpoint", fallback="tcp://127.0.0.1:20772")

        return str(config_path.absolute())

    def name(self) -> str:
        return self._name

    def list_groups(self) -> list[str]:
        return list({app.group for app in self.list_apps()})

    def init_app(self, fullname: str, description: str = '') -> str:
        """创建 App 模板目录逻辑 (保持不变)"""
        if fullname.startswith("apps/"):
            parts = fullname.split('/')
            if len(parts) != 3: return f"Error: Invalid address '{fullname}'"
            group, name = parts[1], parts[2]
        else:
            parts = fullname.split('/')
            if len(parts) != 2: return f"Error: Invalid address '{fullname}'"
            group, name = parts[0], parts[1]

        target_dir = self.app_store_directory.joinpath(group, name)
        if target_dir.exists(): return f"Error: Exists at {target_dir}"

        spec = importlib.util.find_spec("ghoshell_moss.host.app_stub")
        if not spec or not spec.origin: return "Error: Stub not found"
        stub_dir = Path(spec.origin).parent

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in stub_dir.iterdir():
                if item.is_file() and item.name != "__init__.py" and item.suffix != ".pyc":
                    shutil.copy2(item, target_dir / item.name)

            app_md_path = target_dir / "APP.md"
            if description and app_md_path.exists():
                new_app_info = AppInfo(name=name, group=group, description=description,
                                       docstring=description, work_directory=str(target_dir.absolute()))
                app_md_path.write_text(new_app_info.as_markdown(), encoding='utf-8')

            self.list_apps(refresh=True)
            return f"Success: App '{fullname}' initialized."
        except Exception as e:
            if target_dir.exists(): shutil.rmtree(target_dir)
            return f"Error: {e}"

    def found_apps(self, refresh: bool = False) -> dict[_AppFullname, AppInfo]:
        if self._found_apps is None or refresh:
            discovered = AppInfo.from_apps_directory(self.app_store_directory)
            founds = self.match_apps(discovered, self._include, self._exclude)
            valid_apps = {app.fullname: app for app in founds}
            self._found_apps = valid_apps
        return self._found_apps

    def list_apps(self, refresh: bool = False) -> Iterable[AppInfo]:
        for app in self.found_apps(refresh).values():
            app.state = self._get_app_state(app.fullname)
            yield app

    def get_app_info(self, fullname: str) -> AppInfo | None:
        app = self.found_apps().get(fullname)
        if not app: return None
        app.state = self._get_app_state(fullname)
        return app

    def get_app_executable(self, fullname: str, args: str | None = None) -> Optional[tuple[str, list[str]]]:
        app = self.get_app_info(fullname)
        if not app: return None
        return self._get_app_executable(app, args)

    def _get_app_script(self, app: AppInfo) -> str:
        return str(Path(app.work_directory).joinpath(app.watcher.script).absolute())

    def _get_app_executable(
            self,
            app: AppInfo,
            arguments: Optional[str] = None,
    ) -> tuple[str, list[str]]:
        # 1. 拆分原始命令。例如 'uv run main.py' -> ['uv', 'run', 'main.py']
        args_list = []
        executable = app.watcher.executable
        if executable == 'uv':
            executable, uv_arguments = get_uv_executable()
            if uv_arguments:
                args_list.extend(uv_arguments)
            args_list.append('run')
        else:
            full_path = shutil.which(executable)
            if full_path:
                executable = full_path
            else:
                # 如果找不到，可以尝试从系统环境变量里捞一下，或者报错
                self._logger.warning(f"Could not find executable {executable} in PATH")
        args_list.append(self._get_app_script(app))

        # 2. 组合参数列表。原始参数 + 传入的 arguments
        arguments = arguments if arguments is not None else app.watcher.arguments
        if arguments:
            args_list.extend(arguments.split())
        return executable, args_list

    def _set_app_state(self, fullname: str, state: AppState) -> None:
        self._app_states[fullname] = state

    def _get_app_state(self, fullname: str) -> str:
        return self._app_states.get(fullname, AppState.STOPPED)

    def _app_to_circus_params(self, app: AppInfo, env: dict[str, str], arguments: str | None = None) -> dict:
        """
        修正后的参数构造
        """
        executable, args_list = self._get_app_executable(app, arguments)
        options = {
            "working_dir": app.work_directory,
            "numprocesses": app.watcher.workers,
            "respawn": app.watcher.respawn,
            "max_age": app.watcher.max_age,
            "env": env,
            "singleton": True,
            "copy_env": True,
        }
        options = {k: v for k, v in options.items() if v is not None}

        return {
            "name": app.address,
            "cmd": executable,  # 仅包含可执行程序名
            "args": [self._get_app_script(app)],  # 参数列表
            "options": options,
        }

    async def start_app(self, app_fullname: str, argument: str = '') -> str:
        app = self.get_app_info(app_fullname)
        if not app: return f"Error: {app_fullname} not found."

        try:
            # 构造参数
            params = self._app_to_circus_params(
                app,
                self._env_obj.dump_moss_env(for_child_process=True, with_os_env=False, cell_address=app.address),
                argument,
            )
            app_runtime_logs_dir = Path(app.work_directory).joinpath("runtime").joinpath("logs").resolve()
            if not app_runtime_logs_dir.exists():
                app_runtime_logs_dir.mkdir(parents=True, exist_ok=True)
            app_stdout_log = app_runtime_logs_dir.joinpath("stdout.log")
            app_stderr_log = app_runtime_logs_dir.joinpath("stderr.log")
            rotation_config = {
                "max_bytes": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,  # 保持最近 5 个旧日志文件
                "time_format": "%Y-%m-%d %H:%M:%S",  # 如果 FileStream 支持在行首加时间戳
            }
            params['options']['stdout_stream'] = {
                "class": "FileStream",
                "filename": str(app_stdout_log.resolve().absolute()),
                **rotation_config,
            }
            params['options']["stderr_stream"] = {
                "class": "FileStream",
                "filename": str(app_stderr_log.resolve().absolute()),
                **rotation_config,
            }
            if app_fullname not in self._managed_apps_with_fullname:
                r1 = await self._call_circus({"command": "add", "properties": params})
                if r1['status'] == "error":
                    self._logger.error(
                        "%s failed to start app %s on error: %s",
                        self._log_prefix, app_fullname, r1,
                    )
                    raise CommandErrorCode.VALUE_ERROR.error(f"failed to start {app_fullname}")

            self._managed_apps_with_fullname.add(app_fullname)
            r2 = await self._call_circus({"command": "start", "name": app.address})
            if r2['status'] == "error":
                self._logger.error(
                    "%s failed to start app %s on error: %s",
                    self._log_prefix, app_fullname, r2,
                )
                raise CommandErrorCode.VALUE_ERROR.error(f"failed to start {app_fullname}")
            self._logger.info("%s start app %s: %s, %s", self._log_prefix, app_fullname, r1, r2)

            self._set_app_state(app_fullname, AppState.STARTING)
            return f"Successfully started {app.address} via Circus Daemon."
        except CommandError as e:
            app.error = str(e)
            raise
        except Exception as e:
            app.error = str(e)
            self._set_app_state(app_fullname, AppState.ERROR)
            raise CommandErrorCode.VALUE_ERROR.error(f"failed to start {app_fullname}")

    async def stop_app(self, app_fullname: str) -> str:
        app = self.get_app_info(app_fullname)
        if not app or app.address not in self._managed_apps_with_fullname:
            return f"App {app_fullname} is not under management."
        try:
            await self._call_circus({"command": "rm", "name": app.address})
            self._managed_apps_with_fullname.remove(app.address)
            self._set_app_state(app_fullname, AppState.STOPPED)
            return f"Stopped and removed {app_fullname}."
        except Exception as e:
            return f"Error stopping {app_fullname}: {e}"

    async def _polling_loop(self) -> None:
        while self._is_running:
            await asyncio.sleep(2)
            if not self._managed_apps_with_fullname: continue
            try:
                res = await self._call_circus({"command": "status"})
                statuses = res.get("statuses", {})
                for fullname in self._managed_apps_with_fullname:
                    app = self.found_apps().get(fullname)
                    if not app: continue
                    c_status = statuses.get(fullname, "stopped")
                    self._set_app_state(fullname, AppState.RUNNING if c_status == "active" else AppState.STOPPED)
            except Exception as e:
                self._logger.debug(f"Polling failed: {e}")

    def is_running(self) -> bool:
        return self._is_running

    async def _call_circus(self, command: dict) -> dict:
        """在后台线程执行同步的 ZMQ 调用，彻底隔离 Tornado/uvloop 冲突"""
        if not self._client:
            return {}
        # 抛入 asyncio 的默认线程池运行，完美兼容 uvloop
        return await asyncio.to_thread(self._client.call, command)

    async def __aenter__(self) -> Self:
        if not self._runnable: raise RuntimeError('AppStore is not runnable')
        if not self._lock.acquire(timeout=5):
            raise RuntimeError(f"Namespace {self._namespace} is locked.")

        # 1. 准备配置并启动外部进程
        config_path = self._ensure_config()
        self._logger.info(f"{self._log_prefix} Launching circusd process...")

        # 使用 subprocess.Popen 启动独立进程，不使用 shell 以便更安全地管理 PID
        log_dir = self._env_obj.workspace_path.joinpath("runtime/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir.joinpath("circusd.log")
        if not log_file_path.exists():
            log_file_path.touch(mode=0o640)

        # 2. 显式以追加模式打开文件
        # 使用 buffering=1 实现行缓冲，或者不传，让系统决定
        # 注意：'a' 模式最安全，多个进程同时写（虽然这里只有 circusd 写）不会互相覆盖
        self._circus_log_file = open(log_file_path, mode="a", encoding="utf-8")

        # 3. 修正权限（如果需要强制 770）
        os.chmod(log_file_path, 0o770)
        python_executable = sys.executable
        self._circus_process = subprocess.Popen(
            [python_executable, "-m", "circus.circusd", config_path],
            stdout=self._circus_log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy()
        )

        # 2. 等待 ZMQ 端口就绪 (重试逻辑)
        # 2. 建立同步连接 (设置一个合理的超时时间防止卡死线程)
        self._client = CircusClient(endpoint=self._endpoint, timeout=2.0)

        connected = False
        for _ in range(10):
            try:
                # 使用包装好的异步方法
                res = await self._call_circus({"command": "list"})
                if res.get("status") == "ok":
                    connected = True
                    break
            except Exception:
                await asyncio.sleep(0.5)

        if not connected:
            self._circus_process.kill()
            raise RuntimeError("Failed to connect to circusd after launch.")

        self._is_running = True
        self.list_apps(refresh=True)
        self._polling_task = asyncio.create_task(self._polling_loop())

        # 3. Bring-up
        bringup_apps_cors = []
        if self._bringup:
            for app_info in self.match_apps(self.list_apps(), self._bringup):
                bringup_apps_cors.append(self.start_app(app_info.fullname))
        if len(bringup_apps_cors) > 0:
            _ = await asyncio.gather(*bringup_apps_cors, return_exceptions=False)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        if self._polling_task:
            self._polling_task.cancel()

        if self._client:
            # 使用包装好的方法发退出指令
            try:
                await asyncio.wait_for(self._call_circus({"command": "quit"}), timeout=2.0)
            except:
                pass
            self._client.stop()

        # 强制确保外部进程结束，防止僵尸进程
        if self._circus_process:
            if self._circus_process.poll() is None:
                self._circus_process.terminate()
                try:
                    self._circus_process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._circus_process.kill()
            self._logger.info(f"{self._log_prefix} circusd process reaped.")

        self._lock.release()

    async def get_apps_context(self) -> str:
        apps = self.list_apps()
        if not apps: return "No apps discovered."
        lines = ["## Managed Apps Context"]
        for app in apps:
            state_str = f"[{app.state.upper()}]" if app.state else "[STOPPED]"
            lines.append(f"- `{app.fullname}`: {state_str} {app.description}")
        return "\n".join(lines)


_Executable = str
_ExecutableArguments = list[str]


def get_uv_executable() -> tuple[_Executable, _ExecutableArguments]:
    # 方案 A: 检查 uv 是否作为一个 python 模块存在
    # 很多现代工具支持 python -m uv ...
    try:
        import uv
        return f"{sys.executable}", ['-m', 'uv']  # 这种方式最能绕过 PATH 问题
    except ImportError:
        pass
    # 方案 B: 检查 Python 所在的 bin/Scripts 目录
    # 如果是在 venv 里 pip install uv，uv 就在这里
    python_bin_dir = Path(sys.executable).parent
    uv_in_venv = shutil.which("uv", path=str(python_bin_dir))
    if uv_in_venv:
        return uv_in_venv, []

    # 方案 C: 回退到全局搜索，但排除 pyenv shims
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv, []

    return "uv", []  # 最后的保底
