import os
from typing import List
from ghoshell_container import Provider, Container, set_container, get_container

from ghoshell_common.contracts import LocalWorkspaceProvider, LoggerItf, WorkspaceConfigsProvider, Workspace
from ghoshell_moss.core import Speech
from pathlib import Path
from contextlib import contextmanager
import logging

__all__ = [
    'get_container', 'set_container',
    'init_container', 'workspace_container',
    'get_example_speech',
]


def setup_simple_logger(log_file: str) -> logging.Logger:
    """设置简单的文件日志记录器"""
    # 创建日志器
    logger = logging.getLogger("mosshell")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建文件handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 设置格式（包含文件名和行号）
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # 添加到日志器
    logger.addHandler(file_handler)

    return logger


def get_example_speech(
        container: Container | None = None,
        default_speaker: str | None= None,
) -> Speech:
    """
    直接初始化音频模块.
    目前应该只在 mac 上比较好用.
    TODO:
        还有许多工作量, 需要把默认的服务选项配到 workspace 里才对.
        而且通过 provider 的方式注册单例.
    """
    from ghoshell_moss.speech import TTSSpeech
    from ghoshell_moss.speech.mock import MockSpeech
    from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
    from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf

    container = container or get_container()
    use_voice = os.environ.get('USE_VOICE_SPEECH', 'no') == 'yes'
    if not use_voice:
        return MockSpeech()
    app_key = os.environ.get("VOLCENGINE_STREAM_TTS_APP")
    app_token = os.environ.get("VOLCENGINE_STREAM_TTS_ACCESS_TOKEN")
    resource_id = os.environ.get("VOLCENGINE_STREAM_TTS_RESOURCE_ID", 'seed-tts-2.0')
    if not app_key or not app_token:
        raise NotImplementedError(
            "Env $VOLCENGINE_STREAM_TTS_APP or $VOLCENGINE_STREAM_TTS_ACCESS_TOKEN not configured."
            "Maybe examples/.env not set, or you need to set $USE_VOICE_SPEECH `no`"
        )
    tts_conf = VolcengineTTSConf(
        app_key=app_key,
        access_token=app_token,
        resource_id=resource_id,
    )
    if default_speaker:
        tts_conf.default_speaker = default_speaker
    return TTSSpeech(
        player=PyAudioStreamPlayer(),
        tts=VolcengineTTS(conf=tts_conf),
        logger=container.get(LoggerItf)
    )


def init_container(
        workspace_dir: Path | str,
        name: str = "moss",
        providers: List[Provider] | None = None,
        env_path: Path | None = None,
) -> Container:
    if isinstance(workspace_dir, str):
        workspace_dir = Path(workspace_dir).absolute()

    env_path = env_path or workspace_dir.parent.joinpath('.env').resolve()
    # 加载环境变量, .env 文件默认和 workspace 同层.
    if env_path.exists():
        import dotenv
        dotenv.load_dotenv(dotenv_path=env_path, override=True, verbose=True)

    container = Container(name=name)
    # 注册 workspace
    container.register(LocalWorkspaceProvider(
        workspace_dir=str(workspace_dir.absolute()),
    ))
    container.register(WorkspaceConfigsProvider())

    # 初始化一个简单的日志.
    logger = setup_simple_logger(
        str(workspace_dir.joinpath('runtime/logs/moss_demo.log').absolute()),
    )
    container.set(LoggerItf, logger)

    # 注册更多 providers.
    if providers:
        for provider in providers:
            container.register(provider)
    return container


@contextmanager
def workspace_container(
        workspace_dir: Path | str,
        name: str = "moss",
        providers: List[Provider] | None = None,
):
    """
    支持 with statement 的全局 container 初始化.

    >>> with workspace_container(workspace_dir=Path('workspace')) as container:
    >>>     pass
    """

    container = init_container(
        workspace_dir=workspace_dir,
        name=name,
        providers=providers,
    )
    # 设置到全局.
    set_container(container)
    # 初始化启动.
    container.bootstrap()
    yield container
    container.shutdown()
