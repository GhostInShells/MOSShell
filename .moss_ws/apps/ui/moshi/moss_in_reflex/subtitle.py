"""字幕子系统：Topic 总线消费 + HTTP 旁路兼容。

提供:
- new_subtitle_callback(): HTTP 旁路回调（fire-and-forget POST → :9733/_internal/subtitle_in）
- setup_subtitle(): 三层回退 → TopicWindow 消费协程 或 HTTP 旁路
"""

import asyncio
import importlib

import aiohttp

from ghoshell_moss.core.speech.subtitle_config import SubtitleTopicConfig
from ghoshell_moss.topics.audio import SubtitleTopic
from ghoshell_moss.contracts.speech import Speech
from ghoshell_moss.core.blueprint.environment import Environment

from moss_in_reflex.state import (
    logger,
    _SUBTITLE_LOCK, _SUBTITLE_QUEUE, _SUBTITLE_EVENT,
)


def new_subtitle_callback():
    """创建句级字幕回调函数（fire-and-forget HTTP POST 到内部桥接）。

    可用于：
    - 同进程：moss() 中注入 matrix.container 的 Speech
    - 跨进程：Ghost 运行时设置到其 Speech 实例

    回调签名：(text: str, is_final: bool, batch_id: str = "") -> None
    """
    def _post(text: str, is_final: bool, batch_id: str = "") -> None:
        async def _do():
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        "http://127.0.0.1:9733/_internal/subtitle_in",
                        json={"text": text, "is_final": is_final},
                        timeout=aiohttp.ClientTimeout(total=2),
                    )
            except Exception:
                pass  # 字幕丢失非关键故障

        try:
            asyncio.get_running_loop().create_task(_do())
        except RuntimeError:
            pass

    return _post


async def setup_subtitle(matrix, config_store) -> bool:
    """配置字幕消费链路。

    三层回退：
    1. ConfigStore.get(SubtitleTopicConfig) — 读 mode 覆盖配置
    2. Mode config 模块回退导入 — 直读 MOSS.modes.<mode>.configs
    3. HTTP 旁路 — 同进程 Speech 注入（仅同进程有效）

    Returns:
        True if Topic 路径已启用（TopicWindow 创建 + 消费协程已启动），
        False if HTTP 旁路（同进程兼容路径）。
    """
    subtitle_config = None

    # Layer 1: ConfigStore
    try:
        subtitle_config = config_store.get(SubtitleTopicConfig)
        logger.info("[subtitle] SubtitleTopicConfig from ConfigStore: enable_topic=%s, topic_path=%s",
                    subtitle_config.enable_topic, subtitle_config.topic_path)
    except Exception as e:
        logger.warning("[subtitle] ConfigStore.get(SubtitleTopicConfig) 异常: %s (%s)", e, type(e).__name__)

    # Layer 2: Mode config 模块回退
    if subtitle_config is None or not subtitle_config.enable_topic:
        mode_name = Environment.discover().moss_mode_name
        if mode_name:
            try:
                mode_configs = importlib.import_module(f"MOSS.modes.{mode_name}.configs")
                _stc = getattr(mode_configs, "subtitle_topic_config", None)
                if isinstance(_stc, SubtitleTopicConfig):
                    subtitle_config = _stc
                    logger.info("[subtitle] SubtitleTopicConfig 从 mode %s 回退导入: enable_topic=%s",
                                mode_name, _stc.enable_topic)
            except ImportError:
                pass

    # Layer 3: Topic 路径 或 HTTP 旁路
    if subtitle_config is not None and subtitle_config.enable_topic:
        # ── Topic 路径（新）：创建 TopicWindow + 消费协程 ──
        _subtitle_window = matrix.session.topics.create_window_for(
            SubtitleTopic, max_size=100,
            topic_name=subtitle_config.topic_path,
        )
        await _subtitle_window.wait_started()
        logger.info("[subtitle] SubtitleTopic 窗口已就绪（Topic 总线）")

        async def _consume_subtitle():
            """在 asyncio 事件循环上消费 SubtitleTopic 窗口，写入 SSE 队列。

            通过 len()/values() 轮询（~100ms），与 AudioRuntimeTopic 门控的
            轮询模式一致。每次只处理新增项，无线程安全问题。
            """
            consumed = 0
            while True:
                try:
                    current_len = len(_subtitle_window)
                    if current_len > consumed:
                        for topic in _subtitle_window.values()[consumed:]:
                            async with _SUBTITLE_LOCK:
                                _SUBTITLE_QUEUE.append({
                                    "type": "full" if topic.is_final else "chunk",
                                    "text": topic.text,
                                })
                            _SUBTITLE_EVENT.set()
                        consumed = current_len
                except Exception:
                    pass
                await asyncio.sleep(0.1)

        asyncio.create_task(_consume_subtitle())
        return True
    else:
        # ── HTTP 旁路（旧，兼容）：同进程 Speech 注入 ──
        _subtitle_cb = new_subtitle_callback()
        try:
            _speech = matrix.container.force_fetch(Speech)
            _speech.set_subtitle_callback(_subtitle_cb)
            logger.info("[subtitle] 字幕回调已注入 Speech（HTTP 旁路，同进程）")
        except Exception:
            logger.info("[subtitle] Speech 不在当前容器（HTTP 旁路在跨进程场景失效）")
        return False
