"""Listener node functions — 把 CLI 的 listen 逻辑迁成可复用的 node 函数 (matrix).

node 函数签名统一为 ``async def xxx(matrix: Matrix, ...)``, 由 ``Matrix.run(...)`` /
``Matrix.discover().run(...)`` 驱动.

- ``listener_node`` — 探测用, 跑一次 once/always 聆听, 不 provide channel.
- ``listener_controller_node`` — 常驻, 启动 always 并提供 listener 控制 channel.
"""
from __future__ import annotations

from typing import Optional

from ghoshell_moss.contracts.audio import AudioCaptureConfig, AudioCaptureSource
from ghoshell_moss.contracts.configs import ConfigStore, get_or_create_conf
from ghoshell_moss.contracts.listener import ASRListener
from ghoshell_moss.contracts.llms import CallSettings, LLMFuncs, MossLLMCaller, MossLLMFuncs
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.listener.controller import ListenerController
from ghoshell_moss.host.listener.stop_judge import STOP_JUDGE_INSTRUCTION

__all__ = ["assemble_controller", "listener_node", "listener_controller_node"]


async def assemble_controller(
        matrix: Matrix,
        *,
        device: Optional[str] = None,
        emit_signals: bool = True,
) -> ListenerController:
    """装配 capture + seedasr + controller, 返回 controller (它托管 listener 生命周期).

    完成两条输出装线: signal_broadcast 注入 ``matrix.session.add_signal`` (识别事件 →
    listener signal), 以及 clause → ClauseTopic (``with_topic_service``).
    """
    con = matrix.container
    if device is not None:
        get_or_create_conf(con, AudioCaptureConfig()).device_pattern = device
    listener = con.get(ASRListener)
    if listener is None:
        raise RuntimeError("ASRListener not provided by IoC")
    asr = listener.asr()
    capture = con.get(AudioCaptureSource)
    sample_rate = capture.sample_rate if capture is not None else asr.get_info().sample_rate
    caller = _try_build_stop_judge_caller(con, matrix.logger)
    controller = ListenerController(
        listener=listener, asr=asr, logger=matrix.logger,
        signal_broadcast=matrix.session.add_signal if emit_signals else None,
        stop_caller=caller,
    )
    await controller.with_topic_service(matrix.session.topics)
    await controller.with_audio_sample_service(matrix.session.topics, sample_rate=sample_rate)
    config_store = con.get(ConfigStore)
    if config_store is not None:
        controller.with_config_store(config_store)
    await matrix.add_lifecycle_object(controller)
    return controller


def _try_build_stop_judge_caller(con, logger) -> Optional[MossLLMCaller]:
    """出口位点降级件的依赖: 模型环境可用时才构建, 否则 None.

    None 不是错误 —— 默认出口本就不依赖模型; 拿到 None 时 judge 件静默缺席,
    礼仪退回 segment_vad/keywords.
    """
    funcs = con.get(LLMFuncs)
    if not isinstance(funcs, MossLLMFuncs):
        return None
    try:
        return funcs.caller(
            instruction=STOP_JUDGE_INSTRUCTION,
            tag="small_fast_model",
            settings=CallSettings(max_output_tokens=1),
        )
    except Exception as exc:
        logger.warning("stop judge caller unavailable — llm_judge disabled: %s", exc)
        return None


async def listener_node(
        matrix: Matrix,
        *,
        mode: str = "always",
        timeout: float = 60.0,
        device: Optional[str] = None,
        emit_signals: bool = True,
) -> None:
    """探测用: 跑一次 once/always 聆听, 不 provide channel. ``mode``: once | always."""
    controller = await assemble_controller(matrix, device=device, emit_signals=emit_signals)
    if mode == "once":
        await controller.once(timeout=timeout)
    elif mode == "always":
        await controller.always(timeout=timeout)
    else:
        raise ValueError(f"unknown listener mode {mode!r} (once | always)")


async def listener_controller_node(
        matrix: Matrix,
        *,
        device: Optional[str] = None,
        emit_signals: bool = True,
) -> None:
    """常驻 listener node: 启动 always 持续聆听, 提供 listener 控制 channel.

    clause → ClauseTopic 与 signal 广播已由 ``assemble_controller`` 装线.
    provide_channel 阻塞到 membrane 关闭 — 这是 node 的唯一入网动作.
    """
    controller = await assemble_controller(matrix, device=device, emit_signals=emit_signals)
    controller.start_default_etiquette()
    await matrix.provide_channel(controller.as_channel())
