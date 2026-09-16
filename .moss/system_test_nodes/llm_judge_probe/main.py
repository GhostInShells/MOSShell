"""long_listen_probe — 智能判停 (长程聆听) 实机测试 node.

自己 assemble listener (ModelListenerController), 旁路监控判停全链路:
- ``on_recognition_result`` → 记录 clause / segment 时序 (clause 到 commit 的 delta)
- ``controller.on_score`` → 记录每次 llm 打分 (请求 clauses + score + cast + tokens)

聚焦打印三个测试点 (clause / judge / segment), 不刷 partial 噪声. 验证智能判停在
真实 ASR + LLM 下的判停时机: 长论述不该在句中误 commit, 停顿后应正常 commit.

用法 (可选设备名作为 argv, 缺省走项目默认设备):

    moss nodes run .moss/system_test_nodes/long_listen_probe/ -- <device_pattern>
"""

import asyncio
import sys
import time

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.listener.controller import ModelListenerController
from ghoshell_moss.host.listener.stop_judge import StopScoreObservation
from ghoshell_moss.host.nodes.listener_node import assemble_controller


async def main(matrix: Matrix):
    device = sys.argv[1] if len(sys.argv) > 1 else None
    controller = await assemble_controller(matrix, device=device, emit_signals=False)
    if not isinstance(controller, ModelListenerController):
        matrix.logger.error("[long_listen_probe] LLMFuncs not available — long_listen disabled")
        return

    controller.on_score(_print_score)

    last_clause_at: float | None = None

    def _on_result(result: RecognitionEvent) -> None:
        nonlocal last_clause_at
        now = time.monotonic()
        if result.phase == RecognitionPhase.CLAUSE:
            last_clause_at = now
            text = result.clause.text if result.clause else result.text
            print(f"[clause] {text}", flush=True)
        elif result.phase == RecognitionPhase.TAIL:
            delta = now - last_clause_at if last_clause_at is not None else 0.0
            print(f"[segment] {result.text}  (+{delta:.2f}s from last clause)", flush=True)
            last_clause_at = None

    controller.on_recognition_result(_on_result)

    matrix.logger.info("[long_listen_probe] listening with llm stop-detection, Ctrl-C to stop")
    controller.long_listen(timeout=None)
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass


def _print_score(obs: StopScoreObservation) -> None:
    usage = obs.result.usage or {}
    clauses = " | ".join(obs.clauses)
    print(
        f"[judge] score={obs.score} cast={obs.result.cast:.2f}s "
        f"tok(in={usage.get('input_tokens', 0)} cache_r={usage.get('cache_read_tokens', 0)} "
        f"out={usage.get('output_tokens', 0)}) clauses=[{clauses}]",
        flush=True,
    )


if __name__ == "__main__":
    Matrix.discover().run(main)
