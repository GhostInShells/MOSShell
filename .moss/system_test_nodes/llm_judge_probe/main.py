"""llm_judge_probe — 智能判停 (llm judge) 实机测试 node.

自己 assemble listener (ModelListenerController), 旁路监控判停全链路, 汇总成一个
per-segment 块打印: clause / llm 打分 (score + cast + token) / commit 机制与 delta.

验证智能判停在真实 ASR + LLM 下的判停时机: 长论述不该在句中误 commit, 停顿后应正常
commit (由 judge 提前, 而非 segment_vad 兜底).

设备选择走 node 自身的 dotenv (``MOSS_AUDIO_CAPTURE_DEVICE``), 不做 argv 加工.

用法:

    moss nodes run .moss/system_test_nodes/llm_judge_probe/
"""

import asyncio
import time

from ghoshell_moss.contracts.asr import RecognitionEvent, RecognitionPhase
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.listener.controller import ModelListenerController
from ghoshell_moss.host.listener.stop_judge import StopScoreObservation
from ghoshell_moss.host.nodes.listener_node import assemble_controller

_JUDGE_THRESHOLD = 7


def _fmt_score(obs: StopScoreObservation) -> str:
    """一次打分的紧凑表示: score + 延迟 + token. result=None 表示 llm 调用失败."""
    if obs.result is None:
        return f"score={obs.score} (llm call failed)"
    usage = obs.result.usage or {}
    return (
        f"score={obs.score} cast={obs.result.cast:.2f}s "
        f"in={usage.get('input_tokens', 0)} cache_r={usage.get('cache_read_tokens', 0)} "
        f"out={usage.get('output_tokens', 0)}"
    )


async def main(matrix: Matrix):
    controller = await assemble_controller(matrix, emit_signals=False)
    if not isinstance(controller, ModelListenerController):
        matrix.logger.error("[llm_judge_probe] LLMFuncs not available — llm_judge disabled")
        return

    # per-segment 聚合状态 (TAIL 时打印并重置).
    seg = {
        "clauses": [],
        "scores": [],
        "judge_committed": False,
        "last_clause_at": None,
    }

    def _record_score(obs: StopScoreObservation) -> None:
        seg["scores"].append(obs)
        if obs.score is not None and obs.score >= _JUDGE_THRESHOLD:
            seg["judge_committed"] = True
        print(f"[judge] {_fmt_score(obs)}  clauses=[{' | '.join(obs.clauses)}]", flush=True)

    def _on_result(result: RecognitionEvent) -> None:
        now = time.monotonic()
        sid = result.segment_id[-4:]
        if result.phase == RecognitionPhase.FIRST:
            print(f"[first seg={sid}] {result.text}", flush=True)
        elif result.phase == RecognitionPhase.PARTIAL:
            rel = f"+{now - seg['last_clause_at']:.2f}s" if seg["last_clause_at"] else "?"
            print(f"[partial {rel} seg={sid}] {result.text}", flush=True)
        elif result.phase == RecognitionPhase.CLAUSE:
            text = result.clause.text if result.clause else result.text
            seg["clauses"].append(text)
            seg["last_clause_at"] = now
            print(f"[clause seg={sid}] {text}", flush=True)
        elif result.phase == RecognitionPhase.TAIL:
            delta = now - seg["last_clause_at"] if seg["last_clause_at"] is not None else 0.0
            commit = "judge" if seg["judge_committed"] else "vad"
            scores = ", ".join(_fmt_score(s) for s in seg["scores"]) or "none"
            print(
                f"[segment seg={sid}] {result.text}  commit={commit} +{delta:.2f}s  "
                f"(clauses={len(seg['clauses'])} judge=[{scores}])",
                flush=True,
            )
            seg["clauses"] = []
            seg["scores"] = []
            seg["judge_committed"] = False
            seg["last_clause_at"] = None

    controller.on_score(_record_score)
    controller.on_recognition_result(_on_result)

    matrix.logger.info("[llm_judge_probe] listening with llm stop-detection, Ctrl-C to stop")
    controller.llm_judge(timeout=None)
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    Matrix.discover().run(main)
