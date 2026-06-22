"""Speech Streaming Monitor — 订阅 speech/streaming Topic，终端逐句显示。

验证 SpeechStreamingTopic 跨进程发布链路是否正常。
运行方式：moss apps test speech/streaming_monitor

启动后等待 Ghost TTS 输出（另一终端运行 moss-run-ghost echo --mode show），
本终端实时显示每句文本及批次边界。
"""

import asyncio
import time

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.topics.audio import SpeechTopic


async def main(matrix: Matrix) -> None:
    print("Speech Streaming Monitor")
    print(f"  topic : {SpeechTopic.default_topic_name()}")
    print(f"  type  : {SpeechTopic.topic_type()}")
    print("  等待 Ghost TTS 输出...\n")

    window = matrix.session.topics.create_window_for(
        SpeechTopic, max_size=200
    )

    t0 = time.monotonic()
    seen = 0
    batch_count = 0

    try:
        while True:
            await asyncio.sleep(0.12)
            items = window.values()
            new_items = items[seen:]
            for item in new_items:
                elapsed = time.monotonic() - t0
                print(f"  [{elapsed:6.1f}s] │ {item.text}")
            seen = len(items)
    except KeyboardInterrupt:
        print(f"\n  共接收 {seen} 条事件，{batch_count} 个批次")


if __name__ == "__main__":
    Matrix.discover().run(main)