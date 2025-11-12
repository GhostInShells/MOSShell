from ghoshell_moss.speech.volcengine_tts import VolcengineTTS
import numpy as np
import asyncio

if __name__ == "__main__":
    # 测试验证 tts 可以运行.
    from ghoshell_common.contracts.logger import get_console_logger


    async def baseline():
        tts = VolcengineTTS(logger=get_console_logger("moss"))

        def print_data_len(data: np.ndarray):
            print("========== audio", len(data.tobytes()))

        async with tts:
            for text in ["Hello World", "Hello World"]:
                batch = tts.new_batch(callback=print_data_len)
                for char in text:
                    batch.feed(char)
                batch.commit()
                await batch.wait_until_done()
                print("batch %s done" % batch.batch_id())
        print("++++++++ tts shall finish")


    asyncio.run(baseline())
