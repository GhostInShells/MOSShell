from ghoshell_moss.speech import make_baseline_tts_speech
import asyncio

if __name__ == "__main__":

    async def main():
        speech = make_baseline_tts_speech()
        async with speech:
            for text in ["Hello World!", "How are you?"]:
                stream = speech.new_stream()
                stream.buffer(text)
                task = stream.as_command_task(commit=True)
                await task.run()
                print("+++++++++", task.tokens)


    asyncio.run(main())
