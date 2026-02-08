from ghoshell_moss_contrib.agent.chat.console import ConsoleChat
from ghoshell_moss_contrib.agent.simple_agent import ModelConf, SimpleAgent


def main():
    import asyncio

    agent = SimpleAgent(
        instruction="你是 JoJo",
        chat=ConsoleChat(),
        model=ModelConf(
            kwargs={
                "thinking": {
                    "type": "disabled",
                },
            },
        ),
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
