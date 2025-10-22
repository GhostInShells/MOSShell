from ghoshell_moss.agent.simple_agent import SimpleAgent


def main():
    import asyncio
    agent = SimpleAgent(
        instruction="你是 JoJo",
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
