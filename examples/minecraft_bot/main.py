import argparse
import asyncio
import logging
from asyncio import Queue
from typing import TypeVar

import dotenv
from ghoshell_common.contracts.workspace import LocalWorkspaceProvider
from ghoshell_container import Container
from javascript import On, require

from ghoshell_moss import PyChannel
from ghoshell_moss.core.shell import new_shell
from ghoshell_moss.message import Message, Text
from ghoshell_moss.speech import make_baseline_tts_speech
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer
from ghoshell_moss.speech.volcengine_tts import VolcengineTTS, VolcengineTTSConf
from ghoshell_moss_contrib.agent import ModelConf, SimpleAgent
from ghoshell_moss_contrib.agent.chat.queue import QueueChat

mineflayer = require("mineflayer")
pathfinder = require("mineflayer-pathfinder")
blockfinder = require("mineflayer-blockfinder")

logging.basicConfig(level=logging.INFO)

RANGE_GOAL = 1
BOT_USERNAME = "Jarvis"
WORKSPACE_DIR = ".workspace"
dotenv.load_dotenv(f"{WORKSPACE_DIR}/.env")

bot = mineflayer.createBot({"host": "127.0.0.1", "port": 25565, "username": BOT_USERNAME})
bot.loadPlugin(pathfinder.pathfinder)
bot.loadPlugin(blockfinder)

BOT = TypeVar("BOT")

chat = QueueChat(input_queue=Queue(), output_queue=Queue())


async def chat_task():
    while True:
        try:
            message = chat.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)
            continue

        if message.role != "assistant":
            continue

        if not message.is_completed():
            continue

        for content in message.contents:
            text = Text.from_content(content)
            if not text:
                continue
            bot.chat(text.text)


@On(bot, "spawn")
def handle_spawn(*args):
    movements = pathfinder.Movements(bot)
    bot.pathfinder.setMovements(movements)


@On(bot, "chat")
def handle_msg(this, sender, message, *args):
    if sender and (sender != BOT_USERNAME):
        moss_message = Message.new(role="user", name=sender).with_content(Text(text=f"{sender}: {message}"))
        chat.input_queue.put_nowait(moss_message)


bot_chan = PyChannel(name=BOT_USERNAME.lower())

to_follow_player = ""


@bot_chan.build.on_idle
async def on_policy_run():
    global to_follow_player
    while to_follow_player != "":
        await asyncio.sleep(0.5)
        try:
            await come(to_follow_player)
        except Exception as ex:
            logging.warning("follow to %s failed: %s", to_follow_player, ex)
            continue


@bot_chan.build.command()
async def set_follow_player(sender: str):
    """
    跟随玩家{sender}

    Args:
        sender: 玩家名称
    """
    global to_follow_player
    to_follow_player = sender


@bot_chan.build.command()
async def stop_follow_player():
    """
    停止跟随玩家
    """
    global to_follow_player
    to_follow_player = ""


@bot_chan.build.context_messages
async def context_messages():
    pos = bot.entity.position
    message = Message.new(role="user", name="__minecraft_bot__").with_content(
        Text(text=f"你当前的位置是：{pos.toString()}，周围的方块信息如下："),
    )
    for x_offset in range(-3, 3):  # 东西
        for z_offset in range(-3, 3):  # 南北
            for y_offset in range(-1, 3):  # 垂直方向
                block = bot.blockAt(pos.offset(x_offset, y_offset, z_offset))
                if block:
                    # 忽略空气方块
                    if block.name == "air":
                        continue
                    message.with_content(Text(text=f"{block.name} at {block.position.toString()}"))
                else:
                    message.with_content(Text(text=f"{block.position.toString()} has no block"))

    return [message]


@bot_chan.build.command()
async def move(x: int, y: int, z: int):
    """
    移动到指定位置

    Args:
        x: 目标x坐标
        y: 目标y坐标
        z: 目标z坐标
    """
    try:
        bot.pathfinder.setGoal(pathfinder.goals.GoalNear(x, y, z, RANGE_GOAL))
    except Exception as e:
        return f"移动失败：{e}"


@bot_chan.build.command()
async def come(sender: str):
    """
    去玩家{sender}的位置
    """
    player = bot.players[sender]
    target = player.entity
    if not target:
        bot.chat("I don't see you !")
        return

    pos = target.position
    bot.pathfinder.setGoal(pathfinder.goals.GoalNear(pos.x, pos.y, pos.z, RANGE_GOAL))


@bot_chan.build.command()
async def where_i_am():
    """
    获取你现在的位置
    """
    p = bot.entity.position
    return p.toString()


@bot_chan.build.command()
async def where_player_is(name: str):
    """
    获取指定玩家{name}的位置
    """
    player = bot.players[name]
    target = player.entity
    if not target:
        return "Not found"
    p = target.position
    return p.toString()


@bot_chan.build.command()
async def find_blocks(block_name: str, max_distance: int = 128, count=10):
    """
    找附近的{block_name}方块

    Args:
        block_name: 方块名称
        max_distance: 最大搜索距离，默认128
        count: 最多搜索数量，默认10
    """
    if bot.registry.blocksByName[block_name] is None:
        return f"{block_name} is not a block name"

    ids = [bot.registry.blocksByName[block_name].id]
    blocks = bot.findBlocks({"matching": ids, "maxDistance": max_distance, "count": count})

    return f"找到 {blocks.length} 个 {block_name} 方块：{blocks}"


@bot_chan.build.command()
async def dig_under():
    """
    挖掘你脚下的方块
    """
    target = bot.blockAt(bot.entity.position.offset(0, -1, 0))
    if target and bot.canDigBlock(target):
        bot.chat(f"starting to dig {target.name}")
        bot.dig(target)
        return f"挖掘了 {target.name}"
    else:
        return "你脚下没有可挖掘的方块"


@bot_chan.build.command()
async def dig_target(x: int, y: int, z: int):
    """
    挖掘指定位置的方块

    Args:
        x: 目标x坐标
        y: 目标y坐标
        z: 目标z坐标
    """
    bpx = bot.entity.position.x
    bpy = bot.entity.position.y
    bpz = bot.entity.position.z

    target = bot.blockAt(bot.entity.position.offset(x - bpx, y - bpy, z - bpz))
    if target and bot.canDigBlock(target):
        bot.lookAt(target.position)
        bot.dig(target)
        return f"挖掘了 {target.name}"
    else:
        return "你周围没有可挖掘的方块"


def init():
    container = Container()
    container.register(LocalWorkspaceProvider(workspace_dir=WORKSPACE_DIR))
    return container


async def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--speech", action="store_true", help="是否启用语音功能（添加该参数为True，不添加为False）")
    args = arg_parser.parse_args()

    container = init()
    speech = None
    if args.speech:
        player = PyAudioStreamPlayer()
        tts = VolcengineTTS(conf=VolcengineTTSConf(default_speaker="zh_male_ruyayichen_saturn_bigtts"))
        speech = make_baseline_tts_speech(player=player, tts=tts)
    shell = new_shell(speech=speech)
    shell.main_channel.import_channels(bot_chan)
    agent = SimpleAgent(
        instruction=f"你叫{BOT_USERNAME}，举止谈吐儒雅脱俗，生活在minecraft世界中",
        container=container,
        shell=shell,
        chat=chat,
        model=ModelConf(
            kwargs={
                "thinking": {
                    "type": "disabled",
                },
            },
        ),
    )
    asyncio.create_task(chat_task())
    await asyncio.sleep(1)
    await agent.run()


async def dry_test():
    container = init()
    await asyncio.sleep(1)

    async with bot_chan.run_in_ctx(container=container):
        res = await find_blocks("oak_log")
        await dig_target(x=8, y=73, z=21)
        pass


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(dry_test())
