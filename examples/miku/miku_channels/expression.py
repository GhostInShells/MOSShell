import asyncio

import live2d.v3 as live2d

from ghoshell_moss.core import PyChannel, ChannelCtx

expression_chan = PyChannel(name="expression")


@expression_chan.build.command()
async def reset():
    """
    reset expression to default
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.ResetExpression()


@expression_chan.build.command()
async def surprised(duration: float = 0):
    """
    surprised expression
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("Chijing")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()


@expression_chan.build.command()
async def dazhihui(duration: float = 0):
    """
    dazhihui expression, 呆呆的大聪明表情
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("Dazhihui")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()


@expression_chan.build.command()
async def mimi_eyes(duration: float = 0):
    """
    mimi eyes expression (Mimiyan)
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("Mimiyan")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()


@expression_chan.build.command()
async def blush(duration: float = 0):
    """
    blush expression (Saihong)
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("Saihong")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()


@expression_chan.build.command()
async def wearing_glass(duration: float = 0):
    """
    wearing a glass expression
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("Yanjing")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()


@expression_chan.build.command()
async def sweat(duration: float = 0):
    """
    sweat expression (liuhan)
    """
    model = ChannelCtx.get_contract(live2d.LAppModel)
    model.SetExpression("liuhan")
    if duration > 0:
        await asyncio.sleep(duration)
        model.ResetExpression()
