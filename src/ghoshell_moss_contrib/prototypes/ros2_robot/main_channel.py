import asyncio
import json

from pydantic import ValidationError

from ghoshell_moss import ChannelUtils, CommandErrorCode, PyChannel
from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import MOSSRobotManager, RobotController
from ghoshell_moss_contrib.prototypes.ros2_robot.models import Animation, Pose, Trajectory


def build_robot_main_channel(controller: RobotController) -> PyChannel:
    """
    创建机器人的主轨道.
    并不是必要的函数, 只是为了展示如何使用.
    """
    # 初始化 Channel
    name = controller.manager().robot().name
    main_channel = PyChannel(name=name, blocking=True)

    # 绑定到 broker.
    main_channel.build.with_binding(RobotController, controller)
    main_channel.build.with_binding(MOSSRobotManager, controller.manager())

    # 注册整个 robot 的 description 生成函数.
    main_channel.build.description()(
        build_robot_description,
    )

    # 注册基础的运行轨迹函数.
    main_channel.build.command(
        # 生成一个轨迹函数的描述.
        doc=run_trajectory_doc,
    )(run_trajectory)

    # 注册位姿相关的函数.
    main_channel.build.command(doc=save_pose_doc)(save_pose)
    main_channel.build.command()(remove_pose)
    main_channel.build.command()(read_pose)
    main_channel.build.command()(reset_pose)
    main_channel.build.command()(move_to_pose)
    main_channel.build.command(doc=move_to_doc)(move_to)

    # # 注册动画相关的函数.
    main_channel.build.command()(play)
    main_channel.build.command()(remove_animation)
    main_channel.build.command(doc=save_animation_doc)(save_animation)

    # 返回这个 channel.
    return main_channel


def build_robot_description() -> str:
    """
    用于生成这个机器人的描述.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    return _controller.robot_state()


run_trajectory_doc = f"""
让机器人立刻运行一个动画轨迹. 机器人会中断当前动作, 执行动画直到结束或出错. 

:param text__: 用 JSON 描述的动画轨迹. 它的 JSON Schema 如右: {Trajectory.model_json_schema()}
"""


async def normalized_wait_fut(future: asyncio.Future) -> None:
    """
    语法糖, 和 RobotController 配合的 Future 处理规则.
    """
    try:
        # 阻塞等待到结束.
        await future
    finally:
        # 退出时, future 一定要 cancel, 用这种方式通知下游终止.
        if not future.done():
            future.cancel()


async def run_trajectory(text__: str) -> None:
    """
    为这个命令所属的机器人提供一个执行轨迹动画的函数.
    注意用 fn_run_trajectory_doc 来重写这个 doc.
    需要提示 AI 尽可能少用这个方法 (生成 token 太长太久), 而用编程或预设动画替代.
    """
    try:
        data = json.loads(text__)
        trajectory = Trajectory(**data)
    except Exception as e:
        raise CommandErrorCode.VALUE_ERROR.error("Invalid text__ format, must follow its JSON Schema")

    _controller = ChannelUtils.ctx_get_contract(RobotController)
    # 运行这个轨迹动画.
    future = _controller.run_trajectory(trajectory)
    await normalized_wait_fut(future)


async def play(name: str) -> None:
    """
    让机器人运行一个已经注册过的动画 (animation).
    :param name: 动画的名称. 必须是机器人信息里定义存在的动画.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    fut = _controller.play_animation_name(name)
    await normalized_wait_fut(fut)


save_animation_doc = f"""
保存一个机器人动画, 该动画可以被 play 未来执行. 

:param text__: 对应 animation 的 json schema 如右: `{Animation.model_json_schema()}`
"""


def save_animation(text__: str) -> None:
    """
    定义一个关键帧动画. 可
    """
    try:
        data = json.loads(text__)
        animation = Animation(**data)
    except Exception as e:
        raise CommandErrorCode.VALUE_ERROR.error("Invalid text__ format, must follow its JSON Schema")

    _controller = ChannelUtils.ctx_get_contract(RobotController)
    # 保存动画.
    _controller.manager().save_animation(animation)


def remove_animation(name: str) -> None:
    """
    移除一个保存过的动画.
    :param name: 动画的名称.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    _controller.manager().remove_animation(name)


move_to_doc = """
移动到一个目标位姿

:param duration: 指定运行所需要的时间, 默认是 1.0 秒. 
:param text__: 用 json 格式, 传入位姿讯息, key 是关节, value 是关节的值. 
"""


async def move_to(text__: str, duration: float = 1.0) -> None:
    try:
        data = json.loads(text__)
        pose = Pose(positions=data)
    except json.JSONDecodeError as e:
        raise CommandErrorCode.VALUE_ERROR.error(f"invalid text__ json format: {e}")

    except ValidationError as e:
        raise CommandErrorCode.VALUE_ERROR.error("Invalid text__ format, must follow its JSON Schema")

    _controller = ChannelUtils.ctx_get_contract(RobotController)
    fut = _controller.move_to_pose(pose, duration=duration)
    await normalized_wait_fut(fut)


async def move_to_pose(name: str, duration: float = 1.0) -> None:
    """
    使当前机器人运行到一个预设的 pose.
    :param name: 已经保存过的位姿名称.
    :param duration: 这个执行轨迹预计消耗的时间.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    # 找到目标姿态.
    pose = _controller.manager().get_pose(name)
    future = _controller.move_to_pose(pose, duration)
    # 阻塞等待到结束.
    await normalized_wait_fut(future)


async def read_pose(name: str) -> str:
    """
    读取一个已经存在的 pose 讯息.
    :return: 目标 pose 的所有关节位置的 json, 方便你深入理解位姿.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    pose = _controller.manager().get_pose(name)
    # 返回它的 json 值.
    return pose.model_dump_json()


def remove_pose(name: str) -> None:
    """
    移除一个已经定义的 pose.
    :param name: 必须是已经定义过的 pose 名称.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    manager = _controller.manager()
    manager.remove_pose(name)


async def reset_pose(duration: float = 1.0) -> None:
    """
    机器人将重置到当前的默认姿态.
    :param duration: 预期重置到默认位姿所花的时间, 单位是秒.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    fut = _controller.reset_pose(duration)
    await normalized_wait_fut(fut)


def set_default_pose(name: str) -> None:
    """
    修改机器人的默认姿态, 为一个已知的位姿.
    接下来 reset_pose 时机器人都会回到这个位姿.
    """
    _controller = ChannelUtils.ctx_get_contract(RobotController)
    manager = _controller.manager()
    manager.set_default_pose(name)


save_pose_doc = f"""
保存一个机器人的位姿信息. 

:param text__: 机器人的位姿数据结构, json schema 如右: {Pose.model_json_schema()}
"""


def save_pose(text__: str) -> None:
    """
    使用 json 方式保存一个新的 pose.
    """
    try:
        data = json.loads(text__)
        pose = Pose(**data)
    except Exception as e:
        raise CommandErrorCode.VALUE_ERROR.error("Invalid text__ format, must follow its JSON Schema")

    _controller = ChannelUtils.ctx_get_contract(RobotController)
    manager = _controller.manager()
    # 保存一个位姿.
    manager.save_pose(pose)
