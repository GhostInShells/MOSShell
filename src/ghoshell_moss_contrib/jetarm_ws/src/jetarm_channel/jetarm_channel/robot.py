from ghoshell_moss_contrib.prototypes.ros2_robot.joint_parsers import DegreeToRadiansParser
from ghoshell_moss_contrib.prototypes.ros2_robot.models import Controller, Joint, RobotInfo

jetarm_robot = RobotInfo(
    name="JetArm",
    description="""
6dof 机械臂, 结构由底盘, 上臂, 前臂, 手腕, 夹爪构成. 
它的仿生形态是一个蛇形机器人, 用夹爪表现面部, 用手腕表现脖子, 用肘来表现蛇的背, 用肩关节表现蛇的腰. 
同时拥有皮克斯台灯式的 AI 交互效果. 
""",
).with_controller(
    Controller(
        name="face",
        description="机械臂仿生机器人的头部, 由夹爪来扮演嘴巴",
    ).with_joint(
        Joint(
            name="gripper",
            origin_name="r_joint",
            description="夹爪的控制舵机, 可以控制夹爪的开合, 值是夹爪展开角度",
            default_value=30.0,
            min_value=-60.0,
            max_value=60.0,
            value_parser=DegreeToRadiansParser.name(),
        ),
    ),
    Controller(
        name="neck",
        description="机械臂仿生机器人的脖子, 由手腕 roll 和 pitch 舵机控制",
    ).with_joint(
        Joint(
            name="wrist_roll",
            origin_name="joint5",
            description="控制手腕 roll 轴, 默认居中, 单位是角度, > 0 向右旋转, < 0 向左旋转, 可以表示歪头.",
            default_value=0.0,
            min_value=-60.0,
            max_value=60.0,
            value_parser=DegreeToRadiansParser.name(),
        ),
        Joint(
            name="wrist_pitch",
            origin_name="joint4",
            description="控制手腕 pitch 轴, 单位是角度, 为 0.0 与前臂方向一致. > 0 后仰, < 0 前倾.",
            default_value=-45.0,
            min_value=-130.0,
            max_value=30.0,
            value_parser=DegreeToRadiansParser.name(),
        ),
    ),
    Controller(
        name="spine",
        description="机械臂仿生机器人的脊柱, 由机械臂的 elbow 关节舵机控制",
    ).with_joint(
        Joint(
            name="elbow_pitch",
            origin_name="joint3",
            description="控制手肘 pitch 轴, 单位是角度, 为 0.0 时手臂伸直. > 0 后仰, < 0 前倾.",
            default_value=-90.0,
            min_value=-135.0,
            max_value=30.0,
            value_parser=DegreeToRadiansParser.name(),
        ),
        Joint(
            name="shoulder_pitch",
            origin_name="joint2",
            description="控制肩膀 pitch 轴, 单位是角度, 为 0.0 时垂直指天, > 0 后仰, < 0 前倾.",
            default_value=45.0,
            min_value=-60.0,
            max_value=60.0,
            value_parser=DegreeToRadiansParser.name(),
        ),
    ),
    Controller(
        name="waist",
        description="机械臂仿生机器人的腰部, 由机械臂的 shoulder 控制, 可以像腰一样左右转",
    ).with_joint(
        Joint(
            name="shoulder_roll",
            origin_name="joint1",
            description="控制肩膀 roll 轴, 单位是角度, 为 0.0 时机械臂全身向正前方. > 0 右旋, < 0 左旋",
            default_value=0.0,
            min_value=-60.0,
            max_value=60.0,
            value_parser=DegreeToRadiansParser.name(),
        )
    ),
)
