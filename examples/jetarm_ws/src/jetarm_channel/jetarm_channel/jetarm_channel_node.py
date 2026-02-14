from os.path import abspath, dirname, join

import rclpy

from ghoshell_moss import Channel, MutableChannel
from ghoshell_moss.transports.zmq_channel.zmq_channel import ZMQChannelProvider
from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import MOSSRobotManager, RobotController
from ghoshell_moss_contrib.prototypes.ros2_robot.joint_parsers import default_parsers

from .channels.body import body_chan
from .robot import jetarm_robot
from .ros2_node import Ros2RobotControllerNode, run_node


def main_channel_builder(main_channel: MutableChannel, controller: RobotController) -> Channel:
    body_chan.build.with_binding(RobotController, controller)
    body_chan.build.with_binding(MOSSRobotManager, controller.manager())
    main_channel.import_channels(body_chan)
    return main_channel


def main(args=None):
    provider = ZMQChannelProvider(
        address="tcp://0.0.0.0:9527",
    )
    rclpy.init(args=args)

    node = Ros2RobotControllerNode(
        node_name="jetarm_channel_node",
        config_dir=abspath(join(dirname(__file__), "config")),
        robot_yaml_filename="jetarm_robot.yaml",
        provider=provider,
        channel_builder=main_channel_builder,
        default_robot=jetarm_robot,
        joint_value_parsers=default_parsers,
        joint_states_topic="/joint_states",
        follow_joint_trajectory_server_name="/joint_trajectory_controller/follow_joint_trajectory",
        goal_interval=0.02,
    )

    run_node(args, node)


if __name__ == "__main__":
    main()
