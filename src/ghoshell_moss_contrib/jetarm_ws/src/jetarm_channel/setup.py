import os
from glob import glob

from setuptools import find_packages, setup

package_name = "jetarm_channel"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("jetarm_channel/config/*.yaml"),
        ),
    ],
    install_requires=[
        "setuptools",
        "ghoshell-moss",
    ],
    zip_safe=True,
    maintainer="ThirdGerb",
    maintainer_email="thirdgerb@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "action_test = jetarm_channel.nodes.action_client_node:main",
            "channel_test = jetarm_channel.nodes.pychannel_with_rclpy:main",
            "jetarm_channel_node = jetarm_channel.jetarm_channel_node:main",
        ],
    },
)
