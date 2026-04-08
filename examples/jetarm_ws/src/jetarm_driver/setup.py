from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetarm_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 新增以下行：将 config 目录及其内容安装到指定位置
        (os.path.join('share', package_name, 'config'), glob('jetarm_driver/config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thirdgerb',
    maintainer_email='thidrgerb@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jetarm_driver_node = jetarm_driver.jetarm_driver_node:main',
            'test_basic = jetarm_driver.test_basic:main',
            'test_concurrent = jetarm_driver.test_concurrent:main',
            'test_trajectory = jetarm_driver.test_trajectory:main',
        ],
    },
)
