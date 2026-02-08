from os.path import dirname, join
import sys

# patch miku 的读取路径.
# 由于 miku 还是一个实验性的数字人项目, 暂时不希望把它打包到 ghoshell_moss_contrib 里 (太大)
# 所以先用比较脏的相对路径来读取它.
current_dir = dirname(__file__)
workspace_dir = join(dirname(current_dir), '.workspace')
try:
    import miku
except ImportError:
    miku_dir = join(dirname(current_dir), 'miku')
    print(miku_dir)
    sys.path.append(miku_dir)
    from miku_provider import run_game_with_zmq_provider

import asyncio
from ghoshell_moss_contrib.example_ws import workspace_container

if __name__ == '__main__':
    with workspace_container(workspace_dir) as _container:
        asyncio.run(run_game_with_zmq_provider(address="tcp://localhost:5555", con=_container))
