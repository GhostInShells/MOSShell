"""topic_subscriber — 订阅任意 topic (按名字), 打印每条原始 Topic 一行 JSON.

不带 topic model: 直接 ``matrix.session.topics.subscribe(topic_name)`` 订阅原始
topic, 阻塞 poll 每条 ``Topic`` 并打印一行 JSON. 用于验证任意 topic 的生产装线
(clause / audio/sample / ...), 无需为每种 topic 各写一个 node.

用法: 把 topic 名作为 argv 传入 (缺省 'clause'):

    moss nodes run .moss/system_test_nodes/topic_subscriber/ -- audio/sample

Start:  moss nodes run .moss/system_test_nodes/topic_subscriber/ -- <topic_name>
Debug:  python main.py <topic_name>
"""

import sys

from ghoshell_moss.core.blueprint.matrix import Matrix


async def main(matrix: Matrix):
    topic_name = sys.argv[1] if len(sys.argv) > 1 else "clause"
    subscriber = matrix.session.topics.subscribe(topic_name)
    matrix.logger.info(f"[topic_subscriber] listening on '{topic_name}', Ctrl-C to stop")

    async with subscriber:
        n = 0
        while True:
            topic = await subscriber.poll()
            n += 1
            print(f"{topic_name} #{n}: {topic.model_dump_json(ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    Matrix.discover().run(main)
