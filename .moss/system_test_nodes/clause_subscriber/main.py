"""clause_subscriber — 订阅 clause topic, 打印每一条说侧/听侧产出的 ClauseTopic.

不带 channel: 直接订阅 ``matrix.session.topics`` 的 ``clause`` topic (跨进程经 zenoh),
阻塞 poll 每条 ClauseTopic 并打印一行 JSON. 用于验证说侧装线
(``moss_runtime._clause_topic_bridge``) 确实把 speech 单例的 ``on_clause`` 结果广播
成了 ClauseTopic, 也对称覆盖听侧 (ASR 定稿 clause 时同 topic).

Start:  moss nodes run .moss/system_test_nodes/clause_subscriber/
Debug:  python main.py
"""

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.topics import ClauseTopic


async def main(matrix: Matrix):
    subscriber = matrix.session.topics.subscribe_model(ClauseTopic)
    matrix.logger.info("[clause_subscriber] listening on 'clause' topic, Ctrl-C to stop")

    async with subscriber:
        n = 0
        while True:
            clause = await subscriber.poll_model()
            if clause is None:
                # 非 clause 类型 (meta.type 不匹配) — 丢弃.
                continue
            n += 1
            print(f"clause #{n}: {clause.model_dump_json()}", flush=True)


if __name__ == "__main__":
    Matrix.discover().run(main)
