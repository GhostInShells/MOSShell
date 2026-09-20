"""RuntimeErrorLog 契约 — 有界 ERROR+ tail + 永不丢 CRITICAL."""

import logging

from ghoshell_moss.core.runtime_error import RuntimeErrorLogImpl


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("runtime_error_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    return logger


def test_captures_error_and_critical_not_warning():
    log = RuntimeErrorLogImpl()
    logger = _make_logger()
    log.attach(logger)

    logger.warning("warn")
    logger.error("err")
    logger.critical("crit")

    assert log.total() == 2  # ERROR + CRITICAL (WARNING 不入)
    assert [r.message for r in log.pull(10)] == ["err", "crit"]
    assert [r.message for r in log.pull_critical()] == ["crit"]


def test_pull_returns_latest_n():
    log = RuntimeErrorLogImpl(fifo_capacity=4)
    logger = _make_logger()
    log.attach(logger)

    for i in range(6):
        logger.error(f"e{i}")

    assert log.total() == 6
    assert [r.message for r in log.pull(3)] == ["e3", "e4", "e5"]


def test_critical_never_dropped_on_fifo_overflow():
    log = RuntimeErrorLogImpl(fifo_capacity=2)
    logger = _make_logger()
    log.attach(logger)

    logger.critical("crit-keep")
    for i in range(5):
        logger.error(f"e{i}")

    assert log.total() == 6
    # CRITICAL 被 FIFO 挤出, 但独立缓冲仍保留.
    assert [r.message for r in log.pull_critical()] == ["crit-keep"]


def test_attach_is_idempotent():
    log = RuntimeErrorLogImpl()
    logger = _make_logger()
    log.attach(logger)
    log.attach(logger)

    logger.error("once")
    assert log.total() == 1  # 重复 attach 不双计
