import logging

from mechagent.core.logging import get_logger, setup_logging


def test_setup_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging("INFO", log_file)

    get_logger("tests").info("hello from the tests")
    for handler in logging.getLogger("mechagent").handlers:
        handler.flush()

    assert "hello from the tests" in log_file.read_text()


def test_setup_logging_twice_does_not_stack_handlers(tmp_path):
    setup_logging("INFO")
    logger = setup_logging("DEBUG", tmp_path / "run.log")

    assert len(logger.handlers) == 2
    assert logger.level == logging.DEBUG


def test_get_logger_is_under_mechagent_namespace():
    assert get_logger("cad").name == "mechagent.cad"
    assert get_logger("mechagent.fem").name == "mechagent.fem"
