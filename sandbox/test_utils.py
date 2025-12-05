import logging

from ariadne.utils.logger import open_log, _handle_exception
from ariadne.utils.config import Config


def test_config_initialization():
    config = Config()

    assert config.log_folder == "logs"


def test_logging():

    log_file = "test_log.log"
    open_log(log_file_name=log_file, clear_log_file=True)

    logger = logging.getLogger()
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")

    with open(log_file, "r") as f:
        log_contents = f.read()
        assert "This is an info message." in log_contents


if __name__ == "__main__":
    test_config_initialization()
    test_logging()
