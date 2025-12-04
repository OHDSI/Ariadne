# Copyright 2025 Observational Health Data Sciences and Informatics
#
# This file is part of Ariadne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys


def _add_stream_handler(logger: logging.Logger):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger


def _add_file_handler(logger: logging.Logger, log_file_name: str):
    file_handler = logging.FileHandler(log_file_name, mode="a")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def open_log(log_file_name: str, clear_log_file: bool = False) -> None:
    """
    Sets up the root logger where it writes all logging events to file, and writing events at or above 'info' to
    console. Events are appended to the log file. The logger will also capture uncaught exceptions.

    Args:
        log_file_name: The name of the file where the log will be written to.
        clear_log_file: If true, the log file will be cleared before writing to it.

    Returns:
        None
    """

    if clear_log_file:
        open(log_file_name, "w").close()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not len(logger.handlers):
        _add_file_handler(logger=logger, log_file_name=log_file_name)
        _add_stream_handler(logger=logger)

    sys.excepthook = _handle_exception


def _handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        logger = logging.getLogger()
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
