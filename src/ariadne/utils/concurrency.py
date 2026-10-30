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

"""Thread-pool helper for I/O-bound work such as LLM API calls."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_concurrently(
    items: Sequence[T], worker: Callable[[T], R], max_workers: int
) -> Iterator[tuple[int, R]]:
    """Yield ``(index, worker(item))`` for each item as it finishes.

    Results arrive in completion order; use the index to restore input order.
    Runs sequentially in input order when ``max_workers <= 1``. Consume the
    results in the calling thread so that side effects (file writes, DataFrame
    updates, logging) stay single-threaded. If a worker raises, items not yet
    started are cancelled and the exception propagates to the caller.
    """
    if max_workers <= 1 or len(items) <= 1:
        for idx, item in enumerate(items):
            yield idx, worker(item)
        return

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {executor.submit(worker, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
