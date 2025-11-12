import multiprocessing as mp
import time
from queue import Empty
from typing import Callable

from src2.types import Action, State


def run_inference_server(
    input_queues: list[mp.Queue],
    output_queues: list[mp.Queue],
    batch_evaluation_fn: Callable[[list[State]], list[tuple[float, dict[Action, float]]]],
    max_batch_size: int = 256,
    max_inference_wait_time: float = 0.01,
) -> None:
    while True:
        batch = []
        batch_queue_refs = []
        start_time = time.time()

        while len(batch) < max_batch_size and (time.time() - start_time) < max_inference_wait_time:
            for queue_index, input_queue in enumerate(input_queues):
                try:
                    batch.append(input_queue.get_nowait())
                    batch_queue_refs.append(output_queues[queue_index])
                except Empty:
                    continue

        evaluations = batch_evaluation_fn(batch)
        for evaluation, output_queue in zip(evaluations, batch_queue_refs):
            output_queue.put(evaluation)
