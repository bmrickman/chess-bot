import multiprocessing as mp
import time
from queue import Empty

from src2.model.evaluate import evaluate_boards

MAX_BATCH_SIZE = 256
MAX_INFERENCE_WAIT_TIME = 0.01


def run_inference_server(input_queues: list[mp.Queue], output_queues: list[mp.Queue]) -> None:
    while True:
        batch = []
        batch_queue_refs = []
        start_time = time.time()

        while len(batch) < MAX_BATCH_SIZE or (time.time() - start_time) < MAX_INFERENCE_WAIT_TIME:
            for queue_index, input_queue in enumerate(input_queues):
                try:
                    board, history = input_queue.get_nowait()
                    batch.append((board, history))
                    batch_queue_refs.append(output_queues[queue_index])
                except Empty:
                    continue

        evaluations = evaluate_boards(batch)
        for (value, policy), output_queue in zip(evaluations, batch_queue_refs):
            output_queue.put((value, policy))
