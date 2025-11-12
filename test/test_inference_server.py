import math
import time
from multiprocessing import Process, Queue

from src2.model.inference_server import run_inference_server


def batch_evaluation_fn(states: list[str]) -> list[str]:
    return ["_" + s for s in states]


def test_inference_server_parameterized_by_batch_size():
    input_queues = [Queue() for _ in range(2)]
    output_queues = [Queue() for _ in range(2)]

    p = Process(
        target=run_inference_server,
        kwargs={
            "input_queues": input_queues,
            "output_queues": output_queues,
            "batch_evaluation_fn": batch_evaluation_fn,
            "max_batch_size": 2,
            "max_inference_wait_time": math.inf,
        },
    )
    p.start()
    input_queues[0].put("state1")
    time.sleep(1)
    if not output_queues[0].empty():
        raise Exception()
    input_queues[0].put("state2")
    time.sleep(1)
    assert output_queues[0].get() == "_state1"
    assert output_queues[0].get() == "_state2"
    input_queues[0].put("state3")
    input_queues[1].put("state4")
    time.sleep(1)
    assert output_queues[0].get() == "_state3"
    assert output_queues[1].get() == "_state4"
    p.terminate()


def test_inference_server_parameterized_by_wait_time():
    input_queues = [Queue() for _ in range(2)]
    output_queues = [Queue() for _ in range(2)]

    p = Process(
        target=run_inference_server,
        kwargs={
            "input_queues": input_queues,
            "output_queues": output_queues,
            "batch_evaluation_fn": batch_evaluation_fn,
            "max_batch_size": math.inf,
            "max_inference_wait_time": 10,
        },
    )
    p.start()
    input_queues[0].put("state1")
    time.sleep(1)
    if not output_queues[0].empty():
        raise Exception()
    time.sleep(9)
    assert output_queues[0].get() == "_state1"
    p.terminate()
