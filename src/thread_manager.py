import time
from multiprocessing import Manager, Process, Queue, set_start_method
from threading import Thread
from typing import List

from src.inference_server import run_inference_server
from src.nn import AlphaZeroNet
from src.self_play import play_self_play_game

set_start_method("spawn", force=True)  # safer for nested threads


def main(num_processes: int = 2, threads_per_process: int = 3):
    with Manager() as manager:
        all_queues = manager.dict(
            {
                (process_num, thread_num): (manager.Queue(), manager.Queue(), manager.Queue())
                for process_num in range(num_processes)
                for thread_num in range(threads_per_process)
            }
        )
        processes: List[Process] = []
        for process_id in range(num_processes):
            process_queues = {k: v for k, v in all_queues.items() if k[0] == process_id}
            p = Process(
                target=process_worker,
                kwargs={"queues": process_queues, "process_id": process_id},
                name=f"Process-{process_id}",
            )
            p.start()
            processes.append(p)

        server_proc = Process(
            target=run_inference_server, kwargs={"queues": all_queues, "model": AlphaZeroNet()}, name="InferenceServer"
        )
        server_proc.start()

        for p in processes:
            p.join()

        server_proc.terminate()


def process_worker(queues: dict[tuple[int, int], tuple[Queue, Queue, Queue]], process_id: int):
    threads: List[Thread] = []

    for (process_id, worker_id), (inference_request_queue, inference_response_queue, result_queue) in queues.items():
        th = Thread(
            target=play_self_play_game,
            kwargs={
                "inference_request_queue": inference_request_queue,
                "inference_response_queue": inference_response_queue,
                "result_queue": result_queue,
                "num_simulations": 800,
                "max_moves": 200,
                "re_use_tree": False,
            },
            daemon=True,
            name=f"SelfPlay-{process_id}-{worker_id}",
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()


if __name__ == "__main__":
    main(num_processes=8, threads_per_process=10)
