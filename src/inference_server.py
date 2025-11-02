import multiprocessing as mp
import time
from queue import Empty

import torch
import torch.nn.functional as F

from src.nn import AlphaZeroNet


def run_inference_server(queues: dict[tuple[int, int], tuple[mp.Queue, mp.Queue, mp.Queue]], model: torch.nn.Module):
    batch_size = 256
    timeout = 0.01

    model = AlphaZeroNet()
    model.to("cuda")
    model.eval()

    request_count = 0
    batch_count = 0

    while True:
        # Collect batch from ALL request queues
        batch = []
        batch_queue_ids = []
        start_time = time.time()

        while len(batch) < batch_size:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                break

            # Poll each request queue
            for process_id, thread_id in queues.keys():
                if len(batch) >= batch_size:
                    break

                request_queue = queues[(process_id, thread_id)][0]

                try:
                    # Non-blocking get
                    request = request_queue.get_nowait()
                    batch.append(request)
                    batch_queue_ids.append((process_id, thread_id))
                    request_count += 1
                except Empty:
                    # This queue is empty, continue to next
                    continue

            # Small sleep to avoid busy-waiting if no requests found
            if not batch:
                time.sleep(0.0001)

        # Skip if no requests
        if not batch:
            time.sleep(0.001)
            continue

        # Batch inference
        batch_count += 1
        states = torch.stack(batch).to("cuda")

        with torch.no_grad():
            policy_logits, values = model(states)
            policies = F.softmax(policy_logits, dim=1).cpu()
            values = values.cpu()

        # Route responses back to correct threads
        for (process_id, thread_id), policy, value in zip(batch_queue_ids, policies, values):
            response_queue = queues[(process_id, thread_id)][1]
            try:
                response_queue.put({"policy": policy, "value": value.item()})
            except Exception as e:
                print("here")
