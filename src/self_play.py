import multiprocessing as mp
from typing import List, Optional, Tuple

import chess
import torch

from src.board_encoding import encode_board
from src.mcts import Node, create_child_node, extract_policy, sample_move, search


def play_self_play_game(
    inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
    result_queue: mp.Queue,
    num_simulations: int = 800,
    max_moves: int = 200,
    re_use_tree: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    board: Optional[chess.Board] = chess.Board()
    history: Optional[list[chess.Board]] = []
    examples = []
    move_count = 0

    # Start with fresh root
    root = Node(board=board, history=history, prior_prob=0.0)

    while not root.board.is_game_over() and move_count < max_moves:
        # Run MCTS
        search(root, inference_request_queue, inference_response_queue, num_simulations)

        # Store example
        state = encode_board(root.board, root.history)
        policy = extract_policy(root)
        examples.append(
            {"state": state, "policy": policy, "player": "white" if root.board.turn == chess.WHITE else "black"}
        )

        # Sample move
        temperature = 1.0 if move_count < 30 else 0.1
        move = sample_move(root, temperature)

        # Make move
        if re_use_tree:
            root = root.children[move]
        else:
            root = create_child_node(root, move, prior_prob=0.0)

        move_count += 1

    # Create training examples
    return create_training_examples(examples, root.board)


def create_training_examples(
    examples: List[dict], final_board: chess.Board
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert examples to training format"""
    result = final_board.result()
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    else:
        outcome = 0.0

    training_examples = []
    for example in examples:
        value = outcome * example["player"]
        value_tensor = torch.tensor([[value]], dtype=torch.float32)

        training_examples.append((example["state"], example["policy"], value_tensor))

    return training_examples


def count_nodes(node: Node) -> int:
    if node.children is None or len(node.children) == 0:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())
