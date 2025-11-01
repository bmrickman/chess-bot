from typing import List, Tuple

import chess
import torch

from src.board_encoding import encode_board
from src.mcts import create_root_node, extract_policy, sample_move, search


def play_automated_game(
    model: torch.nn.Module, num_simulations: int = 800, device: str = "cuda", max_moves: int = 200
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Play one self-play game

    Tree reuse is natural with your search() design!
    """
    board = chess.Board()
    history = [board.copy()]
    examples = []
    move_count = 0

    # Start with fresh root
    root = create_root_node(board, history)

    while not board.is_game_over() and move_count < max_moves:
        # Run MCTS
        search(root, model, num_simulations, device=device)

        # Store example
        state = encode_board(board, history)
        policy = extract_policy(root)
        examples.append({"state": state, "policy": policy, "player": "white" if board.turn == chess.WHITE else "black"})

        # Sample move
        temperature = 1.0 if move_count < 30 else 0.1
        move = sample_move(root, temperature)

        # Tree reuse - just move to child!
        root = root.children[move]

        # Make move
        board.push(move)
        history.append(board.copy())
        move_count += 1

    # Create training examples
    return create_training_examples(examples, board)


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


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load or create model
    from src.nn import AlphaZeroNet

    model = AlphaZeroNet(num_res_blocks=10, num_channels=128, input_planes=106).to(device)

    # Play a self-play game
    training_data = play_automated_game(model, num_simulations=800, device=device)

    print(f"Generated {len(training_data)} training examples from self-play game.")
