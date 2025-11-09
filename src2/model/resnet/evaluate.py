from typing import Callable, Iterable

import chess
import torch
import torch.nn.functional as F

from src2.model.resnet.board_encoding import encode_board
from src2.model.resnet.move_translation import encode_move
from src2.model.resnet.nn import model as default_model

# perhaps evaluate should have a protocol to define a clean interface for swapping evaluations?


def evaluate_boards(
    boards_and_histories: list[tuple[chess.Board, list[chess.Board]]],
    model: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = default_model,
) -> Iterable[tuple[float, dict[chess.Move, float]]]:
    with torch.no_grad():
        encoded_boards = [encode_board(board, history) for board, history in boards_and_histories]
        cuda_board_states: torch.Tensor = torch.stack(encoded_boards).to("cuda")
        policy_logits, values = model(cuda_board_states)
        policies = F.softmax(policy_logits, dim=1).cpu()
        values = values.cpu()
        values = [v.item() for v in values]
        pol = []
        for i, (board, _) in enumerate(boards_and_histories):
            pol.append({})
            for move in board.legal_moves:
                move_idx = encode_move(move)
                move_prob = policies[i][move_idx].item()
                pol[i][move] = move_prob
        return zip(values, pol)
