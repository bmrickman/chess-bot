from math import exp

import chess
import torch
import torch.nn.functional as F

from src2.model.board_encoding import encode_board
from src2.model.evaluate import evaluate_boards
from src2.model.move_translation import encode_move


class testBoard(chess.Board):
    @property
    def legal_moves(self):
        return iter([chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")])


board = testBoard()
board_history = []
encoded_board_state = encode_board(board, board_history)
board_state_value = 0.75
policy_tensor = torch.ones(4672)
policy_tensor[encode_move(chess.Move.from_uci("e2e4"))] = 100


def dummy_model(cuda_board_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(cuda_board_states) == 1
    encoded_board = cuda_board_states.cpu()[0]
    assert torch.equal(encoded_board, encode_board(board, board_history))
    value_tensor = torch.tensor([[board_state_value]])
    return policy_tensor.unsqueeze(0), value_tensor.unsqueeze(0)


def test_evaluate():
    evaluations = list(evaluate_boards([(board, board_history)], dummy_model))
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    state_value = evaluation[0]
    assert state_value == 0.75
    policy_probs = evaluation[1]
    assert policy_probs[chess.Move.from_uci("e2e4")] == exp(100) / (sum([exp(1)] * 4671) + exp(100))
