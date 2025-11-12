from dataclasses import dataclass

import chess

from src2.model.chess_resnet.evaluate_chess_board_and_history import evaluate_boards_and_histories
from src2.policy.mcts.mcts import MCTS, MCTSNode
from src2.types import Agent


@dataclass
class ChessNode(MCTSNode[tuple[chess.Board, list[chess.Board]], chess.Move]):
    state: tuple[chess.Board, list[chess.Board]]  # (current_board, board_history)
    prior_prob: float = 0.0

    @property
    def current_board(self) -> chess.Board:
        return self.state[0]

    @property
    def board_history(self) -> list[chess.Board]:
        return self.state[1]

    def is_terminal(self) -> bool:
        return self.current_board.is_game_over()

    def terminal_value(self) -> float:
        result = self.current_board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    def legal_moves(self) -> list[chess.Move]:
        return list(self.current_board.legal_moves)

    def apply_move(self, move: chess.Move) -> "ChessNode":
        new_board = self.current_board.copy(stack=False)
        new_board.push(move)

        # Update history: add current board and keep last 6 (including the new one)
        new_history = self.board_history + [self.current_board]
        if len(new_history) > 6:
            new_history = new_history[-6:]

        return ChessNode(state=(new_board, new_history), prior_prob=0.0)


class MCTSChessAgent(Agent[chess.Board, chess.Move]):
    def __init__(self, starting_state: chess.Board):
        # Initialize with empty history
        initial_history: list[chess.Board] = []
        self.node = ChessNode(state=(starting_state, initial_history), prior_prob=0.0)

        def evaluate_single_state(
            state: tuple[chess.Board, list[chess.Board]],
        ) -> tuple[dict[chess.Move, float], float]:
            # Adapt batch evaluation to single state evaluation
            board, history = state
            results = list(evaluate_boards_and_histories([(board, history)]))
            value, policy = results[0]  # Unpack the single result
            return policy, value  # MCTS expects (policy, value) order

        self.mcts = MCTS[chess.Move, tuple[chess.Board, list[chess.Board]]](
            evaluate=evaluate_single_state, node_type=ChessNode, c_puct=1.5, sims_per_move=800
        )

    def select_action(self) -> chess.Move:
        move, child = self.mcts.select_best_move_and_child(self.node)
        return move

    def update_state(self, state: chess.Board):
        for child in self.node.children.values():
            if isinstance(child, ChessNode) and child.current_board.fen() == state.fen():
                self.node = child
                self.mcts.simulate(self.node)
                return
        raise Exception("something has gone wrong")
