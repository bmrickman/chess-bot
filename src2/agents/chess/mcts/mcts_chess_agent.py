from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Optional

import chess

from src2.policy.mcts.mcts import MCTS, MCTSNode
from src2.types import Agent


@dataclass
class ChessNode(MCTSNode[tuple[chess.Board, list[chess.Board]], chess.Move]):
    state: tuple[chess.Board, list[chess.Board]]  # (current_board, board_history)
    prior_prob: float = 0.0
    children: dict[chess.Move, "ChessNode"] = field(default_factory=dict)

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


## This agent composes:
# The MCTS Policy
# A chess Node (Node implementation required for MCTS policy)
# an chess state inference method
class MCTSChessAgent(Agent[chess.Board, chess.Move]):
    def __init__(
        self,
        starting_state: chess.Board,
        c_puct: float = 1.5,
        sims_per_move: int = 800,
        inference_queues: Optional[tuple[Queue, Queue]] = None,
    ):
        # Initialize with empty history
        initial_history: list[chess.Board] = []
        self.node = ChessNode(state=(starting_state, initial_history), prior_prob=0.0)

        if inference_queues:

            def evaluate_fn(state: tuple[chess.Board, list[chess.Board]]) -> tuple[dict[chess.Move, float], float]:
                # Send state to inference server
                input_queue, output_queue = inference_queues
                input_queue.put(state)
                policy, value = output_queue.get()
                return policy, value
        else:
            from src2.model.chess_resnet.evaluate_chess_board_and_history import evaluate_boards_and_histories

            def evaluate_fn(state: tuple[chess.Board, list[chess.Board]]) -> tuple[dict[chess.Move, float], float]:
                evaluations = evaluate_boards_and_histories([state])
                policy_probs, state_value = list(evaluations)[0]
                return state_value, policy_probs

        self.mcts = MCTS[chess.Move, tuple[chess.Board, list[chess.Board]]](
            evaluate=evaluate_fn, node_type=ChessNode, c_puct=c_puct, sims_per_move=sims_per_move
        )

        self.mcts.simulate(self.node)

    def select_action(self) -> chess.Move:
        move, child = self.mcts.select_best_move_and_child(self.node)
        return move

    def update_state(self, state: chess.Board):
        for move, child in self.node.children.items():
            if child.current_board.fen() == state.fen():
                self.node = child
                self.mcts.simulate(self.node)
                return
        raise Exception("No child node board matches the given board")
