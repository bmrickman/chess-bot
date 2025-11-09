from dataclasses import dataclass

import chess

from src2.model.resnet.evaluate import evaluate_boards
from src2.policy.mcts.mcts import MCTS, Node
from src2.types import Agent


@dataclass
class ChessNode(Node[chess.Board, chess.Move]):
    state: chess.Board
    prior_prob: float

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def terminal_value(self) -> float:
        result = self.state.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    def legal_moves(self) -> list[chess.Move]:
        return list(self.state.legal_moves)

    def apply_move(self, move: chess.Move, prior_prob: float) -> "ChessNode":
        newstate = self.state.copy(stack=False)
        newstate.push(move)
        return ChessNode(state=newstate, prior_prob=prior_prob)


class MCTSChessAgent(Agent[chess.Board, chess.Move]):
    def __init__(self, starting_state: chess.Board):
        self.node = ChessNode(state=starting_state, prior_prob=0.0)
        self.mcts = MCTS(evaluate=evaluate_boards, node_type=ChessNode, c_puct=1.5, sims_per_move=800)

    def select_action(self) -> chess.Move:
        move, child = self.mcts.select_best_move_and_child(self.node)
        return move

    def update_state(self, state: chess.Board):
        for child in self.node.children.values():
            if child.state.fen() == state.fen():
                self.node = child
                return
        raise Exception("something has gone wrong")
