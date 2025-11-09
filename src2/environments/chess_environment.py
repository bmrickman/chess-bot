import chess

from src2.types import Environment


class ChessEnvironment(Environment[chess.Board, chess.Move]):
    def __init__(self, starting_state: chess.Board = chess.Board()):
        self.state = starting_state

    def get_visible_state(self) -> chess.Board:
        return self.state

    def apply_action(self, action: chess.Move):
        self.state.push(action)

    def game_over(self):
        return self.state.is_game_over()
