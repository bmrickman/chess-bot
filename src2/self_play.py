import chess
from environments.chess_environment import ChessEnvironment

from src2.types import Agent, Environment


def chess_self_play() -> None:
    environment = ChessEnvironment()
    player1 = MCTSChessAgent(starting_state=chess.Board())
    player2 = MCTSChessAgent(starting_state=chess.Board())
    while not environment.game_over():
        pass
