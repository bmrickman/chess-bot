import chess

from src2.environments.chess_environment import ChessEnvironment
from src2.gameplay.chess.mcts_chess_agent import MCTSChessAgent


def chess_self_play():
    environment = ChessEnvironment()
    player1 = MCTSChessAgent(starting_state=chess.Board())
    player2 = MCTSChessAgent(starting_state=chess.Board())
    while not environment.game_over():
        pass
