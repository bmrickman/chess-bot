import chess

from src2.agents.chess.mcts.mcts_chess_agent import MCTSChessAgent


def test_agent():
    board = chess.Board()
    history = []
    agent = MCTSChessAgent(starting_state=board.copy(stack=False), inference_queues=None)
    for i in range(10):
        move = agent.select_action()
        assert isinstance(move, chess.Move)
        history.append(board.copy(stack=False))
        board.push(move)
        agent.update_state(state=board)
    assert agent.node.current_board.fen() == board.fen()
    assert agent.node.board_history == history[-6:]
