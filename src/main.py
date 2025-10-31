import chess
import torch

from board_encoding import encode_board
from nn import AlphaZeroNet

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroNet(num_res_blocks=10, num_channels=128, input_planes=106)
    board = chess.Board()
    history = []
    state = encode_board(board, history)
    batched_state = state.unsqueeze(0)
    policy_logits, value = model(batched_state)  # inference step
    print("here")
