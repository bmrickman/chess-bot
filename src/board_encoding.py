from typing import Dict, List

import chess
import torch


def encode_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
    """
    Encode board to 106 planes as a PyTorch tensor

    Args:
        board: Current chess board position
        history: Previous board positions (excluding current), including starting position

    Returns:
        torch.Tensor of shape [106, 8, 8] with dtype float32
    """
    # Initialize tensor directly
    planes: torch.Tensor = torch.zeros((106, 8, 8), dtype=torch.float32)

    piece_indices: Dict[int, int] = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Planes 0-11: Current position (6 piece types × 2 colors)
    for square in chess.SQUARES:
        piece: chess.Piece | None = board.piece_at(square)
        if piece:
            rank: int = chess.square_rank(square)
            file: int = chess.square_file(square)

            plane_idx: int = piece_indices[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6

            planes[plane_idx, rank, file] = 1.0

    # Planes 12-95: Previous 7 positions (84 planes = 7 × 12)
    recent_history: List[chess.Board] = history[-7:] if len(history) > 7 else history

    for i, prev_board in enumerate(recent_history):
        plane_offset: int = 12 + i * 12

        for square in chess.SQUARES:
            piece = prev_board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                plane_idx = piece_indices[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6

                planes[plane_offset + plane_idx, rank, file] = 1.0

    # Planes 96-99: Castling rights (fill entire 8x8 plane with 0 or 1)
    planes[96, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[97, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[98, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[99, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Planes 100-101: Repetition counters
    repetition_count: int = sum(1 for prev_board in history if prev_board.fen() == board.fen())
    if repetition_count == 0:
        planes[100, :, :] = 1.0
    elif repetition_count == 1:
        planes[101, :, :] = 1.0
    # If repetition_count >= 2, both planes stay 0

    # Plane 102: Move count (halfmove clock for fifty-move rule, normalized)
    planes[102, :, :] = board.halfmove_clock / 100.0

    # Plane 103: En passant square
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[103, rank, file] = 1.0

    # Plane 104: Color to move (1 for white, 0 for black)
    planes[104, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Plane 105: Total move count (fullmove number, normalized)
    planes[105, :, :] = board.fullmove_number / 200.0

    # Planes 106-118: Reserved for additional features

    return planes


# def get_canonical_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
#     """
#     Get board state from current player's perspective (flip if black to move)

#     Args:
#         board: Current chess board position
#         history: List of previous board positions

#     Returns:
#         torch.Tensor of shape [106, 8, 8] encoded from current player's perspective
#     """
#     state: torch.Tensor = encode_board(board, history)

#     # If black to move, flip the board vertically
#     if board.turn == chess.BLACK:
#         state = torch.flip(state, dims=[1])  # Flip along rank axis

#     return state


# Example usage
if __name__ == "__main__":
    # Create a board and history
    board: chess.Board = chess.Board()
    history: List[chess.Board] = [board.copy()]  # Start with initial position

    # Make some moves
    board.push_san("e4")
    history.append(board.copy())
    board.push_san("e5")
    history.append(board.copy())

    # Encode to tensor (ready for neural network!)
    state_tensor: torch.Tensor = encode_board(board, history)

    print(f"Type: {type(state_tensor)}")  # <class 'torch.Tensor'>
    print(f"Shape: {state_tensor.shape}")  # torch.Size([106, 8, 8])
    print(f"Dtype: {state_tensor.dtype}")  # torch.float32

    # Can directly use with model (just add batch dimension)
    from nn import AlphaZeroNet

    model = AlphaZeroNet()

    # Add batch dimension
    batch_state = state_tensor.unsqueeze(0)  # Shape: [1, 106, 8, 8]

    # Forward pass - no conversion needed!
    policy_logits, value = model(batch_state)

    print(f"Policy shape: {policy_logits.shape}")  # [1, 4864]
    print(f"Value shape: {value.shape}")  # [1, 1]
