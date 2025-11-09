from typing import Dict, List

import chess
import torch


def encode_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
    """
    Encode board to 106 planes as a PyTorch tensor

    Args:
        board: Current chess board position
        history: Previous board positions

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
