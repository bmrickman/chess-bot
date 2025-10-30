"""
Move encoding for AlphaZero chess.

Encodes chess moves to policy indices [0-4863].
Encoding: 64 squares × 76 move types = 4864 possible moves

Move types (76 total):
- Queen moves: 8 directions × 7 distances = 56
- Knight moves: 8 L-shapes = 8
- Promotions: 3 directions × 4 pieces = 12
"""

from typing import Dict, Optional, Tuple

import chess
import torch

# Module-level encoding dictionaries
_move_to_idx: Dict[Tuple[int, int, Optional[int]], int] = {}
_idx_to_pattern: Dict[int, Tuple[int, int, Optional[int]]] = {}


def _build_encoding() -> None:
    """Build move encoding lookup tables (called at module load)"""
    move_type = 0

    # 1. Queen moves (56 types): 8 directions × 7 distances
    directions = [
        (0, 1),  # North
        (0, -1),  # South
        (1, 0),  # East
        (-1, 0),  # West
        (1, 1),  # Northeast
        (1, -1),  # Southeast
        (-1, 1),  # Northwest
        (-1, -1),  # Southwest
    ]

    for direction in directions:
        for distance in range(1, 8):
            _idx_to_pattern[move_type] = (direction[0] * distance, direction[1] * distance, None)
            move_type += 1

    # 2. Knight moves (8 types)
    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

    for knight_move in knight_moves:
        _idx_to_pattern[move_type] = (knight_move[0], knight_move[1], None)
        move_type += 1

    # 3. Promotions (12 types): 3 directions × 4 pieces
    promotion_directions = [
        (-1, 1),  # Forward-left
        (0, 1),  # Forward
        (1, 1),  # Forward-right
    ]

    promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    for direction in promotion_directions:
        for piece in promotion_pieces:
            _idx_to_pattern[move_type] = (direction[0], direction[1], piece)
            move_type += 1

    # Build reverse mapping: (from_square, to_square, promotion) -> index
    for from_square in chess.SQUARES:
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)

        for move_type, (file_delta, rank_delta, promotion) in _idx_to_pattern.items():
            to_file = from_file + file_delta
            to_rank = from_rank + rank_delta

            # Check if target square is valid
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square = chess.square(to_file, to_rank)
                key = (from_square, to_square, promotion)
                _move_to_idx[key] = from_square * 76 + move_type


# Build encoding when module is imported
_build_encoding()


def encode_move(move: chess.Move) -> int:
    """
    Encode a chess move to a policy index

    Args:
        move: Chess move to encode

    Returns:
        Policy index in range [0, 4863]

    Raises:
        ValueError: If move cannot be encoded
    """
    key = (move.from_square, move.to_square, move.promotion)

    if key not in _move_to_idx:
        raise ValueError(f"Cannot encode move {move.uci()}")

    return _move_to_idx[key]


def decode_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Decode a policy index to a chess move

    Args:
        index: Policy index in range [0, 4863]
        board: Current board position (for checking legality)

    Returns:
        Chess move if legal, None otherwise
    """
    if not (0 <= index < 4864):
        return None

    from_square = index // 76
    move_type = index % 76

    if move_type not in _idx_to_pattern:
        return None

    file_delta, rank_delta, promotion = _idx_to_pattern[move_type]

    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)

    to_file = from_file + file_delta
    to_rank = from_rank + rank_delta

    # Check bounds
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None

    to_square = chess.square(to_file, to_rank)
    move = chess.Move(from_square, to_square, promotion=promotion)

    # Check if move is legal
    if move in board.legal_moves:
        return move

    return None
