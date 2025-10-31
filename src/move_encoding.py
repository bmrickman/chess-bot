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

# Module-level encoding dictionaries
_encode: Dict[Tuple[int, int, Optional[int]], int] = {}
_decode: Dict[int, Tuple[int, int, Optional[int]]] = {}


def _build_encoding() -> None:
    """Build move encoding lookup tables (called at module load)"""

    # Define move patterns
    move_patterns = []

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
            move_patterns.append((direction[0] * distance, direction[1] * distance, None))

    # 2. Knight moves (8 types)
    knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

    for knight_move in knight_moves:
        move_patterns.append((knight_move[0], knight_move[1], None))

    # 3. Promotions (12 types): 3 directions × 4 pieces
    promotion_directions = [(-1, 1), (0, 1), (1, 1)]  # Forward-left, Forward, Forward-right
    promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    for direction in promotion_directions:
        for piece in promotion_pieces:
            move_patterns.append((direction[0], direction[1], piece))

    # Build encoding: (from_square, to_square, promotion) -> policy_index
    for from_square in chess.SQUARES:
        from_rank = chess.square_rank(from_square)
        from_file = chess.square_file(from_square)

        for move_pattern_idx, (file_delta, rank_delta, promotion) in enumerate(move_patterns):
            to_file = from_file + file_delta
            to_rank = from_rank + rank_delta

            # Check if target square is valid
            if 0 <= to_file < 8 and 0 <= to_rank < 8:
                to_square = chess.square(to_file, to_rank)
                policy_index = int(from_square) * 76 + move_pattern_idx

                _encode[(from_square, to_square, promotion)] = policy_index

    # Reverse the encoding to get decoding
    _decode = {v: k for k, v in _encode.items()}


# Build encoding when module is imported
_build_encoding()


def encode_move(move: chess.Move) -> int:
    key = (move.from_square, move.to_square, move.promotion)
    if key not in _encode:
        raise ValueError(f"Cannot encode move {move.uci()}")
    return _encode[key]


def decode_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    if index not in _decode:
        return None
    from_square, to_square, promotion = _decode[index]
    move = chess.Move(from_square, to_square, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None
