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

index = 0

# Part 1: All (from, to) pairs for non-promotion moves
# Indices 0-4095
for from_square in range(64):
    for to_square in range(64):
        _encode[(from_square, to_square, None)] = index
        _decode[index] = (from_square, to_square, None)
        index += 1

# Part 2: Promotion moves (from_square, to_square, piece)
# Only valid for moves to rank 8 (squares 56-63) or rank 1 (squares 0-7)
promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Promotions to rank 8 (White pawns)
for from_square in range(48, 56):  # Rank 7 (squares 48-55)
    for to_square in range(56, 64):  # Rank 8 (squares 56-63)
        # Only encode if it's a valid pawn move (forward or diagonal capture)
        from_file = from_square % 8
        to_file = to_square % 8

        # Valid if: same file (forward) or adjacent file (capture)
        if abs(from_file - to_file) <= 1:
            for piece in promotion_pieces:
                _encode[(from_square, to_square, piece)] = index
                _decode[index] = (from_square, to_square, piece)
                index += 1

# Promotions to rank 1 (Black pawns) - encoded from Black's perspective
for from_square in range(8, 16):  # Rank 2 (squares 8-15)
    for to_square in range(0, 8):  # Rank 1 (squares 0-7)
        from_file = from_square % 8
        to_file = to_square % 8

        if abs(from_file - to_file) <= 1:
            for piece in promotion_pieces:
                _encode[(from_square, to_square, piece)] = index
                _decode[index] = (from_square, to_square, piece)
                index += 1

print(f"Total move encodings: {index}")


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
