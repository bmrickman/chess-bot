"""
CustomBoard - Lightweight chess board that stores only FEN string
"""

from typing import Any

import chess


class CustomBoard:
    """
    Lightweight board wrapper that stores only FEN string

    Proxies all method calls to a temporary chess.Board instance
    Memory: ~80 bytes (just the FEN string)
    """

    __slots__ = ("_fen",)

    def __init__(self, fen: str = chess.STARTING_FEN):
        """
        Initialize from FEN string

        Args:
            fen: FEN string representing the position
        """
        self._fen = fen

    @classmethod
    def from_board(cls, board: chess.Board) -> "CustomBoard":
        """Create CustomBoard from chess.Board"""
        return cls(board.fen())

    @property
    def fen(self) -> str:
        """Get FEN string"""
        return self._fen

    def push(self, move: chess.Move) -> "CustomBoard":
        board = self._get_board()
        board.push(move)
        return CustomBoard(board.fen())

    def _get_board(self) -> chess.Board:
        """Create temporary chess.Board from FEN"""
        return chess.Board(self._fen)

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all attribute access to a temporary board

        This allows: board.legal_moves, board.push(move), etc.
        """
        board = self._get_board()
        attr = getattr(board, name)

        # If it's a method that modifies the board, we need to update our FEN
        if callable(attr):

            def wrapper(*args, **kwargs):
                # Call the method
                result = attr(*args, **kwargs)

                # Update our FEN if the board was modified
                # (Methods like push, pop, etc. modify the board)
                self._fen = board.fen()

                return result

            return wrapper

        # For non-callable attributes (properties, etc.), just return them
        return attr

    def __str__(self) -> str:
        """String representation"""
        return self._get_board().__str__()

    def __repr__(self) -> str:
        """Repr"""
        return f"CustomBoard('{self._fen}')"

    def copy(self) -> "CustomBoard":
        """Create a copy"""
        return CustomBoard(self._fen)

    def __eq__(self, other) -> bool:
        """Check equality"""
        if isinstance(other, CustomBoard):
            return self._fen == other._fen
        elif isinstance(other, chess.Board):
            return self._fen == other.fen()
        return False

    def __hash__(self):
        """Make hashable"""
        return hash(self._fen)


# Convenience function for creating from existing board
def to_custom_board(board: chess.Board) -> CustomBoard:
    """Convert chess.Board to CustomBoard"""
    return CustomBoard.from_board(board)
