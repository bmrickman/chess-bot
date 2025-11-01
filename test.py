"""
Test suite for MCTS implementation

Run with: pytest test_mcts.py -v
"""

import math
from dataclasses import dataclass
from typing import Tuple

import chess
import pytest
import torch

from self_play import play_self_play_game

# Import your MCTS implementation
from src.mcts import (
    Node,
    create_child_node,
    create_root_node,
    expand_node,
    extract_policy,
    get_terminal_value,
    get_ucb_score,
    get_value,
    is_terminal,
    sample_move,
    search,
    select_best_child,
    update_node,
)

# ============================================================================
# Mock Model for Testing
# ============================================================================


class MockModel(torch.nn.Module):
    """Mock neural network for testing"""

    def __init__(self, policy_mode="uniform", value=0.0):
        super().__init__()
        self.policy_mode = policy_mode
        self.value = value
        self.call_count = 0

    def forward(self, state_batch):
        """
        Return mock policy and value

        Args:
            state_batch: [batch_size, 119, 8, 8]

        Returns:
            policy_logits: [batch_size, 4864]
            values: [batch_size, 1]
        """
        self.call_count += 1
        batch_size = state_batch.shape[0]

        # Mock policy logits
        if self.policy_mode == "uniform":
            policy_logits = torch.zeros(batch_size, 4864)
        elif self.policy_mode == "favor_first":
            policy_logits = torch.zeros(batch_size, 4864)
            policy_logits[:, 0] = 2.0  # Higher logit for first move
        else:
            policy_logits = torch.randn(batch_size, 4864)

        # Mock value
        values = torch.full((batch_size, 1), self.value, dtype=torch.float32)

        return policy_logits, values

    def eval(self):
        return self


# ============================================================================
# Test Node Operations
# ============================================================================


class TestNodeOperations:
    """Test basic node operations"""

    def test_create_root_node(self):
        """Test root node creation"""
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)

        assert root.visit_count == 0
        assert root.value_sum == 0.0
        assert root.prior_prob == 0.0
        assert len(root.children) == 0
        assert root.board.fen() == board.fen()

    def test_create_child_node(self):
        """Test child node creation"""
        board = chess.Board()
        history = [board.copy()]
        root = create_root_node(board, history)

        move = chess.Move.from_uci("e2e4")
        child = create_child_node(root, move, prior_prob=0.5)

        assert child.visit_count == 0
        assert child.prior_prob == 0.5
        assert child.board.fen() != root.board.fen()
        assert len(child.history) == len(root.history) + 1

    def test_get_value(self):
        """Test value calculation"""
        board = chess.Board()
        history = [board.copy()]
        node = create_root_node(board, history)

        # No visits -> 0 value
        assert get_value(node) == 0.0

        # Add some visits
        node.visit_count = 10
        node.value_sum = 5.0
        assert get_value(node) == 0.5

        # Negative values work
        node.value_sum = -3.0
        assert get_value(node) == -0.3

    def test_update_node(self):
        """Test node update"""
        board = chess.Board()
        history = [board.copy()]
        node = create_root_node(board, history)

        assert node.visit_count == 0
        assert node.value_sum == 0.0

        update_node(node, 0.5)
        assert node.visit_count == 1
        assert node.value_sum == 0.5

        update_node(node, -0.3)
        assert node.visit_count == 2
        assert node.value_sum == 0.2

    def test_get_ucb_score(self):
        """Test UCB score calculation"""
        board = chess.Board()
        history = [board.copy()]
        node = create_root_node(board, history)
        node.prior_prob = 0.5

        # Unvisited child, parent has 100 visits
        ucb = get_ucb_score(node, parent_visits=100, c_puct=1.5)

        # Q = 0 (no visits), U = 1.5 * 0.5 * sqrt(100) / 1 = 7.5
        expected = 0.0 + 7.5
        assert abs(ucb - expected) < 0.001

        # Child with visits
        node.visit_count = 10
        node.value_sum = 3.0
        ucb = get_ucb_score(node, parent_visits=100, c_puct=1.5)

        # Q = 3.0/10 = 0.3, U = 1.5 * 0.5 * sqrt(100) / 11 = 0.682
        expected = 0.3 + 0.682
        assert abs(ucb - expected) < 0.001

    def test_select_best_child(self):
        """Test child selection"""
        board = chess.Board()
        history = [board.copy()]
        root = create_root_node(board, history)
        root.visit_count = 100

        # Create children with different priors
        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("d2d4")

        child1 = create_child_node(root, move1, prior_prob=0.6)
        child2 = create_child_node(root, move2, prior_prob=0.4)

        root.children[move1] = child1
        root.children[move2] = child2

        # Child1 has higher prior, should be selected first
        selected_move, selected_child = select_best_child(root, c_puct=1.5)
        assert selected_move == move1

        # After child1 gets visits, child2 should be selected
        child1.visit_count = 50
        child1.value_sum = 5.0

        selected_move, selected_child = select_best_child(root, c_puct=1.5)
        # Child2 has 0 visits and will have higher UCB due to exploration
        assert selected_move == move2


# ============================================================================
# Test Tree Operations
# ============================================================================


class TestTreeOperations:
    """Test tree expansion and terminal detection"""

    def test_expand_node(self):
        """Test node expansion"""
        board = chess.Board()
        history = [board.copy()]
        root = create_root_node(board, history)

        # Mock policy probs
        policy_probs = torch.ones(4864) / 4864

        expand_node(root, policy_probs)

        # Starting position has 20 legal moves
        assert len(root.children) == 20

        # All children should have prior probs
        for move, child in root.children.items():
            assert child.prior_prob > 0
            assert child.visit_count == 0

    def test_is_terminal(self):
        """Test terminal detection"""
        # Non-terminal position
        board = chess.Board()
        history = [board.copy()]
        root = create_root_node(board, history)

        assert not is_terminal(root)

        # Checkmate position
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        board.push(chess.Move.from_uci("h5f7"))  # Checkmate
        checkmate_node = create_root_node(board, [board.copy()])

        assert is_terminal(checkmate_node)

    def test_get_terminal_value(self):
        """Test terminal value extraction"""
        # White wins
        board = chess.Board()
        board.set_fen("k7/8/K7/8/8/8/8/4R3 w - - 0 1")
        board.push(chess.Move.from_uci("e1e8"))  # Checkmate
        node = create_root_node(board, [board.copy()])

        value = get_terminal_value(node)
        assert value == 1.0  # White wins

        # Stalemate
        board = chess.Board()
        board.set_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # Stalemate
        node = create_root_node(board, [board.copy()])

        value = get_terminal_value(node)
        assert value == 0.0  # Draw


# ============================================================================
# Test MCTS Search
# ============================================================================


class TestMCTSSearch:
    """Test MCTS search functionality"""

    def test_search_basic(self):
        """Test basic search"""
        model = MockModel(policy_mode="uniform", value=0.1)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=10, device="cpu")

        # Root should have visits
        assert root.visit_count == 10

        # Root should have children
        assert len(root.children) > 0

        # Children should have some visits
        total_child_visits = sum(c.visit_count for c in root.children.values())
        assert total_child_visits > 0

        # Model should have been called
        assert model.call_count > 0

    def test_search_determinism(self):
        """Test that search is deterministic with same seed"""
        torch.manual_seed(42)
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root1 = create_root_node(board, history)
        search(root1, model, num_simulations=50, device="cpu")
        visits1 = {move.uci(): child.visit_count for move, child in root1.children.items()}

        torch.manual_seed(42)
        model = MockModel(policy_mode="uniform", value=0.0)
        root2 = create_root_node(board, history)
        search(root2, model, num_simulations=50, device="cpu")
        visits2 = {move.uci(): child.visit_count for move, child in root2.children.items()}

        # Should get same visit counts with same seed
        assert visits1 == visits2

    def test_search_accumulation(self):
        """Test that multiple searches accumulate visits"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)

        # First search
        search(root, model, num_simulations=10, device="cpu")
        assert root.visit_count == 10

        # Second search
        search(root, model, num_simulations=10, device="cpu")
        assert root.visit_count == 20

        # Third search
        search(root, model, num_simulations=30, device="cpu")
        assert root.visit_count == 50

    def test_tree_reuse(self):
        """Test tree reuse between moves"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        # First search
        root = create_root_node(board, history)
        search(root, model, num_simulations=100, device="cpu")

        # Get a move that was explored
        best_move = sample_move(root, temperature=0.0)
        child_visits_before = root.children[best_move].visit_count

        # Move to child (tree reuse)
        root = root.children[best_move]

        # Child should have visits from previous search
        assert root.visit_count == child_visits_before
        assert root.visit_count > 0

        # Search again
        search(root, model, num_simulations=100, device="cpu")

        # Should have accumulated visits
        assert root.visit_count == child_visits_before + 100


# ============================================================================
# Test Policy Extraction
# ============================================================================


class TestPolicyExtraction:
    """Test policy extraction functions"""

    def test_extract_policy(self):
        """Test full policy tensor extraction"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=100, device="cpu")

        policy = extract_policy(root)

        # Should be [4864] tensor
        assert policy.shape == (4864,)

        # Should sum to 1.0
        assert abs(policy.sum().item() - 1.0) < 0.001

        # Should have 20 non-zero entries (20 legal moves)
        non_zero = (policy > 0).sum().item()
        assert non_zero == 20

    def test_get_best_move(self):
        """Test best move selection"""
        model = MockModel(policy_mode="favor_first", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=100, device="cpu")

        best_move = sample_move(root, temperature=0.0)

        # Should return a legal move
        assert best_move in board.legal_moves

        # Should be the most visited child
        most_visited = max(root.children.values(), key=lambda c: c.visit_count)
        assert root.children[best_move] == most_visited


# ============================================================================
# Test Move Sampling
# ============================================================================


class TestMoveSampling:
    """Test move sampling with temperature"""

    def test_sample_move_greedy(self):
        """Test greedy move selection (temperature=0)"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=100, device="cpu")

        # Sample 10 times with temperature=0
        moves = [sample_move(root, temperature=0.0) for _ in range(10)]

        # All should be the same (greedy)
        assert len(set(move.uci() for move in moves)) == 1

        # Should match get_best_move
        best = sample_move(root, temperature=0.0)
        assert all(move == best for move in moves)

    def test_sample_move_stochastic(self):
        """Test stochastic move selection (temperature=1.0)"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=200, device="cpu")

        # Sample 100 times with temperature=1.0
        torch.manual_seed(42)
        moves = [sample_move(root, temperature=1.0) for _ in range(100)]

        # Should have variety (not all the same)
        unique_moves = set(move.uci() for move in moves)
        assert len(unique_moves) > 1

        # All should be legal
        legal = set(board.legal_moves)
        assert all(move in legal for move in moves)

    def test_temperature_effect(self):
        """Test that temperature affects exploration"""
        model = MockModel(policy_mode="uniform", value=0.0)
        board = chess.Board()
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=200, device="cpu")

        # Low temperature (T=0.1): Should be more deterministic
        torch.manual_seed(42)
        moves_low = [sample_move(root, temperature=0.1) for _ in range(50)]
        unique_low = len(set(move.uci() for move in moves_low))

        # High temperature (T=2.0): Should be more exploratory
        torch.manual_seed(42)
        moves_high = [sample_move(root, temperature=2.0) for _ in range(50)]
        unique_high = len(set(move.uci() for move in moves_high))

        # High temperature should explore more moves
        assert unique_high >= unique_low


# ============================================================================
# Test Self-Play Integration
# ============================================================================


class TestSelfPlayIntegration:
    """Test full self-play game"""

    def test_play_short_game(self):
        """Test playing a short game"""
        model = MockModel(policy_mode="uniform", value=0.0)

        training_examples = play_self_play_game(
            model,
            num_simulations=10,
            device="cpu",
            max_moves=5,  # Short game for testing
        )

        # Should generate some training examples
        assert len(training_examples) > 0
        assert len(training_examples) <= 5

        # Each example should be a tuple of 3 tensors
        for state, policy, outcome in training_examples:
            assert state.shape == (119, 8, 8)
            assert policy.shape == (4864,)
            assert outcome.shape == (1, 1)

            # Policy should sum to 1
            assert abs(policy.sum().item() - 1.0) < 0.001

            # Outcome should be -1, 0, or 1
            assert outcome.item() in [-1.0, 0.0, 1.0]

    def test_game_ends(self):
        """Test that game terminates"""
        model = MockModel(policy_mode="uniform", value=0.0)

        # Game should end within max_moves
        training_examples = play_self_play_game(model, num_simulations=10, device="cpu", max_moves=200)

        # Should have examples from the game
        assert len(training_examples) > 0
        assert len(training_examples) <= 200


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_root(self):
        """Test behavior with root that has no children"""
        board = chess.Board()
        history = [board.copy()]
        root = create_root_node(board, history)

        # Root has no children yet
        assert len(root.children) == 0

    def test_terminal_position_search(self):
        """Test search on terminal position"""
        model = MockModel(policy_mode="uniform", value=0.0)

        # Checkmate position
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        board.push(chess.Move.from_uci("f7f8"))  # Checkmate
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=10, device="cpu")

        # Should recognize it's terminal
        assert root.visit_count == 10

        # Should not expand (no legal moves)
        assert len(root.children) == 0

    def test_single_legal_move(self):
        """Test position with only one legal move"""
        model = MockModel(policy_mode="uniform", value=0.0)

        # Position with only one legal move
        board = chess.Board("7k/8/6K1/8/8/8/7Q/8 w - - 0 1")
        history = [board.copy()]

        root = create_root_node(board, history)
        search(root, model, num_simulations=10, device="cpu")

        # Should work fine with one move
        assert len(root.children) >= 1

        best = sample_move(root, temperature=0.0)
        assert best in board.legal_moves


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
