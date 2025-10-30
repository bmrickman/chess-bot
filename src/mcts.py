import math
from typing import Tuple

import chess
import torch
import torch.nn.functional as F

from board_encoder import encode_board
from nn import AlphaZeroNet
from src.move_encoding import encode_move


class MCTSNode:
    """Node in the MCTS search tree"""

    def __init__(self, prior_prob: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.children: dict[chess.Move, "MCTSNode"] = {}

    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Upper confidence bound for tree search"""
        q_value = self.value()
        u_value = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + u_value


class MCTS:
    """Monte Carlo Tree Search with neural network guidance"""

    def __init__(self, model: AlphaZeroNet, num_simulations: int = 800, c_puct: float = 1.5, device: str = "cuda"):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _evaluate(self, board: chess.Board, history: list[chess.Board]) -> Tuple[torch.Tensor, float]:
        """
        Evaluate position with neural network

        Args:
            board: Current position
            history: Previous positions

        Returns:
            (policy_probs, value) - [4672] tensor and scalar
        """
        # Encode board
        state = encode_board(board, history)  # [119, 8, 8]

        # Add batch dimension and move to device
        state_batch = state.unsqueeze(0).to(self.device)  # [1, 119, 8, 8]

        # Get NN predictions
        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_batch)

        # Convert to probabilities and extract value
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0)  # [4672]
        value = value_tensor.item()  # Scalar float

        return policy_probs, value

    def search(self, board: chess.Board, history: list[chess.Board]) -> torch.Tensor:
        """
        Run MCTS and return improved policy

        Args:
            board: Current position
            history: Previous positions

        Returns:
            policy: [4672] tensor with move probabilities based on visit counts
        """
        # Get NN evaluation for root position
        policy_probs, value = self._evaluate(board, history)

        # Initialize root
        root = MCTSNode(prior_prob=0.0)

        # Expand root with legal moves and prior probabilities based on NN
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            move_idx = encode_move(move)
            root.children[move] = MCTSNode(prior_prob=policy_probs[move_idx].item())

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(board.copy(), history.copy(), root)

        # Extract policy from visit counts
        policy = torch.zeros(4672)
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for move, child in root.children.items():
                move_idx = encode_move(move)
                policy[move_idx] = child.visit_count / total_visits

        return policy

    def _simulate(self, board: chess.Board, history: list[chess.Board], node: MCTSNode) -> float:
        """Run one MCTS simulation"""

        # Check if game is over
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0
            elif result == "0-1":
                return -1.0
            else:
                return 0.0

        # If leaf node, expand with NN and return value
        if not node.children:
            # Evaluate with neural network
            policy_probs, value = self._evaluate(board, history)

            # Expand node with legal moves
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                move_idx = encode_move(move)
                node.children[move] = MCTSNode(prior_prob=policy_probs[move_idx].item())

            return value

        # Select best child using UCB
        best_move, best_child = max(
            node.children.items(), key=lambda item: item[1].ucb_score(node.visit_count, self.c_puct)
        )

        # Make move and recurse
        board.push(best_move)
        history.append(board.copy())

        value = -self._simulate(board, history, best_child)  # Negate for opponent

        # Backpropagate
        best_child.visit_count += 1
        best_child.value_sum += value

        return value


def test_mcts():
    """Test MCTS with cleaner design"""
    print("Testing MCTS with method-based evaluation...")

    # Create model
    model = AlphaZeroNet(num_res_blocks=5, num_channels=64)
    device = "cpu"

    # Create MCTS
    mcts = MCTS(model, num_simulations=50, device=device)

    # Test position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    history = [chess.Board()]

    # Run MCTS - evaluation happens internally via self._evaluate()
    print("\nRunning MCTS search...")
    policy = mcts.search(board, history)

    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Non-zero moves: {(policy > 0).sum().item()}")

    print("\n✓ MCTS working with cleaner design!")


if __name__ == "__main__":
    test_mcts()
