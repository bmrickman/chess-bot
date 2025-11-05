"""
Functional-style MCTS with efficient mutable nodes
"""

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from email import policy
from typing import Dict, List, Optional, Tuple

import chess
import torch
import torch.nn.functional as F

from src.board_encoding import encode_board
from src.move_encoding import encode_move

# ============================================================================
# Mutable Node (efficient!)
# ============================================================================


@dataclass(slots=True)
class Node:
    """
    Mutable node for efficiency

    Still feels functional - you operate on nodes with functions,
    but updates happen in-place for performance
    """

    board: chess.Board
    history: List[str]
    prior_prob: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Node(visits={self.visit_count}, value={get_value(self):.3f}, children={len(self.children)})"


def create_child_node(parent: Node, move: chess.Move, prior_prob: float) -> Node:
    """Create child node from parent"""
    child_history = parent.history[-6:] + [
        parent.board.fen()
    ]  # keep last 6 board states for 3 fold repetition detection
    child_board = parent.board.copy()
    child_board.push(move)

    return Node(board=child_board, history=child_history, prior_prob=prior_prob)


# ============================================================================
# Node Operations (in-place updates!)
# ============================================================================


def get_value(node: Node) -> float:
    if node.visit_count == 0:
        return 0.0
    return node.value_sum / node.visit_count


def update_node(node: Node, value: float) -> None:
    node.visit_count += 1
    node.value_sum += value


def get_ucb_score(node: Node, parent_visits: int, c_puct: float = 1.5) -> float:
    q_value = get_value(node)
    u_value = c_puct * node.prior_prob * math.sqrt(parent_visits) / (1 + node.visit_count)
    return q_value + u_value


def select_best_child(node: Node, c_puct: float = 1.5) -> Tuple[chess.Move, Node]:
    best_move = None
    best_child = None
    best_score = float("-inf")

    for move, child in node.children.items():
        score = get_ucb_score(child, node.visit_count, c_puct)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    assert best_move is not None and best_child is not None
    return best_move, best_child


# ============================================================================
# Neural Network Evaluation
# ============================================================================


def evaluate_node(
    node: Node, inference_request_queue: mp.Queue, inference_response_queue: mp.Queue
) -> Tuple[torch.Tensor, float]:
    """Evaluate node with neural network (pure function)"""
    state = encode_board(node.board, node.history)
    inference_request_queue.put(state)

    inference_response = inference_response_queue.get()
    policy_probs = inference_response["policy"]
    value = inference_response["value"]

    return policy_probs, value


# ============================================================================
# Tree Operations
# ============================================================================


def expand_node(node: Node, policy_probs: torch.Tensor) -> None:
    """
    Expand node with children (mutates node.children!)

    Creates child nodes and adds them to parent
    """
    for move in node.board.legal_moves:
        move_idx = encode_move(move)
        prior_prob = policy_probs[move_idx].item()

        child = create_child_node(node, move, prior_prob)
        node.children[move] = child


def is_terminal(node: Node) -> bool:
    """Check if node is terminal (pure function)"""
    return node.board.is_game_over()


def get_terminal_value(node: Node) -> float:
    """Get value for terminal node (pure function)"""
    result = node.board.result()
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


# ============================================================================
# MCTS Simulation
# ============================================================================


def simulate(node: Node, inference_request_queue: mp.Queue, inference_response_queue: mp.Queue, c_puct: float) -> float:
    # Terminal node
    if is_terminal(node):
        value = get_terminal_value(node)
        update_node(node, value)  # Update in-place
        return value

    # Leaf node - expand and evaluate
    if not node.children:
        policy_probs, value = evaluate_node(node, inference_request_queue, inference_response_queue)
        expand_node(node, policy_probs)  # Mutates node.children
        update_node(node, value)  # Update in-place
        return value

    # Internal node - select and recurse
    best_move, best_child = select_best_child(node, c_puct)

    # Recurse (child updates happen in the recursive call)
    value = simulate(best_child, inference_request_queue, inference_response_queue, c_puct)

    # Negate value for opponent
    value = -value

    # Update this node in-place
    update_node(node, value)

    return value


# ============================================================================
# MCTS Search
# ============================================================================


def search(
    root: Node,
    inference_inference_request_queue: mp.Queue,
    inference_response_queue: mp.Queue,
    num_simulations: int = 800,
    c_puct: float = 1.5,
) -> None:
    for _ in range(num_simulations):
        simulate(root, inference_inference_request_queue, inference_response_queue, c_puct)


# ============================================================================
# Information Extraction
# ============================================================================


def extract_policy(root: Node) -> torch.Tensor:
    """Extract policy tensor for training"""
    policy = torch.zeros(4864)
    total_visits = sum(child.visit_count for child in root.children.values())

    if total_visits > 0:
        for move, child in root.children.items():
            move_idx = encode_move(move)
            policy[move_idx] = child.visit_count / total_visits

    return policy


def get_visit_counts(root: Node) -> Dict[chess.Move, int]:
    """Get visit counts as dict"""
    return {move: child.visit_count for move, child in root.children.items()}


def sample_move(root: Node, temperature: float = 1.0) -> chess.Move:
    """
    Sample move from root with temperature

    Args:
        root: MCTS root node
        temperature: Sampling temperature
            - 0.0: Greedy (always pick best)
            - 1.0: Proportional to visit counts
            - >1.0: Flatten distribution (more exploration)
            - <1.0: Sharpen distribution (prefer best)

    Returns:
        Selected chess move
    """
    if temperature == 0:
        # Greedy: return move with highest visit count
        return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

    # Get policy tensor [4864]
    policy_tensor = extract_policy(root)

    # Extract probabilities for legal moves (children)
    moves = list(root.children.keys())
    probs = []
    for move in moves:
        move_idx = encode_move(move)
        probs.append(policy_tensor[move_idx].item())

    # Apply temperature
    probs_tensor = torch.tensor(probs)
    tempered = probs_tensor ** (1 / temperature)
    tempered = tempered / tempered.sum()

    # Sample from tempered distribution
    move_idx = torch.multinomial(tempered, 1).item()
    return moves[int(move_idx)]
