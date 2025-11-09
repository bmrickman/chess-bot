import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Type

from src2.types import Action, State

# You implement a Node to get an MCTS


class Node[State, Action](ABC):
    state: State
    total_value: float = 0.0
    nvisits: int = 0
    children: dict[Action, State] = {}
    prior_prob: float = 0.0

    def update(self, value: float):
        self.total_value += value
        self.nvisits += 1

    @abstractmethod
    def apply_move(self, move: Action) -> "Node[State, Action]": ...

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def terminal_value(self) -> float: ...

    @abstractmethod
    def legal_moves(self) -> list[Action]: ...


@dataclass
class MCTS[Action, State]:
    evaluate: Callable[[Node], tuple[dict[Action, float], float]]
    node_type: Type[Node[State, Action]]
    c_puct: float
    sims_per_move: int

    # simulation interface
    def _select_best_move_and_child(self, node: Node) -> tuple[Action, Node]:
        total_visits = sum(child.nvisits for child in node.children.values())

        def ucb_score(move: Action, child: Node):
            q = child.total_value / (child.nvisits or 1)
            u = self.c_puct * child.prior_prob * (math.sqrt(total_visits) / (1 + child.nvisits))
            return q + u

        return max(node.children.items(), key=lambda mc: ucb_score(*mc))

    def _simulate(self, node: Node) -> float:
        # Terminal node
        if node.is_terminal():
            value = node.terminal_value()
            node.update(value)  # Update in-place
            return value

        # Leaf node - expand and evaluate
        elif not node.children:
            policy_probs, value = self.evaluate(node)
            node.update(value)
            for move in node.legal_moves():
                node.children[move] = self.node_type(state=node.apply_move(move), prior_prob=policy_probs[move])
            return value
        # recursion
        else:
            _, best_child = self._select_best_move_and_child(node)
            value = self._simulate(best_child)
            value = -value
            node.update(value)
            return value

    # gameplay interface
    def choose_move(self, node: Node):
        for i in range(self.sims_per_move):
            self._simulate(node)
        return self._select_best_move_and_child(node)
