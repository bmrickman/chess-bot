import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generic, Type

from src2.types import Action, State


@dataclass
class MCTSNode(Generic[State, Action], ABC):
    state: State
    prior_prob: float

    total_value: float = 0.0
    nvisits: int = 0
    children: dict[Action, "MCTSNode[State, Action]"] = field(default_factory=dict)

    def update(self, value: float) -> None:
        self.total_value += value
        self.nvisits += 1

    @abstractmethod
    def apply_move(self, move: Action) -> "MCTSNode[State, Action]": ...

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def terminal_value(self) -> float: ...

    @abstractmethod
    def legal_moves(self) -> list[Action]: ...


@dataclass
class MCTS(Generic[Action, State]):
    evaluate: Callable[[State], tuple[dict[Action, float], float]]
    node_type: Type[MCTSNode[State, Action]]
    c_puct: float
    sims_per_move: int

    def _ucb_score(self, child: MCTSNode, parent: MCTSNode) -> float:
        q = child.total_value / (child.nvisits or 1)
        u = self.c_puct * child.prior_prob * (math.sqrt(parent.nvisits) / (1 + child.nvisits))
        return q + u

    # simulation interface
    def select_best_move_and_child(self, node: MCTSNode) -> tuple[Action, MCTSNode]:
        return max(node.children.items(), key=lambda mc: self._ucb_score(mc[1], node))

    def simulate(self, node: MCTSNode) -> None:
        path = [node]
        # walk down to leaf node
        while node.children:
            _, node = self.select_best_move_and_child(node)
            path.append(node)
        # evaluate leaf
        if node.is_terminal():
            value = node.terminal_value()
        else:
            policy, value = self.evaluate(node.state)
            for move in node.legal_moves():
                node.children[move] = self.node_type(state=node.apply_move(move).state, prior_prob=policy[move])
        # Backpropagation
        for n in reversed(path):
            n.update(value)
            value = -value

    def print_tree(
        self, node: MCTSNode[State, Action], parent_node: MCTSNode[State, Action] | None = None, depth: int = 0
    ):
        indent = "    " * depth
        print(
            f"{indent}- s:{node.state}, vis:{node.nvisits}, val: {node.total_value}, pp:{node.prior_prob}, ucb: {self._ucb_score(node, parent_node) if parent_node else 'N/A'}"
        )

        for action, child in node.children.items():
            print(f"{indent}  Action: {action}")
            self.print_tree(child, node, depth + 1)
