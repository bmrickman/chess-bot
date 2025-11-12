from dataclasses import dataclass

from src2.policy.mcts.mcts import MCTS, MCTSNode


def test_mcts():
    @dataclass
    class TestNode(MCTSNode[str, str]):
        state: str
        prior_prob: float = 0.0

        def apply_move(self, move: str) -> "TestNode":
            new_state = self.state + move
            return TestNode(state=new_state)

        def is_terminal(self) -> bool:
            return len(self.state) >= 3

        def terminal_value(self) -> float:
            return self.state.count("R") - self.state.count("L")

        def legal_moves(self) -> list[str]:
            return ["L", "R"]

    def evaluate_state(state: str) -> tuple[dict[str, float], float]:
        value = state.count("R") - state.count("L")
        policy = {"L": -1.0, "R": 1.0}
        return policy, float(value)

    mcts = MCTS[str, str](evaluate=evaluate_state, node_type=TestNode, c_puct=1, sims_per_move=1)

    root = TestNode(state=".", prior_prob=0.0)
    mcts.print_tree(root)
    # print("\n\n")
    # ". 0/0"
    assert root.nvisits == 0
    assert root.total_value == 0.0
    assert not root.children
    mcts.simulate(root)
    # mcts.print_tree(root)
    # print("\n\n")
    assert root.nvisits == 1
    assert root.total_value == 0.0
    assert root.children["L"].nvisits == 0
    assert root.children["L"].total_value == 0.0
    assert root.children["R"].nvisits == 0
    assert root.children["R"].total_value == 0.0
    mcts.simulate(root)
    # mcts.print_tree(root)
    # print("\n\n")
    assert root.nvisits == 2
    assert root.total_value == -1.0
    assert root.children["L"].nvisits == 0
    assert root.children["L"].total_value == 0.0
    assert root.children["R"].nvisits == 1
    assert root.children["R"].total_value == 1.0
    assert root.children["R"].children["L"].nvisits == 0
    assert root.children["R"].children["L"].total_value == 0.0
    assert root.children["R"].children["R"].nvisits == 0
    assert root.children["R"].children["R"].total_value == 0.0
    mcts.simulate(root)
    # mcts.print_tree(root)
    assert root.nvisits == 3
    assert root.total_value == 1.0
    assert root.children["L"].nvisits == 0
    assert root.children["L"].total_value == 0.0
    assert root.children["R"].nvisits == 2
    assert root.children["R"].total_value == -1.0
    assert root.children["R"].children["L"].nvisits == 0
    assert root.children["R"].children["L"].total_value == 0.0
    assert root.children["R"].children["R"].nvisits == 1
    assert root.children["R"].children["R"].total_value == 2.0
