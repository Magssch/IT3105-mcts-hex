from typing import Tuple

from TreeNode import TreeNode


class MCTS:

    def search(self, initial_state: Tuple[int]) -> TreeNode:
        self.root = TreeNode(initial_state)
        # ...
        # return leafNode

    def do_rollout(self, node: TreeNode, ANET) -> TreeNode:
        current_node = node
        while not current_node.is_terminal:
            current_node = ANET(current_node)  # <- default policy
            current_node.visits += 1
            # Update ...
        # return finalNode
        pass

    def do_backpropagation(self, node: TreeNode, score: int) -> None:
        child = node
        parent = node.parent
        while parent != None:
            parent.value += child.score
            parent = parent.parent

    def get_normalized_distribution():
        pass

    def add_child_node():
        pass

    def uct(self, node: TreeNode):
        # P_t = UCB1 = average_value + C * sqrt(ln(N_i)/n_i)
        # UCT = w_i / n_i + parameters.UCT_C * sqrt(ln(N_i)/n_i)
        pass
