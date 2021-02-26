from typing import Tuple

from TreeNode import TreeNode
from src import parameters
from math import sqrt, log
class MCT:

    def search(self, initial_state: Tuple[int, ...]) -> TreeNode:
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

    def do_backpropagation(self, node: TreeNode, score: int) -> None:
        child = node
        parent = node.parent
        while parent != None:
            parent.score += child.score
            parent = parent.parent

    def get_normalized_distribution(self) -> Tuple[int, ...]:
        pass

    def add_child_node(self, node: TreeNode) -> None:
        pass

    def UCT(self, node: TreeNode) -> float:
        return node.value + parameters.UCT_C * sqrt(2 * log(self.root.visit_count) / node.visit_count)
